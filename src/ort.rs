use anyhow::Result;
use image::RgbImage;
use ndarray::{Array, Array3, Array5};
use ort::{GraphOptimizationLevel, Session};

const TARGET_IMAGE_SIZE: u32 = 224;

pub fn preprocess_rgb8_image(image: &RgbImage) -> anyhow::Result<Array5<f32>> {
    // resize image
    let (w, h) = image.dimensions();
    let w = w as f32;
    let h = h as f32;
    let (w, h) = if w < h {
        (
            TARGET_IMAGE_SIZE,
            ((TARGET_IMAGE_SIZE as f32) * h / w) as u32,
        )
    } else {
        (
            ((TARGET_IMAGE_SIZE as f32) * w / h) as u32,
            TARGET_IMAGE_SIZE,
        )
    };

    let mut image = image::imageops::resize(image, w, h, image::imageops::FilterType::CatmullRom);

    // center crop the image
    let left = (w - TARGET_IMAGE_SIZE) / 2;
    let top = (h - TARGET_IMAGE_SIZE) / 2;
    let image = image::imageops::crop(&mut image, left, top, TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE)
        .to_image();

    // normalize according to CLIP
    let mut array = Array3::zeros((3, TARGET_IMAGE_SIZE as usize, TARGET_IMAGE_SIZE as usize));

    for i in 0..TARGET_IMAGE_SIZE {
        for j in 0..TARGET_IMAGE_SIZE {
            let p = image.get_pixel(j, i);

            array[[0, i as usize, j as usize]] = (p[0] as f32 / 255.0 - 0.48145466) / 0.26862954;
            array[[1, i as usize, j as usize]] = (p[1] as f32 / 255.0 - 0.4578275) / 0.26130258;
            array[[2, i as usize, j as usize]] = (p[2] as f32 / 255.0 - 0.40821073) / 0.27577711;
        }
    }

    let reshaped = array.into_shape_with_order((1, 16, 3, 12, 12))?;

    Ok(reshaped)
}

pub async fn run() -> Result<()> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Disable)?
        .with_intra_threads(1)?
        .commit_from_file("models/phi-3-v-128k-instruct-vision.onnx")?;
    // .commit_from_url("https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cpu/tree/main/cpu-int4-rtn-block-32-acc-level-4")?;

    let img = image::open("./models/20240923-173209.jpeg").unwrap();
    let img = preprocess_rgb8_image(&img.to_rgb8())?;

    // for input in model.inputs {
    //     println!("Input name: {}", input.name);
    //     println!("Input type: {:?}", input.input_type);
    // }

    let pixel_values = img; //.insert_axis(Axis(0)).insert_axis(Axis(0));
    let image_sizes = Array::from_shape_vec(
        (1, 2),
        vec![TARGET_IMAGE_SIZE as i64, TARGET_IMAGE_SIZE as i64],
    )
    .unwrap();

    println!("{:?}", &pixel_values.shape());
    println!("{:?}", &image_sizes.shape());

    let model_inputs = ort::inputs![
        "pixel_values" => pixel_values,
        "image_sizes" => image_sizes
    ]?;
    let outputs = model.run(model_inputs)?;

    println!("{:?}", outputs);

    // let predictions = outputs["output0"].try_extract_tensor::<f32>()?;

    Ok(())
}
