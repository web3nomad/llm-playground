use anyhow::Result;
use image::{DynamicImage, RgbImage};
use ndarray::{s, Array, Array2, Array3, Array4, Array5, ArrayView, Axis};
use ort::{GraphOptimizationLevel, Session};

const OPENAI_CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const OPENAI_CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

pub fn preprocess_rgb8_image(image: &RgbImage) -> Result<(Array5<f32>, Array2<i64>)> {
    // let (width, height) = image.dimensions();
    let image = DynamicImage::ImageRgb8(image.clone());

    // HD transform
    let transformed = hd_transform(&image, 16)?;
    let (new_width, new_height) = transformed.dimensions();

    // Normalize
    let normalized = normalize_image(&transformed, &OPENAI_CLIP_MEAN, &OPENAI_CLIP_STD)?;

    // Create global image
    let global_image = resize_image(&normalized, 336, 336)?;

    // Reshape local patches
    let local_patches = reshape_to_patches(&normalized)?;

    // Concatenate global and local patches
    let mut all_patches = vec![global_image];
    all_patches.extend(local_patches);

    // Pad to max_num_crops
    let padded = pad_to_max_num_crops(&all_patches, 17)?;

    // Add batch dimension
    let pixel_values = padded.insert_axis(Axis(0));

    let image_sizes = Array2::from_shape_vec((1, 2), vec![new_height as i64, new_width as i64])?;

    Ok((pixel_values, image_sizes))
}

fn hd_transform(image: &DynamicImage, _hd_num: u32) -> Result<RgbImage> {
    // Implement HD transform logic here
    // For simplicity, let's just resize to 336x336 for now
    Ok(image
        .resize_exact(336, 336, image::imageops::FilterType::Lanczos3)
        .to_rgb8())
}

fn normalize_image(image: &RgbImage, mean: &[f32; 3], std: &[f32; 3]) -> Result<Array3<f32>> {
    let (width, height) = image.dimensions();
    let mut normalized = Array3::<f32>::zeros((3, height as usize, width as usize));

    for (x, y, pixel) in image.enumerate_pixels() {
        for c in 0..3 {
            normalized[[c, y as usize, x as usize]] = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
        }
    }

    Ok(normalized)
}

fn resize_image(image: &Array3<f32>, _new_height: u32, _new_width: u32) -> Result<Array3<f32>> {
    // Implement image resizing logic here
    // For simplicity, let's just return the original image
    Ok(image.clone())
}

fn reshape_to_patches(image: &Array3<f32>) -> Result<Vec<Array3<f32>>> {
    // Implement patch extraction logic here
    // For simplicity, let's just return the original image as a single patch
    Ok(vec![image.clone()])
}

fn pad_to_max_num_crops(patches: &[Array3<f32>], max_crops: usize) -> Result<Array4<f32>> {
    let (channels, height, width) = patches[0].dim();
    let mut padded = Array4::<f32>::zeros((max_crops, channels, height, width));

    for (i, patch) in patches.iter().enumerate() {
        if i >= max_crops {
            break;
        }
        padded.slice_mut(s![i, .., .., ..]).assign(patch);
    }

    Ok(padded)
}

pub async fn get_image_embedding() -> Result<Array3<f32>> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Disable)?
        .with_intra_threads(4)?
        .commit_from_file("models/phi-3-v-128k-instruct-vision.onnx")?;
    // .commit_from_url("https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cpu/tree/main/cpu-int4-rtn-block-32-acc-level-4")?;
    // for input in model.inputs {
    //     println!("Input name: {}", input.name);
    //     println!("Input type: {:?}", input.input_type);
    // }

    let img = image::open("./models/20240923-173209.jpeg").unwrap();
    let (pixel_values, image_sizes) = preprocess_rgb8_image(&img.to_rgb8())?;
    println!("{:?}", &pixel_values.shape());
    println!("{:?}", &image_sizes.shape());

    let model_inputs = ort::inputs![
        "pixel_values" => pixel_values,
        "image_sizes" => image_sizes,
    ]?;
    let outputs = model.run(model_inputs)?;
    // println!("{:?}", outputs);

    let predictions_view: ArrayView<f32, _> =
        outputs["visual_features"].try_extract_tensor::<f32>()?;
    let shape = predictions_view.shape();
    if shape.len() != 3 {
        return Err(anyhow::anyhow!("Expected 3D array, got {}D", shape.len()));
    }
    let predictions: Array3<f32> = Array::from_shape_vec(
        (shape[0], shape[1], shape[2]),
        predictions_view.iter().cloned().collect(),
    )
    .map_err(|_| anyhow::anyhow!("Failed to create Array3 from predictions"))?;
    println!("{:?}", predictions);

    Ok(predictions)
}

pub async fn run() -> Result<()> {
    let _predictions = get_image_embedding().await?;
    Ok(())
}
