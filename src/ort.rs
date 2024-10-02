use crate::phi3v::{image_process::Phi3VImageProcessor, text_process::Phi3VTextProcessor};
use anyhow::Result;
use image::DynamicImage;
use ndarray::{Array, Array2, Array3, Array4, ArrayView, Axis, Ix3};
use ort::{
    // GraphOptimizationLevel,
    CPUExecutionProvider,
    Session,
    SessionInputValue,
    Tensor,
};
use std::borrow::Cow;

fn get_image_embedding(img: &Option<DynamicImage>) -> Result<Array3<f32>> {
    let visual_features = if let Some(img) = img {
        let image_processor = Phi3VImageProcessor::new();
        let result = image_processor.preprocess(img)?;
        println!(
            "image process result, num_img_tokens: {num_img_tokens:?}, pixel_values: {pixel_values:?}, image_sizes: {image_sizes:?}",
            num_img_tokens = result.num_img_tokens,
            pixel_values = result.pixel_values.shape(),
            image_sizes = result.image_sizes.shape(),
        );
        let model_inputs = ort::inputs![
            "pixel_values" => result.pixel_values,
            "image_sizes" => result.image_sizes,
        ]?;
        let model = Session::builder()?
            // .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_intra_threads(4)?
            .commit_from_file("models/phi-3-vision/phi-3-v-128k-instruct-vision.onnx")?;
        let outputs = model.run(model_inputs)?;
        let predictions_view: ArrayView<f32, _> =
            outputs["visual_features"].try_extract_tensor::<f32>()?;
        let predictions = predictions_view.into_dimensionality::<Ix3>()?.to_owned();
        predictions
    } else {
        Array::zeros((1, 0, 0))
    };
    // println!("visual_features {:?}", visual_features);
    Ok(visual_features)
}

fn get_text_embedding(
    text: &str,
    text_processor: &Phi3VTextProcessor,
) -> Result<(Array3<f32>, Array2<i64>)> {
    let (input_ids, attention_mask) = text_processor.preprocess(&text)?;
    println!("input_ids: {:?}", input_ids);
    println!("attention_mask: {:?}", attention_mask);
    let model_inputs = ort::inputs![
        "input_ids" => input_ids,
    ]?;

    let model = Session::builder()?
        // .with_optimization_level(GraphOptimizationLevel::Disable)?
        .with_execution_providers([CPUExecutionProvider::default().build()])?
        .with_intra_threads(4)?
        .commit_from_file("models/phi-3-vision/phi-3-v-128k-instruct-text-embedding.onnx")?;

    let outputs = model.run(model_inputs)?;
    let predictions_view: ArrayView<f32, _> =
        outputs["inputs_embeds"].try_extract_tensor::<f32>()?;
    println!("predictions_view: {:?}", predictions_view);
    // {
    //     let shape = predictions_view.shape();
    //     for i in 0..shape[0] {
    //         for j in 0..shape[1] {
    //             let row = predictions_view.slice(ndarray::s![i, j, ..]);
    //             println!("predictions[{}][{}]: {:?}", i, j, row);
    //         }
    //     }
    // }
    let text_inputs_embeds = predictions_view.into_dimensionality::<Ix3>()?.to_owned();
    // println!("text_inputs_embeds {:?}", text_inputs_embeds);
    // let shape = predictions_view.shape();
    // if shape.len() != 3 {
    //     return Err(anyhow::anyhow!("Expected 3D array, got {}D", shape.len()));
    // }
    // let predictions: Array3<f32> = Array::from_shape_vec(
    //     (shape[0], shape[1], shape[2]),
    //     predictions_view.iter().cloned().collect(),
    // )
    // .map_err(|_| anyhow::anyhow!("Failed to create Array3 from predictions"))?;
    Ok((text_inputs_embeds, attention_mask))
}

pub async fn run() -> Result<()> {
    // let img: Option<DynamicImage> = Some(image::open("./models/frames/4000.jpg").unwrap()); //./models/20240923-173209.jpeg
    let img: Option<DynamicImage> = None;
    let visual_features = get_image_embedding(&img)?;

    let text_processor = Phi3VTextProcessor::new("models/phi-3-vision/tokenizer.json")?;
    let text = if let Some(_img) = &img {
        format!(
            "<|image_1|>\n{prompt}",
            prompt = "Describe the image in detail."
        )
    } else {
        format!("{prompt}", prompt = "Who are you?")
    };
    let (text_inputs_embeds, text_attention_mask) = get_text_embedding(&text, &text_processor)?;

    Ok(())
}
