use crate::phi3v::{image_process::Phi3VImageProcessor, text_process::Phi3VTextProcessor};
use anyhow::Result;
use ndarray::{Array, Array3, ArrayView};
use ort::{GraphOptimizationLevel, Session};

pub async fn run() -> Result<()> {
    let visual_features = {
        let image_processor = Phi3VImageProcessor::new();
        let img = image::open("./models/20240923-173209.jpeg").unwrap();
        let result = image_processor.preprocess(&img)?;
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
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_intra_threads(4)?
            .commit_from_file("models/phi-3-vision/phi-3-v-128k-instruct-vision.onnx")?;
        let outputs = model.run(model_inputs)?;
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
        predictions
    };

    println!("visual_features {:?}", visual_features);

    let xxx = {
        let text_processor = Phi3VTextProcessor::new("models/phi-3-vision/tokenizer.json");
        let text = "Describe the image.".to_string();
        let (input_ids, attention_mask) = text_processor.preprocess(&text)?;
        let model_inputs = ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
        ]?;
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_intra_threads(4)?
            .commit_from_file("models/phi-3-vision/phi-3-v-128k-instruct-text-embedding.onnx")?;
        let outputs = model.run(model_inputs)?;
        println!("outputs {:?}", outputs);
        let predictions_view: ArrayView<f32, _> =
            outputs["text_embedding"].try_extract_tensor::<f32>()?;
        let shape = predictions_view.shape();
        println!("text_embedding shape {:?}", shape);
    };

    Ok(())
}
