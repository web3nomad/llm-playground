use crate::phi3v::image_process::Phi3VImageProcessor;
use anyhow::Result;
use ndarray::{Array, Array3, ArrayView};
use ort::{GraphOptimizationLevel, Session};

pub async fn run() -> Result<()> {
    // let _predictions = get_image_embedding().await?;
    //
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Disable)?
        .with_intra_threads(4)?
        .commit_from_file("models/phi-3-v-128k-instruct-vision.onnx")?;

    let predictions = {
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

    println!("{:?}", predictions);
    Ok(())
}
