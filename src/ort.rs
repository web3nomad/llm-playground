use crate::phi3v::{image_process::Phi3VImageProcessor, text_process::Phi3VTextProcessor};
use anyhow::Result;
use ndarray::{Array, Array4, ArrayView, Axis, Ix3};
use ort::{
    // GraphOptimizationLevel,
    CPUExecutionProvider,
    Session,
    SessionInputValue,
    Tensor,
};
use std::borrow::Cow;

pub async fn run() -> Result<()> {
    #[allow(unused_assignments)]
    let mut img = Some(image::open("./models/frames/4000.jpg").unwrap()); //./models/20240923-173209.jpeg
    img = None;
    let visual_features = if let Some(img) = &img {
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

    let text_processor = Phi3VTextProcessor::new("models/phi-3-vision/tokenizer.json")?;
    let (text_inputs_embeds, text_attention_mask) = {
        // 参考了 https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi3v.py
        let text = if let Some(_img) = &img {
            format!(
                "<|image_1|>\n{prompt}",
                prompt = "Describe the image in detail."
            )
        } else {
            format!("{prompt}", prompt = "Who are you?")
        };
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
        let predictions = predictions_view.into_dimensionality::<Ix3>()?.to_owned();
        // let shape = predictions_view.shape();
        // if shape.len() != 3 {
        //     return Err(anyhow::anyhow!("Expected 3D array, got {}D", shape.len()));
        // }
        // let predictions: Array3<f32> = Array::from_shape_vec(
        //     (shape[0], shape[1], shape[2]),
        //     predictions_view.iter().cloned().collect(),
        // )
        // .map_err(|_| anyhow::anyhow!("Failed to create Array3 from predictions"))?;
        (predictions, attention_mask)
    };

    // println!("text_inputs_embeds {:?}", text_inputs_embeds);
    // println!("attention_mask {:?}", attention_mask);

    let logits = {
        // // 确保两个数组在批次维度和嵌入维度上匹配
        let (combined_features, combined_attention_mask) = if let Some(_img) = &img {
            assert_eq!(text_inputs_embeds.shape()[0], visual_features.shape()[0]);
            assert_eq!(text_inputs_embeds.shape()[2], visual_features.shape()[2]);
            // 在时间维度（通常是第二个维度，索引为 1）上拼接
            let combined_features = ndarray::concatenate(
                Axis(1),
                &[text_inputs_embeds.view(), visual_features.view()],
            )?;
            // 更新 attention_mask
            let visual_attention =
                Array::ones((text_attention_mask.shape()[0], visual_features.shape()[1]));
            let combined_attention_mask = ndarray::concatenate(
                Axis(1),
                &[text_attention_mask.view(), visual_attention.view()],
            )?;
            (combined_features, combined_attention_mask)
        } else {
            (text_inputs_embeds, text_attention_mask)
        };

        let model_inputs = {
            let mut model_inputs = ort::inputs![
                "inputs_embeds" => combined_features,
                "attention_mask" => combined_attention_mask,
            ]?;
            // let batch_size = 1;
            // let num_heads = 32; // This should match your model's configuration
            // let sequence_length = 0; // Start with 0 for the first run
            // let head_size = 96; // This should match your model's configuration
            let shape = [1, 32, 13, 96];
            let num_layers = 32;
            for i in 0..num_layers {
                let past_key_tensor = Tensor::from_array(Array4::<f32>::zeros(shape))?;
                let past_value_tensor = Tensor::from_array(Array4::<f32>::zeros(shape))?;
                // {
                //     let v = &past_key_tensor.try_extract_tensor::<f32>();
                //     println!("past_key_tensor[{}]: {:?}", i, v);
                //     let v = &past_value_tensor.try_extract_tensor::<f32>();
                //     println!("past_value_shape[{}]: {:?}", i, v);
                // }
                let key: Cow<'_, str> = format!("past_key_values.{}.key", i).into();
                let val: SessionInputValue<'_> = past_key_tensor.into();
                model_inputs.push((key, val));
                let key: Cow<'_, str> = format!("past_key_values.{}.value", i).into();
                let val: SessionInputValue<'_> = past_value_tensor.into();
                model_inputs.push((key, val));
            }
            model_inputs
        };

        let model = Session::builder()?
            // .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_intra_threads(4)?
            .commit_from_file("models/phi-3-vision/phi-3-v-128k-instruct-text.onnx")?;

        let outputs = model.run(model_inputs)?;
        let predictions_view: ArrayView<f32, _> = outputs["logits"].try_extract_tensor::<f32>()?;
        let predictions = predictions_view.into_dimensionality::<Ix3>()?.to_owned();
        predictions
    };
    // println!("logits {:?}", logits);

    // 获取每个位置上概率最高的 token ID
    let predicted_token_ids: Vec<u32> = logits
        .axis_iter(Axis(1))
        .map(|logits_for_position| {
            logits_for_position
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index as u32)
                .unwrap()
        })
        .collect();

    // 将 token ID 转换回文本
    let output_text = text_processor.decode(&predicted_token_ids)?;
    println!("{}", output_text);

    Ok(())
}
