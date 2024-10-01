use std::borrow::Cow;

use crate::phi3v::{image_process::Phi3VImageProcessor, text_process::Phi3VTextProcessor};
use anyhow::Result;
use ndarray::{Array, Array3, ArrayView, Axis};
use ort::{GraphOptimizationLevel, Session, SessionInputValue, Tensor};

pub async fn run() -> Result<()> {
    let visual_features = {
        let image_processor = Phi3VImageProcessor::new();
        // let img = image::open("./models/20240923-173209.jpeg").unwrap();
        let img = image::open("./models/frames/4000.jpg").unwrap();
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

    let text_processor = Phi3VTextProcessor::new("models/phi-3-vision/tokenizer.json");
    let (text_inputs_embeds, attention_mask) = {
        // 参考了 https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi3v.py
        let text = format!(
            "<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n",
            prompt = "What is shown in this image?"
        );
        let (input_ids, attention_mask) = text_processor.preprocess(&text)?;
        let model_inputs = ort::inputs![
            "input_ids" => input_ids,
        ]?;
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_intra_threads(4)?
            .commit_from_file("models/phi-3-vision/phi-3-v-128k-instruct-text-embedding.onnx")?;
        let outputs = model.run(model_inputs)?;
        let predictions_view: ArrayView<f32, _> =
            outputs["inputs_embeds"].try_extract_tensor::<f32>()?;
        let shape = predictions_view.shape();
        if shape.len() != 3 {
            return Err(anyhow::anyhow!("Expected 3D array, got {}D", shape.len()));
        }
        let predictions: Array3<f32> = Array::from_shape_vec(
            (shape[0], shape[1], shape[2]),
            predictions_view.iter().cloned().collect(),
        )
        .map_err(|_| anyhow::anyhow!("Failed to create Array3 from predictions"))?;
        (predictions, attention_mask)
    };

    println!("text_inputs_embeds {:?}", text_inputs_embeds);
    println!("attention_mask {:?}", attention_mask);

    let logits = {
        // 确保两个数组在批次维度和嵌入维度上匹配
        assert_eq!(text_inputs_embeds.shape()[0], visual_features.shape()[0]);
        assert_eq!(text_inputs_embeds.shape()[2], visual_features.shape()[2]);

        // 在时间维度（通常是第二个维度，索引为 1）上拼接
        let combined_features = ndarray::concatenate(
            Axis(1),
            &[text_inputs_embeds.view(), visual_features.view()],
        )?;

        // 更新 attention_mask
        let visual_attention = Array::ones((attention_mask.shape()[0], visual_features.shape()[1]));
        let combined_attention_mask =
            ndarray::concatenate(Axis(1), &[attention_mask.view(), visual_attention.view()])?;

        let mut model_inputs = ort::inputs![
            "inputs_embeds" => combined_features,
            "attention_mask" => combined_attention_mask,
            // "inputs_embeds" => text_inputs_embeds,
            // "attention_mask" => attention_mask,
        ]?;

        let batch_size = 1;
        let num_layers = 32; // This should match your model's configuration
        let num_heads = 32; // This should match your model's configuration
        let sequence_length = 0; // Start with 0 for the first run
        let head_size = 96; // This should match your model's configuration

        // Create empty past key and value tensors
        let past_key_shape = vec![batch_size, num_heads, sequence_length, head_size];
        let past_value_shape = past_key_shape.clone();

        let empty_past_state = vec![0.0f32; past_key_shape.iter().product()];
        // Add past key-value pairs for each layer
        for i in 0..num_layers {
            let past_key_tensor = Tensor::from_array(Array::from_shape_vec(
                past_key_shape.clone(),
                empty_past_state.clone(),
            )?)?;
            let past_value_tensor = Tensor::from_array(Array::from_shape_vec(
                past_value_shape.clone(),
                empty_past_state.clone(),
            )?)?;

            let key: Cow<'_, str> = format!("past_key_values.{}.key", i).into();
            let val: SessionInputValue<'_> = past_key_tensor.into();
            model_inputs.push((key, val));
            let key: Cow<'_, str> = format!("past_key_values.{}.value", i).into();
            let val: SessionInputValue<'_> = past_value_tensor.into();
            model_inputs.push((key, val));
        }

        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_intra_threads(4)?
            .commit_from_file("models/phi-3-vision/phi-3-v-128k-instruct-text.onnx")?;
        let outputs = model.run(model_inputs)?;
        println!("outputs {:?}", outputs);
        let predictions_view: ArrayView<f32, _> = outputs["logits"].try_extract_tensor::<f32>()?;
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
    println!("logits {:?}", logits);

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
    println!("Generated text: {}", output_text);

    Ok(())
}
