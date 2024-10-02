use crate::phi3v::{image_process::Phi3VImageProcessor, text_process::Phi3VTextProcessor};
use anyhow::Result;
use image::DynamicImage;
use ndarray::{Array, Array2, Array3, Array4, ArrayView, Ix3, Ix4};
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
    Ok(visual_features)
}

fn get_text_embedding(input_ids: &Array2<i64>) -> Result<Array3<f32>> {
    let model_inputs = ort::inputs![
        "input_ids" => input_ids.to_owned(),
    ]?;

    let model = Session::builder()?
        // .with_optimization_level(GraphOptimizationLevel::Disable)?
        .with_execution_providers([CPUExecutionProvider::default().build()])?
        .with_intra_threads(4)?
        .commit_from_file("models/phi-3-vision/phi-3-v-128k-instruct-text-embedding.onnx")?;

    let outputs = model.run(model_inputs)?;
    let inputs_embeds_view: ArrayView<f32, _> =
        outputs["inputs_embeds"].try_extract_tensor::<f32>()?;
    // println!("predictions_view: {:?}", predictions_view);
    // {
    //     let shape = predictions_view.shape();
    //     for i in 0..shape[0] {
    //         for j in 0..shape[1] {
    //             let row = predictions_view.slice(ndarray::s![i, j, ..]);
    //             println!("predictions[{}][{}]: {:?}", i, j, row);
    //         }
    //     }
    // }
    let inputs_embeds = inputs_embeds_view.into_dimensionality::<Ix3>()?.to_owned();
    // println!("inputs_embeds {:?}", inputs_embeds);
    // let shape = inputs_embeds_view.shape();
    // if shape.len() != 3 {
    //     return Err(anyhow::anyhow!("Expected 3D array, got {}D", shape.len()));
    // }
    // let inputs_embeds: Array3<f32> = Array::from_shape_vec(
    //     (shape[0], shape[1], shape[2]),
    //     inputs_embeds_view.iter().cloned().collect(),
    // )
    // .map_err(|_| anyhow::anyhow!("Failed to create Array3 from predictions"))?;
    Ok(inputs_embeds)
}

pub async fn run() -> Result<()> {
    // let img: Option<DynamicImage> = Some(image::open("./models/frames/4000.jpg").unwrap()); //./models/20240923-173209.jpeg
    let img: Option<DynamicImage> = None;
    let _visual_features = get_image_embedding(&img)?;
    // println!("visual_features {:?}", visual_features);

    let text_processor = Phi3VTextProcessor::new("models/phi-3-vision/tokenizer.json")?;
    let (mut text_inputs_embeds, mut text_attention_mask) = {
        let text = if let Some(_img) = &img {
            format!(
                "<|image_1|>\n{prompt}",
                prompt = "Describe the image in detail."
            )
        } else {
            format!("{prompt}", prompt = "Who are you?")
        };
        let (input_ids, attention_mask) = text_processor.preprocess(&text)?;
        // println!("input_ids: {:?}", input_ids);
        // println!("attention_mask: {:?}", attention_mask);
        let inputs_embeds = get_text_embedding(&input_ids)?;
        (inputs_embeds, attention_mask)
    };

    // 加载文本生成模型
    let model = Session::builder()?
        .with_execution_providers([CPUExecutionProvider::default().build()])?
        .with_intra_threads(4)?
        .commit_from_file("models/phi-3-vision/phi-3-v-128k-instruct-text.onnx")?;

    let mut generated_tokens = Vec::new();
    let max_length = 100; // 设置最大生成长度
    let eos_token_id = 32007; // 根据模型配置设置

    // let batch_size = 1;
    // let num_heads = 32; // This should match your model's configuration
    // let sequence_length = 0; // Start with 0 for the first run
    // let head_size = 96; // This should match your model's configuration
    // 32 layers, each with a key and value
    let mut past_key_values: Vec<Array4<f32>> = vec![Array4::zeros((1, 32, 0, 96)); 64];

    for _ in 0..max_length {
        let mut model_inputs = ort::inputs![
            "inputs_embeds" => text_inputs_embeds.clone(),
            "attention_mask" => text_attention_mask.clone(),
        ]?;
        for i in 0..32 {
            let key: Cow<'_, str> = format!("past_key_values.{}.key", i).into();
            let val: SessionInputValue<'_> =
                Tensor::from_array(past_key_values[i * 2].view())?.into();
            model_inputs.push((key, val));
            let key: Cow<'_, str> = format!("past_key_values.{}.value", i).into();
            let val: SessionInputValue<'_> =
                Tensor::from_array(past_key_values[i * 2 + 1].view())?.into();
            model_inputs.push((key, val));
        }

        let outputs = model.run(model_inputs)?;

        let logits: ArrayView<f32, _> = outputs["logits"].try_extract_tensor::<f32>()?;
        let logits = logits.into_dimensionality::<Ix3>()?;

        // 获取最后一个 token 的 logits
        let last_token_logits = logits.slice(ndarray::s![0, -1, ..]);
        let next_token_id = last_token_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as i64;

        generated_tokens.push(next_token_id);

        if next_token_id == eos_token_id {
            break;
        }

        // 更新 inputs_embeds 和 attention_mask
        let new_token_id = Array2::from_elem((1, 1), next_token_id);
        let new_token_embed = get_text_embedding(&new_token_id)?;
        // 合并新的 token 嵌入与之前的嵌入
        let mut combined_embeds = Array3::zeros((
            text_inputs_embeds.shape()[0],
            text_inputs_embeds.shape()[1] + 1,
            text_inputs_embeds.shape()[2],
        ));
        combined_embeds
            .slice_mut(ndarray::s![.., ..text_inputs_embeds.shape()[1], ..])
            .assign(&text_inputs_embeds);
        combined_embeds
            .slice_mut(ndarray::s![.., text_inputs_embeds.shape()[1].., ..])
            .assign(&new_token_embed);

        text_inputs_embeds = combined_embeds;
        text_attention_mask = Array2::ones((1, text_attention_mask.shape()[1] + 1));

        // 更新 past_key_values
        for i in 0..32 {
            past_key_values[i * 2] = outputs[format!("present.{}.key", i)]
                .try_extract_tensor::<f32>()?
                .into_dimensionality::<Ix4>()?
                .to_owned();
            past_key_values[i * 2 + 1] = outputs[format!("present.{}.value", i)]
                .try_extract_tensor::<f32>()?
                .into_dimensionality::<Ix4>()?
                .to_owned();
        }

        // 解码生成的 tokens
        let generated_text =
            text_processor.decode(&generated_tokens.iter().map(|&id| id as u32).collect())?;
        println!("Generated text: {}", generated_text);
    }

    // // 解码生成的 tokens
    // let generated_text =
    //     text_processor.decode(&generated_tokens.iter().map(|&id| id as u32).collect())?;
    // println!("Generated text: {}", generated_text);

    Ok(())
}
