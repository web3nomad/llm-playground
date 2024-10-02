use crate::phi3v::{image_process::Phi3VImageProcessor, text_process::Phi3VTextProcessor};
use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, Array2, Array3, Array4, ArrayView, Ix3, Ix4};
use ort::{
    // CPUExecutionProvider,
    // CoreMLExecutionProvider,
    // GraphOptimizationLevel,
    Session,
    SessionInputValue,
    Tensor,
};
use std::borrow::Cow;

fn get_image_embedding(model: &Session, img: &Option<DynamicImage>) -> Result<Array3<f32>> {
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

fn get_text_embedding(model: &Session, input_ids: &Array2<i64>) -> Result<Array3<f32>> {
    let model_inputs = ort::inputs![
        "input_ids" => input_ids.to_owned(),
    ]?;
    let outputs = model.run(model_inputs)?;
    let inputs_embeds_view: ArrayView<f32, _> =
        outputs["inputs_embeds"].try_extract_tensor::<f32>()?;
    // println!("inputs_embeds_view: {:?}", inputs_embeds_view);
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
    let text_processor = Phi3VTextProcessor::new("models/phi-3-vision/tokenizer.json")?;
    let text_embedding_model = Session::builder()?
        // .with_optimization_level(GraphOptimizationLevel::Level3)?
        // .with_execution_providers([CoreMLExecutionProvider::default().build()])?
        // .with_intra_threads(16)?
        // .with_inter_threads(16)?
        .commit_from_file("models/phi-3-vision/phi-3-v-128k-instruct-text-embedding.onnx")?;
    let vision_model = Session::builder()?
        // .with_optimization_level(GraphOptimizationLevel::Level3)?
        // .with_execution_providers([CoreMLExecutionProvider::default().build()])?
        // .with_intra_threads(16)?
        // .with_inter_threads(16)?
        .commit_from_file("models/phi-3-vision/phi-3-v-128k-instruct-vision.onnx")?;
    let generation_model = Session::builder()?
        // .with_optimization_level(GraphOptimizationLevel::Level3)?
        // .with_execution_providers([CoreMLExecutionProvider::default().build()])?
        // .with_intra_threads(16)?
        // .with_inter_threads(16)?
        .commit_from_file("models/phi-3-vision/phi-3-v-128k-instruct-text.onnx")?;

    let img: Option<DynamicImage> = Some(image::open("./models/frames/4000.jpg").unwrap());
    // let img: Option<DynamicImage> = None;
    let visual_features = get_image_embedding(&vision_model, &img)?;
    println!("visual_features {:?}", visual_features);

    let (mut inputs_embeds, mut attention_mask) = {
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
        let inputs_embeds = get_text_embedding(&text_embedding_model, &input_ids)?;
        println!("inputs_embeds: {:?}", inputs_embeds);

        if let Some(_img) = &img {
            let mut combined_embeds = Array3::zeros((
                1,
                inputs_embeds.shape()[1] + visual_features.shape()[1],
                inputs_embeds.shape()[2],
            ));

            // 找到 <|image_1|> 标记的位置, 在 <|user|> (id 是 32010) 后面
            let image_token_position = input_ids.iter().position(|&id| id == 32010).unwrap_or(0);

            // 复制文本嵌入直到 <|image_1|> 标记
            combined_embeds
                .slice_mut(s![.., ..image_token_position, ..])
                .assign(&inputs_embeds.slice(s![.., ..image_token_position, ..]));

            // 插入视觉特征
            combined_embeds
                .slice_mut(s![
                    ..,
                    image_token_position..(image_token_position + visual_features.shape()[1]),
                    ..
                ])
                .assign(&visual_features);

            // 复制剩余的文本嵌入
            combined_embeds
                .slice_mut(s![
                    ..,
                    (image_token_position + visual_features.shape()[1])..,
                    ..
                ])
                .assign(&inputs_embeds.slice(s![.., image_token_position.., ..]));

            // 更新 attention_mask
            let mut new_attention_mask =
                Array2::ones((1, attention_mask.shape()[1] + visual_features.shape()[1]));
            new_attention_mask
                .slice_mut(s![.., ..image_token_position])
                .assign(&attention_mask.slice(s![.., ..image_token_position]));
            new_attention_mask
                .slice_mut(s![
                    ..,
                    (image_token_position + visual_features.shape()[1])..
                ])
                .assign(&attention_mask.slice(s![.., image_token_position..]));

            (combined_embeds, new_attention_mask)
        } else {
            (inputs_embeds, attention_mask)
        }
    };

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
            "inputs_embeds" => inputs_embeds.clone(),
            "attention_mask" => attention_mask.clone(),
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

        let outputs = generation_model.run(model_inputs)?;

        let logits: ArrayView<f32, _> = outputs["logits"].try_extract_tensor::<f32>()?;
        let logits = logits.into_dimensionality::<Ix3>()?;

        // 获取最后一个 token 的 logits
        let last_token_logits = logits.slice(s![0, -1, ..]);
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
        let new_token_embed = get_text_embedding(&text_embedding_model, &new_token_id)?;
        // 合并新的 token 嵌入与之前的嵌入
        let mut combined_embeds = Array3::zeros((
            inputs_embeds.shape()[0],
            inputs_embeds.shape()[1] + 1,
            inputs_embeds.shape()[2],
        ));
        combined_embeds
            .slice_mut(s![.., ..inputs_embeds.shape()[1], ..])
            .assign(&inputs_embeds);
        combined_embeds
            .slice_mut(s![.., inputs_embeds.shape()[1].., ..])
            .assign(&new_token_embed);

        inputs_embeds = combined_embeds;
        attention_mask = Array2::ones((1, attention_mask.shape()[1] + 1));

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

    Ok(())
}
