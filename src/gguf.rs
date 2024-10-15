use super::qllava::{
    format_prompt, load_qllama_model, HFPreProcessorConfig, ImageProcessor, QLlama,
};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::{
    generation::LogitsProcessor,
    models::{clip::vision_model::ClipVisionConfig, quantized_llama},
};
use std::io::Write;
use tokenizers::Tokenizer;

pub fn generate(device: &Device, prompt_str: &str, mut qllama: QLlama) -> anyhow::Result<()> {
    let mut tos = TokenOutputStream::new(qllama.tokenizer);
    let to_sample = qllama.sample_len.saturating_sub(1);

    let prompt_tokens = {
        let tokens = tos
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg)?;
        let prompt_tokens = tokens.get_ids().to_vec();
        if prompt_tokens.len() + to_sample > quantized_llama::MAX_SEQ_LEN - 10 {
            let to_remove = prompt_tokens.len() + to_sample + 10 - quantized_llama::MAX_SEQ_LEN;
            prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
        } else {
            prompt_tokens
        }
    };

    let mut logits_processor = LogitsProcessor::from_sampling(qllama.seed, qllama.sampling);

    let mut next_token = {
        let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
        let logits = qllama.model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?;
        logits_processor.sample(&logits)?
    };

    let mut all_tokens = vec![];
    all_tokens.push(next_token);
    if let Some(t) = tos.next_token(next_token)? {
        print!("{t}");
        std::io::stdout().flush()?;
    }

    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = qllama.model.forward(&input, prompt_tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        let logits = if qllama.repeat_penalty == 1. {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(qllama.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                qllama.repeat_penalty,
                &all_tokens[start_at..],
            )?
        };
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
        if next_token == qllama.eos_token_id {
            break;
        };
    }

    if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? {
        print!("{rest}");
    }
    std::io::stdout().flush()?;
    println!("");

    Ok(())
}

fn load_image_to_tensor(
    device: &Device,
    image_file_path: &str,
    preprocessor_config_file_path: &str,
) -> anyhow::Result<((u32, u32), Tensor)> {
    let preprocessor_config: HFPreProcessorConfig =
        serde_json::from_slice(&std::fs::read(preprocessor_config_file_path)?)?;
    let image_processor = ImageProcessor::from_hf_preprocessor_config(&preprocessor_config);

    let img = image::ImageReader::open(image_file_path)?.decode()?;
    let image_tensor = image_processor.preprocess(&img)?.unsqueeze(0)?;
    let image_tensor = image_tensor.to_dtype(DType::F16)?.to_device(&device)?;

    println!("Image size: {:?}", (img.width(), img.height()));
    println!("Image tensor: {:?}", image_tensor);

    Ok(((img.width(), img.height()), image_tensor))
}

fn load_clip() -> anyhow::Result<(ClipVisionConfig,)> {
    let clip_vision_config = ClipVisionConfig::clip_vit_large_patch14_336();
    Ok((clip_vision_config,))
}

fn duplicate_vec<T>(vec: &[T], n: usize) -> Vec<T>
where
    T: Clone,
{
    let mut res = Vec::new();
    for _ in 0..n {
        res.extend(vec.to_owned());
    }
    res
}

fn insert_separator<T>(x: Vec<Vec<T>>, sep: Vec<T>) -> Vec<Vec<T>>
where
    T: Clone,
{
    let sep = vec![sep];
    let sep = duplicate_vec(&sep, x.len());
    let mut res = x
        .iter()
        .zip(sep.iter())
        .flat_map(|(x, y)| vec![x.clone(), y.clone()])
        .collect::<Vec<Vec<T>>>();
    res.pop();
    res
}

fn tokenizer_image_token(
    prompt: &str,
    tokenizer: &Tokenizer,
    image_token_index: i64,
    bos_token_id: i64,
) -> anyhow::Result<Tensor> {
    let prompt_chunks = prompt
        .split("<image>")
        .map(|s| {
            tokenizer
                .encode(s, true)
                .unwrap()
                .get_ids()
                .to_vec()
                .iter()
                .map(|x| *x as i64)
                .collect()
        })
        .collect::<Vec<Vec<i64>>>();
    let mut input_ids = Vec::new();
    let mut offset = 0;
    if !prompt_chunks.is_empty()
        && !prompt_chunks[0].is_empty()
        && prompt_chunks[0][0] == bos_token_id
    {
        offset = 1;
        input_ids.push(prompt_chunks[0][0]);
    }

    for x in insert_separator(
        prompt_chunks,
        duplicate_vec(&[image_token_index], offset + 1),
    )
    .iter()
    {
        input_ids.extend(x[1..].to_vec())
    }
    println!("input_ids: {:?}", input_ids);
    let input_len = input_ids.len();
    Tensor::from_vec(input_ids, (1, input_len), &Device::Cpu).map_err(anyhow::Error::msg)
}

fn encode_images(x: &Tensor) -> anyhow::Result<Tensor> {
    // let image_features = self.clip_vision_tower.forward(x)?;
    // let image_features = self.mm_projector.forward(&image_features)?;
    // Ok(image_features)
    Ok(x.to_owned())
}

fn prepare_inputs_labels_for_multimodal(
    device: &Device,
    qllama: &QLlama,
    input_ids: &Tensor,
    images: &[Tensor],
    image_sizes: &[(u32, u32)],
    image_token_index: i64,
) -> anyhow::Result<Tensor> {
    let concat_images = Tensor::cat(images, 0)?;
    let image_features_together = encode_images(&concat_images)?;
    let split_sizes = images
        .iter()
        .map(|x| x.shape().dims()[0])
        .collect::<Vec<usize>>();
    // can be replaced by split
    let mut index_pos = 0;
    let mut image_features = Vec::new();
    for split_size in split_sizes.iter() {
        image_features.push(image_features_together.i(index_pos..index_pos + (*split_size))?);
        index_pos += *split_size;
    }

    let image_features = image_features
        .iter()
        .map(|x| x.flatten(0, 1).unwrap())
        .collect::<Vec<Tensor>>();

    let input_ids_vec = input_ids.squeeze(0)?.to_vec1::<i64>()?;
    let mut image_indices = {
        let mut image_indices = vec![0_i64];
        image_indices.extend(
            input_ids_vec
                .iter()
                .enumerate()
                .filter_map(|(i, x)| {
                    if *x == image_token_index {
                        Some(i as i64)
                    } else {
                        None
                    }
                })
                .collect::<Vec<i64>>(),
        );
        image_indices
    };

    // if image_indices.len() == 1 {
    //     //no image, only [0],
    //     return self.llama.embed(input_ids);
    // }

    let input_ids_noim = input_ids_vec
        .iter()
        .filter_map(|x| {
            if *x != image_token_index {
                Some(*x)
            } else {
                None
            }
        })
        .collect::<Vec<i64>>();
    let input_ids_noim_len = input_ids_noim.len();
    image_indices.push((input_ids_noim_len) as i64);
    let input_ids_noim = Tensor::from_vec(input_ids_noim, input_ids_noim_len, &device)?;
    println!("input_ids_noim: {:?}", input_ids_noim);
    let cur_input_embeds = qllama.embed(&input_ids_noim)?;
    println!("cur_input_embeds: {:?}", cur_input_embeds);
    // can be replace by split if it is implemented in candle
    let input_embed_no_ims = {
        let mut input_embeds = Vec::new();
        for i in 0..image_indices.len() - 1 {
            let start = (image_indices[i]) as usize;
            let end = image_indices[i + 1] as usize;
            input_embeds.push(cur_input_embeds.i((start..end, ..))?)
        }
        input_embeds
    };

    // continue

    let tensor = Tensor::cat(images, 0)?;
    Ok(tensor)
}

pub async fn run() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;
    let qllama = load_qllama_model(
        &device,
        "models/llava-phi-3/llava-phi-3-mini-int4.gguf",
        "models/llava-phi-3/tokenizer.json",
    )?;
    let prompt_str = format_prompt("Describe this image.");
    println!("{}", &prompt_str);

    let (image_size, image_tensor) = load_image_to_tensor(
        &device,
        "models/20240923-173209.jpeg",
        "models/llava-phi-3/preprocessor_config.json",
    )?;

    let (clip_vision_config,) = load_clip()?;

    let image_token_index: i64 = 32038; // see tokenizer.json
    let bos_token_id: i64 = 1;
    let tokens = tokenizer_image_token(
        prompt_str.as_str(),
        &qllama.tokenizer,
        image_token_index,
        bos_token_id,
    )?;

    let mut input_embeds = prepare_inputs_labels_for_multimodal(
        &device,
        &qllama,
        &tokens,
        &[image_tensor],
        &[image_size],
        image_token_index,
    )?;

    generate(&device, prompt_str.as_str(), qllama)?;

    Ok(())
}
