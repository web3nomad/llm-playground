use super::qllava::{
    format_prompt, load_clip, load_image_to_tensor, load_qllama_model,
    prepare_inputs_labels_for_multimodal, tokenizer_image_token, QLlama,
};
use candle_core::{Device, IndexOp, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::{generation::LogitsProcessor, models::quantized_llama};
use std::io::Write;

pub fn generate_legacy(
    device: &Device,
    prompt_str: &str,
    mut qllama: QLlama,
) -> anyhow::Result<()> {
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
        // let logits = if qllama.repeat_penalty == 1. {
        //     logits
        // } else {
        //     let start_at = all_tokens.len().saturating_sub(qllama.repeat_last_n);
        //     candle_transformers::utils::apply_repeat_penalty(
        //         &logits,
        //         qllama.repeat_penalty,
        //         &all_tokens[start_at..],
        //     )?
        // };
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

pub fn generate(
    device: &Device,
    mut input_embeds: Tensor,
    mut qllama: QLlama,
) -> anyhow::Result<()> {
    let mut tos = TokenOutputStream::new(qllama.tokenizer.clone());
    let to_sample = qllama.sample_len.saturating_sub(1);

    let mut logits_processor = LogitsProcessor::from_sampling(qllama.seed, qllama.sampling.clone());
    let mut index_pos = 0;
    for index in 0..to_sample {
        let (_, input_embeds_len, _) = input_embeds.dims3()?;
        // use kv cache, it is implemented in quantized llama
        let (context_size, context_index) = if index > 0 {
            (1, index_pos)
        } else {
            (input_embeds_len, 0)
        };
        let input = input_embeds.i((.., input_embeds_len.saturating_sub(context_size).., ..))?;
        let logits = qllama.model.forward_input_embed(&input, context_index)?;
        let logits = logits.squeeze(0)?;
        let (_, input_len, _) = input.dims3()?;
        index_pos += input_len;
        let next_token = logits_processor.sample(&logits)?;
        let next_token_tensor = Tensor::from_vec(vec![next_token], 1, &device)?;
        let next_embeds = qllama.model.embed(&next_token_tensor)?.unsqueeze(0)?;
        input_embeds = Tensor::cat(&[input_embeds, next_embeds], 1)?;
        if next_token == qllama.eos_token_id {
            break;
        }
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? {
        print!("{rest}");
    }
    std::io::stdout().flush()?;
    println!("");

    Ok(())
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

    let (_clip_vision_config,) = load_clip()?;

    let image_token_id: i64 = 32038; // see tokenizer.json
    let bos_token_id: i64 = 1;
    let tokens = tokenizer_image_token(
        prompt_str.as_str(),
        &qllama.tokenizer,
        image_token_id,
        bos_token_id,
    )?;

    let input_embeds = prepare_inputs_labels_for_multimodal(
        &device,
        &qllama,
        &tokens,
        &[image_tensor],
        &[image_size],
        image_token_id,
    )?;

    generate(&device, input_embeds, qllama)?;
    // generate_legacy(&device, prompt_str.as_str(), qllama)?;

    Ok(())
}
