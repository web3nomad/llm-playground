use super::qllava::{
    format_prompt, load_clip, load_image_to_tensor, load_qllama_model,
    prepare_inputs_labels_for_multimodal, tokenizer_image_token, QLlama,
};
use candle_core::{Device, IndexOp, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::LogitsProcessor;
use std::io::Write;

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
    let prompt_str = format_prompt(r#"Describe the image in less than 50 words."#);
    println!("{}", &prompt_str);

    let (image_size, image_tensor) = load_image_to_tensor(
        &device,
        "models/20240923-173209.jpeg",
        // "models/frames/4000.jpg",
        "models/llava-phi-3/preprocessor_config.json",
    )?;

    let (clip_vision_model, mm_projector) = load_clip(
        &device,
        "models/llava-phi-3/llava-phi-3-mini-mmproj-f16.gguf",
    )?;

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
        &clip_vision_model,
        &mm_projector,
    )?;

    generate(&device, input_embeds, qllama)?;

    Ok(())
}
