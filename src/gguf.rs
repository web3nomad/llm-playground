use super::quantized_llava_phi_3::{QLLaVAPhi3, EOS_TOKEN_ID, IMAGE_TOKEN_ID};
use candle_core::{Device, IndexOp, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::io::Write;

/// The length of the sample to generate (in tokens).
const SAMPLE_LEN: usize = 1000;
const REPEAT_PENALTY: f32 = 1.1;
const REPEAT_LAST_N: usize = 64;
// from models/llava-phi-3/config.json
// let vocab_size: usize = 32064;
// let hidden_size: usize = 3072;

pub fn generate(
    device: &Device,
    mut input_embeds: Tensor,
    QLLaVAPhi3 {
        mut llama,
        tokenizer,
        ..
    }: QLLaVAPhi3,
    seed: u64,
    temperature: f64,
) -> anyhow::Result<()> {
    let mut tos = TokenOutputStream::new(tokenizer.clone());
    let sampling = {
        if temperature <= 0. {
            Sampling::ArgMax
        } else {
            Sampling::All { temperature }
        }
    };
    let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling.clone());
    let mut index_pos = 0;
    let mut all_tokens = vec![];
    for index in 0..SAMPLE_LEN.saturating_sub(1) {
        let (_, input_embeds_len, _) = input_embeds.dims3()?;
        // use kv cache, it is implemented in quantized llama
        let (context_size, context_index) = if index > 0 {
            (1, index_pos)
        } else {
            (input_embeds_len, 0)
        };
        let input = input_embeds.i((.., input_embeds_len.saturating_sub(context_size).., ..))?;
        let logits = llama.forward_input_embed(&input, context_index)?;
        let logits = logits.squeeze(0)?;

        let logits = if REPEAT_PENALTY == 1. {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(REPEAT_LAST_N);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                REPEAT_PENALTY,
                &all_tokens[start_at..],
            )?
        };

        let (_, input_len, _) = input.dims3()?;
        index_pos += input_len;
        let next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        let next_token_tensor = Tensor::from_vec(vec![next_token], 1, &device)?;
        let next_embeds = llama.embed(&next_token_tensor)?.unsqueeze(0)?;
        input_embeds = Tensor::cat(&[input_embeds, next_embeds], 1)?;
        if next_token == EOS_TOKEN_ID {
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
    let qllavaphi3 = QLLaVAPhi3::load(
        &device,
        "models/llava-phi-3/llava-phi-3-mini-int4.gguf",
        "models/llava-phi-3/llava-phi-3-mini-mmproj-f16.gguf",
        "models/llava-phi-3/tokenizer.json",
    )?;

    let prompt_str = QLLaVAPhi3::format_prompt(
        r#"You are an advanced image analysis AI. Examine the image and describe its contents in a concise, text-only format. Focus on identifying: People (including celebrities), actions, objects, animals or pets, nature elements, visual cues of sounds, human speech (if text bubbles present), displayed text (OCR), and brand logos. Provide specific examples for each category found in the image. Only mention categories that are present; omit any that are not detected. Use plain text format without lists or JSON. Be accurate and concise in your descriptions."#,
    );
    println!("{}", &prompt_str);

    let (image_size, image_tensor) = QLLaVAPhi3::load_image(
        &device,
        "models/20240923-173209.jpeg",
        // "models/frames/4000.jpg",
        "models/llava-phi-3/preprocessor_config.json",
    )?;

    let tokens = QLLaVAPhi3::tokenizer_image_token(prompt_str.as_str(), &qllavaphi3.tokenizer)?;

    let input_embeds = qllavaphi3.prepare_inputs_labels_for_multimodal(
        &device,
        &qllavaphi3,
        &tokens,
        &[image_tensor],
        &[image_size],
        IMAGE_TOKEN_ID as i64,
    )?;

    generate(&device, input_embeds, qllavaphi3, 299792458, 0.2)?;

    Ok(())
}
