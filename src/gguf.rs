use super::qllava::{format_prompt, load_qllama_model, QLlama};
use candle_core::{Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::{generation::LogitsProcessor, models::quantized_llama};
use std::io::Write;

pub fn generate(mut qllama: QLlama, device: &Device) -> anyhow::Result<()> {
    let mut tos = TokenOutputStream::new(qllama.tokenizer);
    let to_sample = qllama.sample_len.saturating_sub(1);

    let prompt_tokens = {
        let prompt_str = format_prompt("Describe this image.");
        println!("{}", &prompt_str);
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

pub async fn run() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;
    let qllama = load_qllama_model(&device)?;
    generate(qllama, &device)?;

    Ok(())
}
