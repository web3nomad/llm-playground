use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_llama,
};
use std::io::Write;
use tokenizers::Tokenizer;

struct QLlama {
    model: quantized_llama::ModelWeights,
    tokenizer: Tokenizer,
    eos_token_id: u32,
    sampling: Sampling,
    sample_len: usize,
    seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

fn load_qllama_model(device: &Device) -> anyhow::Result<QLlama> {
    let model_path = std::path::PathBuf::from("models/llava-phi-3/llava-phi-3-mini-int4.gguf");
    let mut file = std::fs::File::open(&model_path)?;
    let gguf_content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
    // let gguf_metadata = gguf_content.metadata.clone();
    let model = quantized_llama::ModelWeights::from_gguf(gguf_content, &mut file, &device)?;
    // println!("model built");
    let tokenizer =
        Tokenizer::from_file("models/llava-phi-3/tokenizer.json").map_err(anyhow::Error::msg)?;
    // let eos_token_id = gguf_metadata["tokenizer.ggml.eos_token_id"].to_u32()?;
    // let eos_token = tokenizer.id_to_token(eos_token_id);
    let sampling = {
        let temperature = 0.0; // TODO: make this a parameter
        if temperature <= 0. {
            Sampling::ArgMax
        } else {
            Sampling::All { temperature }
        }
    };
    Ok(QLlama {
        model,
        tokenizer,
        // eos_token_id, // <|endoftext|>
        eos_token_id: 32007, // <|end|>  模型实际输出的是 32007, 而不是 gguf 里配置的 32000
        sampling,
        sample_len: 1000,
        seed: 299792458,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
    })
}

fn format_prompt(prompt: &str) -> String {
    format!(
        "<s><|system|>\n<|end|>\n<|user|>\n{text_msg}<|end|>\n<|assistant|>\n",
        text_msg = prompt,
    )
}

pub async fn run() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;
    let mut qllama = load_qllama_model(&device)?;
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
