use candle_core::{quantized::gguf_file, Device};
use candle_transformers::{generation::Sampling, models::quantized_llama};
use tokenizers::Tokenizer;

// fn load_llava_config() -> anyhow::Result<serde_json::Value> {
//     let config_filename = api.get("config.json")?;
//     let llava_config: LLaVAConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
//     (
//         llava_config.clone(),
//         None,
//         ImageProcessor::from_pretrained(&llava_config.mm_vision_tower.unwrap())?,
//     )
// }

pub struct QLlama {
    pub model: quantized_llama::ModelWeights,
    pub tokenizer: Tokenizer,
    pub eos_token_id: u32,
    pub sampling: Sampling,
    pub sample_len: usize,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

pub fn load_qllama_model(
    device: &Device,
    gguf_model_path: &str,
    tokenizer_path: &str,
) -> anyhow::Result<QLlama> {
    let model_path = std::path::PathBuf::from(gguf_model_path);
    let mut file = std::fs::File::open(&model_path)?;
    let gguf_content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
    // let gguf_metadata = gguf_content.metadata.clone();
    let model = quantized_llama::ModelWeights::from_gguf(gguf_content, &mut file, &device)?;
    // println!("model built");
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
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
        eos_token_id: 32007, // 模型实际输出的是 32007 <|end|>, 而不是 gguf 里配置的 32000 <|endoftext|>
        sampling,
        sample_len: 1000,
        seed: 299792458,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
    })
}

pub fn format_prompt(prompt: &str) -> String {
    format!(
        "<s><|system|>\n<|end|>\n<|user|>\n<image>\n{text_msg}<|end|>\n<|assistant|>\n",
        text_msg = prompt,
    )
}
