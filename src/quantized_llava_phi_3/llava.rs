use super::quantized_llama;
use candle_core::{quantized::gguf_file, Device};
use tokenizers::Tokenizer;

pub struct QLLaVAPhi3 {
    pub llama: quantized_llama::ModelWeights,
    pub tokenizer: Tokenizer,
}

impl QLLaVAPhi3 {
    pub fn load(
        device: &Device,
        gguf_model_path: &str,
        tokenizer_path: &str,
    ) -> anyhow::Result<Self> {
        let model_path = std::path::PathBuf::from(gguf_model_path);
        let mut file = std::fs::File::open(&model_path)?;
        let gguf_content =
            gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
        let llama = quantized_llama::ModelWeights::from_gguf(gguf_content, &mut file, &device)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        Ok(Self { llama, tokenizer })
    }

    pub fn format_prompt(prompt: &str) -> String {
        format!(
            "<s><|system|>\n<|end|>\n<|user|>\n<image>\n{text_msg}<|end|>\n<|assistant|>\n",
            text_msg = prompt,
        )
    }
}
