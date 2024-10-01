use anyhow::Result;
use ndarray::Array2;
use std::path::Path;
// use tokenizers::models::bpe::BPE;
use tokenizers::Tokenizer;

// 参考了 https://github.com/pykeio/ort/blob/main/examples/sentence-transformers/examples/semantic-similarity.rs

pub struct Phi3VTextProcessor {
    tokenizer: Tokenizer,
}

impl Phi3VTextProcessor {
    pub fn new(tokenizer_path: &str) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join(tokenizer_path))
                .map_err(|e| anyhow::anyhow!("Error loading tokenizer: {:?}", e))?;
        // let model = tokenizer.get_model();
        // println!("tokenizer model is {:?}", model); // should be tokenizers::models::bpe::BPE
        Ok(Self { tokenizer })
    }

    pub fn decode(&self, ids: &Vec<u32>) -> Result<String> {
        let ids: Vec<u32> = ids.iter().cloned().collect();
        let text = self.tokenizer.decode(&ids, false).unwrap();
        Ok(text)
    }

    pub fn preprocess(&self, text: &str) -> Result<(Array2<i64>, Array2<i64>)> {
        let formatted_text = self.format_chat_template(text);
        let encoding = self
            .tokenizer
            .encode(formatted_text, true)
            .map_err(|e| anyhow::anyhow!("Error encoding: {:?}", e))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&mask| mask as i64)
            .collect();

        let input_ids = Array2::from_shape_vec((1, input_ids.len()), input_ids)?;
        let attention_mask = Array2::from_shape_vec((1, attention_mask.len()), attention_mask)?;

        Ok((input_ids, attention_mask))
    }

    // 参考了 https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi3v.py
    // 包含了 `<s>` token，这是 LlamaTokenizer 通常使用的 BOS token
    fn format_chat_template(&self, text: &str) -> String {
        format!("<s><|user|>\n{text}<|end|>\n<|assistant|>\n", text = text)
    }
}
