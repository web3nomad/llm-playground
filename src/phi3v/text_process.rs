use anyhow::Result;
use ndarray::Array2;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct Phi3VTextProcessor {
    tokenizer_path: String,
}

impl Phi3VTextProcessor {
    pub fn new(tokenizer_path: &str) -> Self {
        Self {
            tokenizer_path: tokenizer_path.to_string(),
        }
    }

    pub fn preprocess(&self, text: &str) -> Result<(Array2<i64>, Array2<i64>)> {
        let tokenizer =
            Tokenizer::from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join(&self.tokenizer_path))
                .unwrap();
        let inputs = vec![text.to_string()];

        // Encode our input strings. `encode_batch` will pad each input to be the same length.
        let encodings = tokenizer
            .encode_batch(inputs.clone(), false)
            .map_err(|e| anyhow::anyhow!("Error encoding batch: {:?}", e))?;

        // Get the padded length of each encoding.
        let padded_token_length = encodings[0].len();

        // Get our token IDs & mask as a flattened array.
        let ids: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_ids().iter().map(|i| *i as i64))
            .collect();
        let mask: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64))
            .collect();
        let input_ids = Array2::from_shape_vec([inputs.len(), padded_token_length], ids).unwrap();
        let attention_mask =
            Array2::from_shape_vec([inputs.len(), padded_token_length], mask).unwrap();

        Ok((input_ids, attention_mask))
    }
}
