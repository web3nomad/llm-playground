use super::image_process::{Phi3VImageProcessor, NUM_IMG_TOKENS};
use anyhow::Result;
use image::DynamicImage;
use ndarray::{Array2, Array5};
use regex::Regex;
use serde_json::Value;
use std::fs::File;
use std::io::Read;

#[allow(dead_code)]
pub struct Phi3VProcessor {
    image_processor: Phi3VImageProcessor,
    tokenizer: LlamaTokenizer,
    special_image_token: String,
    num_img_tokens: usize,
    img_tokens: Vec<String>,
}

#[allow(dead_code)]
impl Phi3VProcessor {
    pub fn new(image_processor: Phi3VImageProcessor, tokenizer: LlamaTokenizer) -> Self {
        let special_image_token = "<|image|>".to_string();
        // TODO let num_img_tokens = image_processor.num_img_tokens;
        let num_img_tokens = NUM_IMG_TOKENS;
        let img_tokens = (1..=1000000).map(|i| format!("<|image_{}|>", i)).collect();

        Self {
            image_processor,
            tokenizer,
            special_image_token,
            num_img_tokens,
            img_tokens,
        }
    }

    pub fn call(&self, text: &str, img: &DynamicImage) -> Result<BatchFeature> {
        let image_inputs = self.image_processor.preprocess(img)?;
        self.convert_images_texts_to_inputs(&image_inputs, text)
    }

    fn convert_images_texts_to_inputs(
        &self,
        images: &super::image_process::BatchFeature,
        text: &str,
    ) -> Result<BatchFeature> {
        let pattern = Regex::new(r"<\|image_\d+\|>")?;
        let prompt_chunks: Vec<Vec<i64>> = pattern
            .split(text)
            .map(|chunk| self.tokenizer.encode(chunk))
            .collect();

        let num_img_tokens = images.num_img_tokens.clone();
        let pixel_values = images.pixel_values.clone();
        let image_sizes = images.image_sizes.clone();

        let image_tags: Vec<&str> = pattern.find_iter(text).map(|m| m.as_str()).collect();
        let image_ids: Vec<i64> = image_tags
            .iter()
            .map(|s| {
                s.split('_')
                    .last()
                    .unwrap()
                    .trim_end_matches('|')
                    .parse::<i64>()
                    .unwrap()
            })
            .collect();

        let mut unique_image_ids: Vec<i64> = image_ids
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        unique_image_ids.sort();

        if unique_image_ids != (1..=unique_image_ids.len() as i64).collect::<Vec<_>>() {
            return Err(anyhow::anyhow!(
                "image_ids must start from 1, and must be continuous int"
            ));
        }

        if unique_image_ids.len() != pixel_values.len() {
            return Err(anyhow::anyhow!(
                "total images must be the same as the number of image tags"
            ));
        }

        let image_ids_pad: Vec<Vec<i64>> = image_ids
            .iter()
            .map(|&id| vec![-id; num_img_tokens[(id - 1) as usize] as usize])
            .collect();

        let mut input_ids = Vec::new();
        let mut offset = 0;
        for (chunk, pad) in prompt_chunks
            .iter()
            .zip(image_ids_pad.iter().chain(std::iter::once(&Vec::new())))
        {
            input_ids.extend_from_slice(&chunk[offset..]);
            input_ids.extend_from_slice(pad);
            offset = 0;
        }

        // let input_ids = Tensor::from_slice(&input_ids).unsqueeze(0);
        // let attention_mask = input_ids
        //     .gt(-1000000)
        //     .to_dtype(ort::TensorElementDataType::Int64);

        Ok(BatchFeature {
            // input_ids,
            // attention_mask,
            pixel_values: pixel_values.clone(),
            image_sizes: image_sizes.clone(),
        })
    }

    pub fn special_image_token_id(&self) -> i64 {
        self.tokenizer
            .convert_tokens_to_ids(&[&self.special_image_token])[0]
    }

    pub fn batch_decode(&self, token_ids: &[Vec<i64>]) -> Vec<String> {
        self.tokenizer.batch_decode(token_ids)
    }

    pub fn decode(&self, token_ids: &[i64]) -> String {
        self.tokenizer.decode(token_ids)
    }
}

#[allow(dead_code)]
pub struct LlamaTokenizer {
    // Fields for the tokenizer
}

#[allow(dead_code)]
impl LlamaTokenizer {
    pub fn new(tokenizer_path: &str) -> Result<Self> {
        let mut file = File::open(tokenizer_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let _tokenizer_json: Value = serde_json::from_str(&contents)?;

        // Initialize the tokenizer with the loaded JSON data
        unimplemented!()
    }

    pub fn encode(&self, _text: &str) -> Vec<i64> {
        // Implement tokenization logic here
        unimplemented!()
    }

    pub fn convert_tokens_to_ids(&self, _tokens: &[&str]) -> Vec<i64> {
        // Implement token to ID conversion logic here
        unimplemented!()
    }

    pub fn batch_decode(&self, _token_ids: &[Vec<i64>]) -> Vec<String> {
        // Implement batch decoding logic here
        unimplemented!()
    }

    pub fn decode(&self, _token_ids: &[i64]) -> String {
        // Implement decoding logic here
        unimplemented!()
    }
}

#[allow(dead_code)]
pub struct BatchFeature {
    // input_ids: Tensor,
    // attention_mask: Tensor,
    pub pixel_values: Array5<f32>,
    pub image_sizes: Array2<i64>,
}
