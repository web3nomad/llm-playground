use super::{
    clip::{ClipVisionTower, MMProjector},
    image_processor::{HFPreProcessorConfig, ImageProcessor},
    quantized_llama,
};
use candle_core::{quantized::gguf_file, Device, IndexOp, Tensor};
use candle_transformers::quantized_var_builder;
use tokenizers::Tokenizer;

pub const BOS_TOKEN_ID: u32 = 1;
pub const EOS_TOKEN_ID: u32 = 32007; // 模型实际输出的是 32007 <|end|>, 而不是 gguf 里配置的 32000 <|endoftext|>
pub const IMAGE_TOKEN_ID: u32 = 32038; // see tokenizer.json, <image>

pub struct QLLaVAPhi3 {
    pub llama: quantized_llama::ModelWeights,
    pub clip_vision_tower: ClipVisionTower,
    pub mm_projector: MMProjector,
    pub tokenizer: Tokenizer,
}

impl QLLaVAPhi3 {
    pub fn load(
        device: &Device,
        gguf_model_path: &str,
        mmproj_gguf_model_path: &str,
        tokenizer_path: &str,
    ) -> anyhow::Result<Self> {
        let llama = {
            let model_path = std::path::PathBuf::from(gguf_model_path);
            let mut file = std::fs::File::open(&model_path)?;
            let gguf_content =
                gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
            quantized_llama::ModelWeights::from_gguf(gguf_content, &mut file, &device)?
        };

        let (clip_vision_tower, mm_projector) = {
            let vb = quantized_var_builder::VarBuilder::from_gguf(mmproj_gguf_model_path, &device)?;
            let mm_projector = MMProjector::new(vb.pp("mm"))?;
            let clip_vision_tower = ClipVisionTower::new(vb.pp("v"))?;
            (clip_vision_tower, mm_projector)
        };

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        Ok(Self {
            llama,
            clip_vision_tower,
            mm_projector,
            tokenizer,
        })
    }

    /// image 564 564
    /// x shape:                        Tensor[dims 1, 3, 336, 336; bf16]
    /// clip_vision_tower result shape: Tensor[dims 1, 576, 1024; bf16]
    /// mm_projector result shape:      Tensor[dims 1, 576, 3072; bf16]
    fn encode_images(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let image_features = self.clip_vision_tower.forward(x)?;
        // let image_features = self.clip_vision_tower.forward(&x.to_dtype(DType::F32)?)?;
        let image_features = self.mm_projector.forward(&image_features)?;
        // let image_features = image_features.to_dtype(DType::BF16)?;
        Ok(image_features)
    }

    pub fn prepare_inputs_labels_for_multimodal(
        &self,
        device: &Device,
        QLLaVAPhi3 { llama, .. }: &QLLaVAPhi3,
        input_ids: &Tensor,
        images: &[Tensor],
        _image_sizes: &[(u32, u32)],
        image_token_id: i64,
    ) -> candle_core::Result<Tensor> {
        let concat_images = Tensor::cat(images, 0)?;
        let image_features_together = self.encode_images(&concat_images)?;
        let split_sizes = images
            .iter()
            .map(|x| x.shape().dims()[0])
            .collect::<Vec<usize>>();
        // can be replaced by split
        let mut index_pos = 0;
        let mut image_features = Vec::new();
        for split_size in split_sizes.iter() {
            image_features.push(image_features_together.i(index_pos..index_pos + (*split_size))?);
            index_pos += *split_size;
        }

        let image_features = image_features
            .iter()
            .map(|x| x.flatten(0, 1).unwrap())
            .collect::<Vec<Tensor>>();

        let input_ids_vec = input_ids.squeeze(0)?.to_vec1::<i64>()?;
        let mut image_indices = {
            let mut image_indices = vec![0_i64];
            image_indices.extend(
                input_ids_vec
                    .iter()
                    .enumerate()
                    .filter_map(|(i, x)| {
                        if *x == image_token_id {
                            Some(i as i64)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<i64>>(),
            );
            image_indices
        };

        // if image_indices.len() == 1 {
        //     //no image, only [0],
        //     return self.llama.embed(input_ids);
        // }

        let input_ids_noim = input_ids_vec
            .iter()
            .filter_map(|x| if *x != image_token_id { Some(*x) } else { None })
            .collect::<Vec<i64>>();
        let input_ids_noim_len = input_ids_noim.len();
        image_indices.push((input_ids_noim_len) as i64);
        let input_ids_noim = Tensor::from_vec(input_ids_noim, input_ids_noim_len, &device)?;
        // println!("input_ids_noim: {:?}", input_ids_noim);
        let cur_input_embeds = llama.embed(&input_ids_noim)?;
        // println!("cur_input_embeds: {:?}", cur_input_embeds);
        // can be replace by split if it is implemented in candle
        let input_embed_no_ims = {
            let mut input_embeds = Vec::new();
            for i in 0..image_indices.len() - 1 {
                let start = (image_indices[i]) as usize;
                let end = image_indices[i + 1] as usize;
                input_embeds.push(cur_input_embeds.i((start..end, ..))?)
            }
            input_embeds
        };
        let mut cur_new_input_embeds = Vec::new();
        for (i, image_feature) in image_features.iter().enumerate() {
            cur_new_input_embeds.push(input_embed_no_ims[i].clone());
            // 如果 encode_images 还没实现, 这里先不放进去, 不然 shape 不对
            cur_new_input_embeds.push(image_feature.clone());
        }
        cur_new_input_embeds.push(input_embed_no_ims[image_features.len()].clone());
        let new_input_embeds = Tensor::cat(&cur_new_input_embeds, 0)?;

        // trancate
        let tokenizer_model_max_length = Some(4096); // in models/llava-phi-3/tokenizer_config.json
        let new_input_embeds = if let Some(tokenizer_model_max_length) = tokenizer_model_max_length
        {
            let (new_input_embeds_length, _) = new_input_embeds.shape().dims2()?;
            if new_input_embeds_length > tokenizer_model_max_length {
                new_input_embeds.i((..tokenizer_model_max_length, ..))?
            } else {
                new_input_embeds
            }
        } else {
            new_input_embeds
        };

        Ok(new_input_embeds.unsqueeze(0)?)
    }

    pub fn format_prompt(prompt: &str) -> String {
        format!(
            "<s><|system|>\n<|end|>\n<|user|>\n<image>\n{text_msg}<|end|>\n<|assistant|>\n",
            text_msg = prompt,
        )
    }

    pub fn load_image(
        device: &Device,
        image_file_path: &str,
        preprocessor_config_file_path: &str,
    ) -> anyhow::Result<((u32, u32), Tensor)> {
        let preprocessor_config: HFPreProcessorConfig =
            serde_json::from_slice(&std::fs::read(preprocessor_config_file_path)?)?;
        let image_processor = ImageProcessor::from_hf_preprocessor_config(&preprocessor_config);

        let img = image::ImageReader::open(image_file_path)?.decode()?;
        let image_tensor = image_processor.preprocess(&img)?.unsqueeze(0)?;
        // let image_tensor = image_tensor.to_dtype(DType::BF16)?.to_device(&device)?;
        let image_tensor = image_tensor.to_device(&device)?;
        // println!("Image size: {:?}", (img.width(), img.height()));
        // println!("Image tensor: {:?}", image_tensor);
        Ok(((img.width(), img.height()), image_tensor))
    }

    /// Input prompt: "A photo of <image> next to <image>"
    /// Output: [bos_token_id, ...(tokens for "A photo of"), image_token_id, ...(tokens for " next to "), image_token_id]
    pub fn tokenizer_image_token(
        prompt: &str,
        tokenizer: &Tokenizer,
    ) -> candle_core::Result<Tensor> {
        let prompt_chunks = prompt
            .split("<image>")
            .map(|s| {
                tokenizer
                    .encode(s, true)
                    .unwrap()
                    .get_ids()
                    .to_vec()
                    .iter()
                    .map(|x| *x as i64)
                    .collect()
            })
            .collect::<Vec<Vec<i64>>>();
        let mut input_ids = Vec::new();
        let mut offset = 0;
        if !prompt_chunks.is_empty()
            && !prompt_chunks[0].is_empty()
            && prompt_chunks[0][0] == BOS_TOKEN_ID as i64
        {
            offset = 1;
            input_ids.push(prompt_chunks[0][0]);
        }

        for x in insert_separator(
            prompt_chunks,
            duplicate_vec(&[IMAGE_TOKEN_ID as i64], offset + 1),
        )
        .iter()
        {
            input_ids.extend(x[1..].to_vec())
        }
        // println!("input_ids: {:?}", input_ids);
        let input_len = input_ids.len();
        Tensor::from_vec(input_ids, (1, input_len), &Device::Cpu)
    }
}

fn duplicate_vec<T>(vec: &[T], n: usize) -> Vec<T>
where
    T: Clone,
{
    let mut res = Vec::new();
    for _ in 0..n {
        res.extend(vec.to_owned());
    }
    res
}

fn insert_separator<T>(x: Vec<Vec<T>>, sep: Vec<T>) -> Vec<Vec<T>>
where
    T: Clone,
{
    let sep = vec![sep];
    let sep = duplicate_vec(&sep, x.len());
    let mut res = x
        .iter()
        .zip(sep.iter())
        .flat_map(|(x, y)| vec![x.clone(), y.clone()])
        .collect::<Vec<Vec<T>>>();
    res.pop();
    res
}
