use super::{
    image_processor::{HFPreProcessorConfig, ImageProcessor},
    linear::QLinear,
    vision_model::ClipVisionTransformer,
    QLlama,
};
use candle_core::{Device, IndexOp, Module, Tensor};
use candle_nn::{
    sequential::{seq, Sequential},
    Activation,
};
use candle_transformers::{models::clip::vision_model::ClipVisionConfig, quantized_var_builder};
use tokenizers::Tokenizer;

pub struct MMProjector {
    pub modules: Sequential,
}

impl MMProjector {
    pub fn load(vb: quantized_var_builder::VarBuilder) -> candle_core::Result<Self> {
        let text_hidden_size: usize = 3072;
        let mm_hidden_size: usize = 1024;
        let modules = {
            let layer = QLinear::load(mm_hidden_size, text_hidden_size, vb.pp("mm.0"))?;
            let mut modules = seq().add(layer);
            let mlp_depth = 2;
            for i in 1..mlp_depth {
                let layer = QLinear::load(
                    text_hidden_size,
                    text_hidden_size,
                    vb.pp(format!("mm.{}", i * 2)),
                )?;
                modules = modules.add(Activation::Gelu).add(layer);
            }
            modules
        };
        Ok(Self { modules })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.modules.forward(x)
    }
}

pub fn load_clip(
    device: &Device,
    gguf_model_path: &str,
) -> candle_core::Result<(ClipVisionTransformer, MMProjector)> {
    let vb = quantized_var_builder::VarBuilder::from_gguf(gguf_model_path, &device)?;
    // println!("Loaded gguf model {:?}", vb.pp("mm.0").get_no_shape("bias"));
    let mm_projector = MMProjector::load(vb.clone())?;
    let clip_vision_config = ClipVisionConfig::clip_vit_large_patch14_336();
    let vision_model = ClipVisionTransformer::new(vb.pp("v"), &clip_vision_config)?;
    // let model_file = {
    //     let api = hf_hub::api::sync::Api::new()?;
    //     let api = api.repo(hf_hub::Repo::with_revision(
    //         "openai/clip-vit-large-patch14-336".to_string(),
    //         hf_hub::RepoType::Model,
    //         "refs/pr/8".to_string(),
    //     ));
    //     api.get("model.safetensors")?
    // };
    // let vb = unsafe {
    //     candle_nn::var_builder::VarBuilder::from_mmaped_safetensors(
    //         &[model_file.clone()],
    //         DType::F32,
    //         &device,
    //     )?
    // };
    // let vision_model = ClipVisionTransformer::new(vb.pp("vision_model"), &clip_vision_config)?;
    Ok((vision_model, mm_projector))
}

pub fn load_image_to_tensor(
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

/// Input prompt: "A photo of <image> next to <image>"
/// Output: [bos_token_id, ...(tokens for "A photo of"), image_token_id, ...(tokens for " next to "), image_token_id]
pub fn tokenizer_image_token(
    prompt: &str,
    tokenizer: &Tokenizer,
    image_token_id: i64,
    bos_token_id: i64,
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
        && prompt_chunks[0][0] == bos_token_id
    {
        offset = 1;
        input_ids.push(prompt_chunks[0][0]);
    }

    for x in insert_separator(prompt_chunks, duplicate_vec(&[image_token_id], offset + 1)).iter() {
        input_ids.extend(x[1..].to_vec())
    }
    // println!("input_ids: {:?}", input_ids);
    let input_len = input_ids.len();
    Tensor::from_vec(input_ids, (1, input_len), &Device::Cpu)
}

/// image 564 564
/// x shape: Tensor[dims 1, 3, 336, 336; bf16, metal:4294969862]
/// clip_vision_tower result shape: Tensor[dims 1, 576, 1024; bf16, metal:4294969862]
/// mm_projector result shape: Tensor[dims 1, 576, 3072; bf16, metal:4294969862]
fn encode_images(
    x: &Tensor,
    clip_vision_model: &ClipVisionTransformer,
    mm_projector: &MMProjector,
) -> candle_core::Result<Tensor> {
    // println!("x shape: {:?}", x);
    // let image_features = clip_vision_tower.forward(x)?;
    let image_features = {
        let select_layer = -2;
        // let x = x.to_dtype(DType::F32)?;
        let result = clip_vision_model.output_hidden_states(x)?;
        let index = result.len() as isize + select_layer;
        let result = result[index as usize].clone();
        result.i((.., 1..))?
    };
    // let image_features = clip_vision_model.forward(&x.to_dtype(DType::F32)?)?;
    // println!("clip_vision_tower result shape: {:?}", image_features);
    let image_features = mm_projector.forward(&image_features)?;
    // println!("mm_projector result shape: {:?}", image_features);
    // let image_features = image_features.to_dtype(DType::BF16)?;
    // println!("image_features shape: {:?}", image_features);
    Ok(image_features)
}

pub fn prepare_inputs_labels_for_multimodal(
    device: &Device,
    qllama: &QLlama,
    input_ids: &Tensor,
    images: &[Tensor],
    _image_sizes: &[(u32, u32)],
    image_token_id: i64,
    clip_vision_model: &ClipVisionTransformer,
    mm_projector: &MMProjector,
) -> candle_core::Result<Tensor> {
    let concat_images = Tensor::cat(images, 0)?;
    let image_features_together = encode_images(&concat_images, clip_vision_model, mm_projector)?;
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
    let cur_input_embeds = qllama.model.embed(&input_ids_noim)?;
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
    let new_input_embeds = if let Some(tokenizer_model_max_length) = tokenizer_model_max_length {
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
