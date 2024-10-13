use candle_core::{quantized::gguf_file, DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::{clip, quantized_llama},
};
use std::io::Write;
use tokenizers::Tokenizer;

#[derive(Debug)]
enum Prompt {
    Interactive,
    Chat,
    One(String),
}

fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}

fn load_image<T: AsRef<std::path::Path>>(path: T, image_size: usize) -> anyhow::Result<Tensor> {
    let img = image::ImageReader::open(path)?.decode()?;
    let (height, width) = (image_size, image_size);
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?;
    Ok(img)
}

fn load_images<T: AsRef<std::path::Path>>(
    paths: &Vec<T>,
    image_size: usize,
) -> anyhow::Result<Tensor> {
    let mut images = vec![];
    for path in paths {
        let tensor = load_image(path, image_size)?;
        images.push(tensor);
    }
    let images = Tensor::stack(&images, 0)?;
    Ok(images)
}

pub async fn run() -> anyhow::Result<()> {
    let device = Device::new_metal(0)?;

    let image_path = "./models/20240923-173209.jpeg";
    // use crate::phi3v::image_process::Phi3VImageProcessor;
    // use image::DynamicImage;
    // let img: DynamicImage = image::open(image_path).unwrap();
    // let image_processor = Phi3VImageProcessor::new();
    // let result = image_processor.preprocess(&img)?;
    // let pixel_values = result.pixel_values;
    // println!("pixel_values {:?}", pixel_values);

    let vision_features = {
        let model_file = {
            let api = hf_hub::api::sync::Api::new()?;

            let api = api.repo(hf_hub::Repo::with_revision(
                "openai/clip-vit-base-patch32".to_string(),
                hf_hub::RepoType::Model,
                "refs/pr/15".to_string(),
            ));

            api.get("model.safetensors")?
        };
        let config = clip::ClipConfig::vit_base_patch32();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_file.clone()], DType::F32, &device)?
        };
        let clip_model = clip::ClipModel::new(vb, &config)?;
        let images = load_images(&vec![image_path], config.image_size)?.to_device(&device)?;
        // let flat_values: Vec<f32> = pixel_values.iter().cloned().collect();
        // let shape: candle_core::Shape = pixel_values.shape().into();
        // // Create a new Tensor
        // let pixel_values_tensor = Tensor::new(flat_values, &device)?.reshape(shape)?;
        let vision_features = clip_model.get_image_features(&images)?;
        vision_features
    };
    println!("vision_features {:?}", vision_features);

    let vision_input = {
        let vision_features = vision_features.reshape((1, 512))?;
        // Quantize to F16
        let quantized_features = vision_features.to_dtype(DType::F32)?;
        let vision_input = quantized_features.unsqueeze(0)?; // 添加批次维度
                                                             // let features_vec = quantized_features.flatten_all()?.to_vec1::<f32>()?;
        vision_input
    };

    let model_path = std::path::PathBuf::from("models/llava-phi-3/llava-phi-3-mini-int4.gguf");
    let mut file = std::fs::File::open(&model_path)?;
    let start = std::time::Instant::now();

    let mut model = {
        let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
        let mut total_size_in_bytes = 0;
        for (_, tensor) in model.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }
        println!(
            "loaded {:?} tensors ({}) in {:.2}s",
            model.tensor_infos.len(),
            &format_size(total_size_in_bytes),
            start.elapsed().as_secs_f32(),
        );
        quantized_llama::ModelWeights::from_gguf(model, &mut file, &device)?
    };
    println!("model built");

    let tokenizer =
        Tokenizer::from_file("models/llava-phi-3/tokenizer.json").map_err(anyhow::Error::msg)?;
    let mut tos = TokenOutputStream::new(tokenizer);
    let prompt = "describe this image";

    let prompt_str = format!(
        "<|system|>\n</end>\n<|user|>\n<image>\n{text_msg}\n</end>\n<|assistant|>",
        text_msg = prompt,
    );
    println!("{}", &prompt_str);
    let tokens = tos
        .tokenizer()
        .encode(prompt_str, true)
        .map_err(anyhow::Error::msg)?;

    let prompt_tokens = tokens.get_ids().to_vec();
    let to_sample = 1000; // TODO: make this a parameter
    let prompt_tokens = if prompt_tokens.len() + to_sample > quantized_llama::MAX_SEQ_LEN - 10 {
        let to_remove = prompt_tokens.len() + to_sample + 10 - quantized_llama::MAX_SEQ_LEN;
        prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
    } else {
        prompt_tokens
    };

    let mut logits_processor = {
        let temperature = 0.0; // TODO: make this a parameter
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            Sampling::All { temperature }
        };
        let seed = 299792458; // TODO: make this a parameter
        LogitsProcessor::from_sampling(seed, sampling)
    };

    let start_prompt_processing = std::time::Instant::now();
    let mut next_token = {
        let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?;
        logits_processor.sample(&logits)?
    };
    let prompt_dt = start_prompt_processing.elapsed();

    let mut all_tokens = vec![];
    all_tokens.push(next_token);
    if let Some(t) = tos.next_token(next_token)? {
        print!("{t}");
        std::io::stdout().flush()?;
    }

    let eos_token = "<|end|>";
    let eos_token = *tos.tokenizer().get_vocab(true).get(eos_token).unwrap();
    let start_post_prompt = std::time::Instant::now();
    let mut sampled = 0;
    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        let repeat_penalty = 1.1; // TODO: make this a parameter
        let repeat_last_n = 64; // TODO: make this a parameter
        let logits = if repeat_penalty == 1. {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &all_tokens[start_at..],
            )?
        };
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
        sampled += 1;
        if next_token == eos_token {
            break;
        };
    }
    if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? {
        print!("{rest}");
    }
    std::io::stdout().flush()?;
    let dt = start_post_prompt.elapsed();
    println!(
        "\n\n{:4} prompt tokens processed: {:.2} token/s",
        prompt_tokens.len(),
        prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
    );
    println!(
        "{sampled:4} tokens generated: {:.2} token/s",
        sampled as f64 / dt.as_secs_f64(),
    );

    Ok(())
}
