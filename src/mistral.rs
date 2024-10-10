use anyhow::Result;
#[allow(unused_imports)]
use mistralrs::{
    GGUFLoader,
    GgufModelBuilder,
    LLaVALoader,
    Loader,
    PagedAttentionMetaBuilder,
    TextMessageRole,
    VisionLoaderType,
    VisionMessages,
    VisionModelBuilder, //
};

pub async fn run() -> Result<()> {
    let model = GgufModelBuilder::new(
        "./models",
        vec![
            "llava-phi-3-mini-int4.gguf",
            // "llava-phi-3-mini-mmproj-f16.gguf",
            // "llava-phi-3-mini-xtuner-q4_k_m.gguf",
        ],
    )
    .with_tokenizer_json("./models/tokenizer.json")
    .with_chat_template("./models/phi3.json")
    .with_logging()
    .with_paged_attn(|| {
        PagedAttentionMetaBuilder::default()
            // .with_block_size(32)
            // .with_gpu_memory(mistralrs::MemoryGpuConfig::ContextSize(1024))
            .build()
    })?
    .build()
    .await?;

    // let model = VisionModelBuilder::new(
    //     // "microsoft/Phi-3.5-vision-instruct".to_string(),
    //     // "xtuner/llava-phi-3-mini-hf".to_string(),
    //     // "xtuner/llava-phi-3-mini-gguf".to_string(),
    //     // "mlx-community/llava-phi-3-mini-4bit".to_string(),
    //     "dwb2023/phi-3-vision-128k-instruct-quantized".to_string(),
    //     VisionLoaderType::Phi3V,
    // )
    // .with_tokenizer_json("./models/phi-3-vision/tokenizer.json")
    // .with_chat_template("./models/phi-3-vision/tokenizer_config.json")
    // // .with_dtype(mistralrs::ModelDType::Auto)
    // // .with_isq(mistralrs::IsqType::Q4_0)
    // .with_logging()
    // .build()
    // .await?;

    let img = image::open("./models/20240923-173209.jpeg").unwrap();
    println!("image {:?} {:?}", img.width(), img.height());
    let messages = VisionMessages::new().add_llava_image_message(
        TextMessageRole::User,
        "Describe the image.",
        img,
    );

    let response = match model.send_chat_request(messages).await {
        Ok(response) => response,
        Err(e) => {
            println!("Error: {:?}", e);
            return Ok(());
        }
    };

    // println!("{:?}", response);
    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
