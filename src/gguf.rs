use anyhow::Result;
use either::Either;
use indexmap::IndexMap;
use mistralrs::{
    Constraint, DefaultSchedulerMethod, Device, DeviceMapMetadata, GGUFLoaderBuilder,
    GGUFSpecificConfig, GgufModelBuilder, MistralRs, MistralRsBuilder, ModelDType, NormalRequest,
    Request, RequestMessage, ResponseOk, SamplingParams, SchedulerConfig, TokenSource,
    VisionLoaderBuilder, VisionLoaderType, VisionSpecificConfig,
};
use std::sync::Arc;
use tokio::sync::mpsc::channel;

pub async fn run() -> Result<()> {
    let vision_loader = VisionLoaderBuilder::new(
        VisionSpecificConfig {
            use_flash_attn: false,
            prompt_batchsize: None,
            topology: None,
            write_uqff: None,
            from_uqff: None,
        },
        None,
        None,
        Some("xtuner/llava-phi-3-mini-hf".to_string()),
    )
    .build(VisionLoaderType::LLaVA);

    let vision_pipeline = vision_loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        &ModelDType::Auto,
        &Device::cuda_if_available(0)?,
        false,
        DeviceMapMetadata::dummy(),
        None,
        None, // No PagedAttention.
    )?;

    let gguf_loader = GGUFLoaderBuilder::new(
        Some("models/phi3.json".to_string()),
        None,
        "./models".to_string(),
        vec![
            "llava-phi-3-mini-int4.gguf".to_string(),
            // "llava-phi-3-mini-mmproj-f16.gguf".to_string(),
        ],
        GGUFSpecificConfig {
            prompt_batchsize: None,
            topology: None,
        },
    )
    .build();

    let gguf_pipeline = gguf_loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        &ModelDType::Auto,
        &Device::cuda_if_available(0)?,
        false,
        DeviceMapMetadata::dummy(),
        None,
        None, // No PagedAttention.
    )?;

    Ok(())
}
