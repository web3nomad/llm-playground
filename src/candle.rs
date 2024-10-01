use anyhow::Result;
use candle_transformers::quantized_var_builder::VarBuilder;
use std::path::PathBuf;

pub async fn run() -> Result<()> {
    let path: PathBuf = "models/llava-phi-3-mini-mmproj-f16.gguf".into();
    let vars = VarBuilder::from_gguf(path, &candle_core::Device::Cpu)?;
    // println!("{:?}", vars);
    Ok(())
}
