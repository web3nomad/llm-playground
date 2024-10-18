use anyhow::Result;
#[allow(dead_code)]
mod candle;
#[allow(dead_code)]
mod gguf;
#[allow(dead_code)]
mod mistral;
#[allow(dead_code)]
mod phi3v;
mod quantized_llava_phi_3;

#[tokio::main]
async fn main() -> Result<()> {
    // mistral::run().await?;
    // candle::run().await?;
    // phi3v::run().await?;
    gguf::run().await?;
    Ok(())
}
