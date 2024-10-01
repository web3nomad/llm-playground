use anyhow::Result;
mod candle;
mod mistral;
mod ort;

#[tokio::main]
async fn main() -> Result<()> {
    // mistral::run().await?;
    // candle::run().await?;
    ort::run().await?;
    Ok(())
}
