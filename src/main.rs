use anyhow::Result;
mod candle;
mod mistral;

#[tokio::main]
async fn main() -> Result<()> {
    // mistral::run().await?;
    candle::run().await?;
    Ok(())
}
