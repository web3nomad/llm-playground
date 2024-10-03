use anyhow::Result;
mod candle;
mod mistral;
mod phi3v;

#[tokio::main]
async fn main() -> Result<()> {
    // mistral::run().await?;
    // candle::run().await?;
    phi3v::run().await?;
    Ok(())
}
