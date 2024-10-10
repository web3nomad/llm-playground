use anyhow::Result;
#[allow(dead_code)]
mod candle;
#[allow(dead_code)]
mod mistral;
#[allow(dead_code)]
mod phi3v;

#[tokio::main]
async fn main() -> Result<()> {
    mistral::run().await?;
    // candle::run().await?;
    // phi3v::run().await?;
    Ok(())
}
