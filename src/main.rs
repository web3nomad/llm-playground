use anyhow::Result;
mod mistral;

#[tokio::main]
async fn main() -> Result<()> {
    mistral::run().await?;
    Ok(())
}
