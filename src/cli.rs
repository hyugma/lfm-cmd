use clap::Parser;
use std::path::PathBuf;

/// A blazing fast, generic stream AI processing CLI tool using Metal & GGUF
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Max tokens per chunk. Defines the "sweet spot" for context comprehension.
    #[arg(short = 't', long, default_value_t = 512)]
    pub tokens: usize,

    /// Number of parallel workers
    #[arg(short = 'w', long, default_value_t = 2)]
    pub workers: usize,

    /// Path to the GGUF model file. If not provided, uses the embedded LFM2.5 model.
    #[arg(short = 'm', long)]
    pub model: Option<PathBuf>,

    /// System prompt
    #[arg(
        short = 'p',
        long,
        default_value = "提供されたテキストを解析・要約し3行で出力してください。"
    )]
    pub prompt: String,

    /// Path to advanced JSON configuration file
    #[arg(short = 'c', long)]
    pub config: Option<PathBuf>,
}
