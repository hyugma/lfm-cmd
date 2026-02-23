// -----------------------------------------------------------------------------
// LFM-CMD Configuration & Constants
// -----------------------------------------------------------------------------
#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// Default Prompts
pub const META_PROMPT_TEMPLATE: &str = "<|startoftext|><|im_start|>system\nあなたは優秀なプロンプトエンジニアです。<|im_end|>\n<|im_start|>user\n以下のテキスト断片を分析し、元のテキストのジャンル（小説、技術論文、システムログ、議事録など）を判定してください。\nその後、このテキスト全体を最も美しく構造化して要約するための「AIへの指示書（システムプロンプト）」を作成してください。\n出力は150文字以内の「指示書」のみとし、解説や挨拶は一切含めないでください。\n【テキスト断片】\n{TEXT}<|im_end|>\n<|im_start|>assistant\n";

pub const WORKER_PROMPT_TEMPLATE: &str = "<|startoftext|><|im_start|>system\n{SYS_PROMPT}\nあなたは入力テキストを要約するAIです。重要なポイントを逃さず、できるだけ簡潔にまとめてください。<|im_end|>\n<|im_start|>user\n以下のテキストを要約してください。\n\n{TEXT}<|im_end|>\n<|im_start|>assistant\n";

pub const INTERMEDIATE_REDUCE_PROMPT: &str = "<|startoftext|><|im_start|>system\n{SYS_PROMPT}<|im_end|>\n<|im_start|>user\n以下のテキスト群を統合・圧縮して、重要なコンテキストを維持した新しい中間要約を生成してください。\n\n{TEXT}<|im_end|>\n<|im_start|>assistant\n";

pub const FINAL_REDUCE_PROMPT: &str = "<|startoftext|><|im_start|>system\n{SYS_PROMPT}<|im_end|>\n<|im_start|>user\n以下の内容を統合し、最終的な全体要約を作成してください。\n\n{TEXT}<|im_end|>\n<|im_start|>assistant\n";


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub meta_ctx_size: u32,
    pub main_ctx_size: u32,
    pub max_generate_tokens: i32,
    pub batch_size_limit: usize,
    
    pub sample_temp: f32,
    pub sample_top_k: i32,
    pub sample_top_p: f32,
    pub penalty_repeat: f32,
    pub penalty_last_n: i32,
    
    pub meta_prompt_template: String,
    pub worker_prompt_template: String,
    pub intermediate_reduce_prompt: String,
    pub final_reduce_prompt: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            meta_ctx_size: 8192,
            main_ctx_size: 32768,
            max_generate_tokens: 32768,
            batch_size_limit: 4096,
            
            sample_temp: 0.2,
            sample_top_k: 50,
            sample_top_p: 0.9,
            penalty_repeat: 1.00,
            penalty_last_n: 32,
            
            meta_prompt_template: META_PROMPT_TEMPLATE.to_string(),
            worker_prompt_template: WORKER_PROMPT_TEMPLATE.to_string(),
            intermediate_reduce_prompt: INTERMEDIATE_REDUCE_PROMPT.to_string(),
            final_reduce_prompt: FINAL_REDUCE_PROMPT.to_string(),
        }
    }
}
