use crate::types::ChunkTask;
use crossbeam_channel::Sender;
use llama_cpp_2::model::LlamaModel;
use std::io::{self, Read};

pub fn parse_and_chunk(
    model: &LlamaModel,
    worker_tx: Sender<ChunkTask>,
    target_tokens: usize,
) {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    
    let mut full_text = String::new();
    reader.read_to_string(&mut full_text).expect("Failed to read stdin");
    let mut chunk_index = 0;
    let mut start_idx = 0;

    let chars: Vec<char> = full_text.chars().collect();
    
    while start_idx < chars.len() {
        // Binary search to find approximately specified tokens
        let mut left = start_idx + 1;
        let mut right = chars.len();
        let mut best_idx = right;
        
        // If remaining is small, directly take the rest
        let remainder: String = chars[start_idx..right].iter().collect();
        let remainder_tokens = model.str_to_token(&remainder, llama_cpp_2::model::AddBos::Never).unwrap_or_default().len();
        
        if remainder_tokens <= target_tokens {
            best_idx = right;
        } else {
            // Binary search max matching offset
            while left <= right {
                let mid = left + (right - left) / 2;
                let chunk_str: String = chars[start_idx..mid].iter().collect();
                let tokens = model.str_to_token(&chunk_str, llama_cpp_2::model::AddBos::Never).unwrap_or_default().len();
                
                if tokens <= target_tokens {
                    best_idx = mid;
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            
            // Search backwards for the last \n or period (。 or .)
            let mut split_idx = best_idx;
            let mut found_punct = false;
            for i in (start_idx..best_idx).rev() {
                if chars[i] == '\n' || chars[i] == '。' || chars[i] == '.' {
                    split_idx = i + 1; // Include the punctuation
                    found_punct = true;
                    break;
                }
            }
            
            // If no punctuation found, just split at best_idx
            if found_punct {
                best_idx = split_idx;
            }
        }
        
        let chunk_str: String = chars[start_idx..best_idx].iter().collect();
        worker_tx.send(ChunkTask {
            index: chunk_index,
            text: chunk_str,
        }).expect("Worker disconnected");
        
        chunk_index += 1;
        start_idx = best_idx;
    }
}
