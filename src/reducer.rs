use std::collections::BTreeMap;
use std::sync::Arc;
use crossbeam_channel::{Receiver, bounded};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::num::NonZeroU32;
use std::thread;
use std::io::{self, Write};
use crate::prompts::generate_meta_prompt;
use crate::config::*;

pub fn run_reducer(
    reducer_model: Arc<LlamaModel>,
    reducer_backend: Arc<LlamaBackend>,
    reducer_prompt: String,
    reducer_rx: Receiver<(usize, String)>,
    config: Arc<AppConfig>,
) {
    let mut ordered_chunks = BTreeMap::new();
    let mut next_expected_idx = 0;
    
    let mut rolling_buffer = String::new();
    let mut rolling_token_count = 0;
    let mut intermediate_count = 0;
    
    let mut sample_summaries = String::new();
    let mut meta_prompt_rx: Option<Receiver<String>> = None;
    let mut dynamic_prompt: Option<String> = None;
    let meta_model = reducer_model.clone();
    let meta_backend = reducer_backend.clone();
    
    // Context configuration for reducing
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(config.main_ctx_size).unwrap()))
        .with_n_batch(config.batch_size_limit as u32)
        .with_n_ubatch(config.batch_size_limit as u32);
    let mut reducer_ctx = reducer_model
        .new_context(reducer_backend.as_ref(), ctx_params)
        .expect("Failed to create reducer context");

    loop {
        // Process all pending chunks
        match reducer_rx.recv() {
            Ok((idx, text)) => {
                ordered_chunks.insert(idx, text);
            }
            Err(_) => {
                // Channel closed, process remaining and break
                break;
            }
        }

        // Continuously append chunks in order
        while let Some(text) = ordered_chunks.remove(&next_expected_idx) {
            let chunk_tokens = reducer_model.str_to_token(&text, llama_cpp_2::model::AddBos::Never).unwrap_or_default().len();
            rolling_buffer.push_str(&format!("[Data {}]\n{}\n\n", next_expected_idx, text));
            rolling_token_count += chunk_tokens;
            
            if next_expected_idx < 3 {
                sample_summaries.push_str(&text);
                sample_summaries.push_str("\n\n");
            }
            
            // Spawning the meta-prompt evaluation asynchronously when it has processed 2 chunks
            if next_expected_idx == 1 && meta_prompt_rx.is_none() {
                let (tx, rx) = bounded(1);
                meta_prompt_rx = Some(rx);
                let m_model = meta_model.clone();
                let m_backend = meta_backend.clone();
                let sample = sample_summaries.clone();
                let m_config = config.clone();
                thread::spawn(move || {
                    let prompt = generate_meta_prompt(m_model, m_backend, sample, m_config);
                    let _ = tx.send(prompt);
                });
            }
            
            next_expected_idx += 1;
            
            // Intermediate Reduce if buffer exceeds 24,000 tokens
            if rolling_token_count >= 24000 {
                println!("\n[Intermediate Reduce {} Triggered]", intermediate_count);
                
                if dynamic_prompt.is_none() {
                    if let Some(rx) = &meta_prompt_rx {
                        dynamic_prompt = Some(rx.recv().unwrap_or_else(|_| reducer_prompt.clone()));
                    } else {
                        dynamic_prompt = Some(generate_meta_prompt(meta_model.clone(), meta_backend.clone(), sample_summaries.clone(), config.clone()));
                    }
                    println!("\n[Meta-Prompt Applied]: {}", dynamic_prompt.as_ref().unwrap());
                }
                
                let intermediate_prompt = config.intermediate_reduce_prompt
                    .replace("{SYS_PROMPT}", dynamic_prompt.as_ref().unwrap())
                    .replace("{TEXT}", &rolling_buffer);
                
                // Execute Reducer Context
                reducer_ctx.clear_kv_cache();
                let tokens = reducer_model.str_to_token(&intermediate_prompt, llama_cpp_2::model::AddBos::Always).unwrap();
                let mut n_eval = 0;
                let mut last_batch_tokens = 0;
                while n_eval < tokens.len() {
                    let chunk_size = std::cmp::min(tokens.len() - n_eval, config.batch_size_limit);
                    let mut batch = LlamaBatch::new(chunk_size, 1);
                    for i in 0..chunk_size {
                        let token = tokens[n_eval + i].clone();
                        let is_last = (n_eval + i) == (tokens.len() - 1);
                        batch.add(token, (n_eval + i) as i32, &[0], is_last).unwrap();
                    }
                    reducer_ctx.decode(&mut batch).unwrap();
                    last_batch_tokens = chunk_size;
                    n_eval += chunk_size;
                }

                let mut batch = LlamaBatch::new(1, 1);
                let mut n_cur = tokens.len() as i32;
                let mut decoder = encoding_rs::UTF_8.new_decoder();
                let mut compressed_text = String::new();
                
                let mut sampler = LlamaSampler::chain_simple([
                    LlamaSampler::temp(config.sample_temp),
                    LlamaSampler::top_k(config.sample_top_k),
                    LlamaSampler::top_p(config.sample_top_p, 1),
                    LlamaSampler::penalties(config.penalty_last_n, config.penalty_repeat, 0.05, 0.05),
                    LlamaSampler::dist(1234),
                ]);
                // Isolate history: only penalize newly generated tokens, not the input prompt.
                // sampler.accept_many(&tokens);

                loop {
                    let mut candidates_p = LlamaTokenDataArray::from_iter(reducer_ctx.candidates_ith(last_batch_tokens as i32 - 1), false);
                    last_batch_tokens = 1;
                    
                    candidates_p.apply_sampler(&mut sampler);
                    let new_token_id = candidates_p.selected_token().expect("Failed to sample token");
                    sampler.accept(new_token_id);
                    
                    if new_token_id == reducer_model.token_eos() || n_cur >= config.max_generate_tokens { break; }
                    let token_str = crate::types::decode_token(&reducer_model, new_token_id, &mut decoder);
                    compressed_text.push_str(&token_str);

                    batch.clear();
                    batch.add(new_token_id, n_cur, &[0], true).unwrap();
                    if reducer_ctx.decode(&mut batch).is_err() { break; }
                    n_cur += 1;
                }
                
                // Reset buffer with compressed memory
                rolling_buffer = format!("[Intermediate Summary {}]\n{}\n\n", intermediate_count, compressed_text);
                rolling_token_count = reducer_model.str_to_token(&rolling_buffer, llama_cpp_2::model::AddBos::Never).unwrap_or_default().len();
                intermediate_count += 1;
            }
        }
    }
    
    // Final Output
    if !rolling_buffer.is_empty() {
         if dynamic_prompt.is_none() {
             if let Some(rx) = &meta_prompt_rx {
                 dynamic_prompt = Some(rx.recv().unwrap_or_else(|_| reducer_prompt.clone()));
             } else {
                 dynamic_prompt = Some(generate_meta_prompt(meta_model.clone(), meta_backend.clone(), sample_summaries.clone(), config.clone()));
             }
             println!("\n[Meta-Prompt Applied]: {}", dynamic_prompt.as_ref().unwrap());
         }

         println!("\n[Final Summary]");
         let final_prompt = config.final_reduce_prompt
            .replace("{SYS_PROMPT}", dynamic_prompt.as_ref().unwrap())
            .replace("{TEXT}", &rolling_buffer);
         reducer_ctx.clear_kv_cache();
         let tokens = reducer_model.str_to_token(&final_prompt, llama_cpp_2::model::AddBos::Always).unwrap();
         let mut n_eval = 0;
         let mut last_batch_tokens = 0;
         while n_eval < tokens.len() {
             let chunk_size = std::cmp::min(tokens.len() - n_eval, config.batch_size_limit);
             let mut batch = LlamaBatch::new(chunk_size, 1);
             for i in 0..chunk_size {
                 let token = tokens[n_eval + i].clone();
                 let is_last = (n_eval + i) == (tokens.len() - 1);
                 batch.add(token, (n_eval + i) as i32, &[0], is_last).unwrap();
             }
             reducer_ctx.decode(&mut batch).unwrap();
             last_batch_tokens = chunk_size;
             n_eval += chunk_size;
         }
         let mut batch = LlamaBatch::new(1, 1);
         let mut n_cur = tokens.len() as i32;
         let mut decoder = encoding_rs::UTF_8.new_decoder();
         
         let mut sampler = LlamaSampler::chain_simple([
             LlamaSampler::temp(config.sample_temp),
             LlamaSampler::top_k(config.sample_top_k),
             LlamaSampler::top_p(config.sample_top_p, 1),
             LlamaSampler::penalties(config.penalty_last_n, config.penalty_repeat, 0.05, 0.05),
             LlamaSampler::dist(1234),
         ]);
         // Isolate history: only penalize newly generated tokens, not the input prompt.
         // sampler.accept_many(&tokens);

         loop {
             let mut candidates_p = LlamaTokenDataArray::from_iter(reducer_ctx.candidates_ith(last_batch_tokens as i32 - 1), false);
             last_batch_tokens = 1;
             
             candidates_p.apply_sampler(&mut sampler);
             let new_token_id = candidates_p.selected_token().expect("Failed to sample token");
             sampler.accept(new_token_id);
             if new_token_id == reducer_model.token_eos() || n_cur >= config.max_generate_tokens { break; }
             let token_str = crate::types::decode_token(&reducer_model, new_token_id, &mut decoder);
             print!("{}", token_str);
             io::stdout().flush().unwrap();
             
             batch.clear();
             batch.add(new_token_id, n_cur, &[0], true).unwrap();
             if reducer_ctx.decode(&mut batch).is_err() { break; }
             n_cur += 1;
         }
         println!();
    }
}
