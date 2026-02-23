use crate::types::ChunkTask;
use crossbeam_channel::Receiver;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::num::NonZeroU32;
use std::sync::Arc;
use crate::config::*;

pub fn worker_loop(
    _worker_id: usize,
    model: Arc<LlamaModel>,
    backend: Arc<LlamaBackend>,
    rx: Receiver<ChunkTask>,
    system_prompt: String,
    config: Arc<AppConfig>,
) -> Vec<(usize, String)> {
    // Each worker has its own context. This prevents locking during inference.
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(config.main_ctx_size).unwrap()))
        .with_n_batch(config.batch_size_limit as u32)
        .with_n_ubatch(config.batch_size_limit as u32);
    let mut ctx = model
        .new_context(backend.as_ref(), ctx_params)
        .expect("Failed to create context");

    let mut outputs = Vec::new();

    for task in rx {
        // Clear the cache from previous chunk processing to prevent overflow and overlap
        ctx.clear_kv_cache();

        // Build the prompt for the model using LFM2.5 ChatML template
        let prompt = config.worker_prompt_template
            .replace("{SYS_PROMPT}", &system_prompt)
            .replace("{TEXT}", &task.text);
        
        let tokens = model
            .str_to_token(&prompt, llama_cpp_2::model::AddBos::Always)
            .expect("Failed to tokenize prompt");

        let mut n_eval = 0;
        let mut last_batch_tokens = 0;
        while n_eval < tokens.len() {
            let chunk_size = std::cmp::min(tokens.len() - n_eval, config.batch_size_limit);
            let mut batch = LlamaBatch::new(chunk_size, 1);
            for i in 0..chunk_size {
                let token = tokens[n_eval + i].clone();
                let is_last = (n_eval + i) == (tokens.len() - 1);
                batch.add(token, (n_eval + i) as i32, &[0], is_last).expect("Failed to add to batch");
            }
            ctx.decode(&mut batch).expect("Failed to decode prompt chunk");
            last_batch_tokens = chunk_size;
            n_eval += chunk_size;
        }

        let mut batch = LlamaBatch::new(1, 1);
        let mut generated_text = String::new();
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
            let candidates = ctx.candidates_ith(last_batch_tokens as i32 - 1);
            last_batch_tokens = 1;
            let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);
            
            candidates_p.apply_sampler(&mut sampler);
            let new_token_id = candidates_p.selected_token().expect("Failed to sample token");
            sampler.accept(new_token_id);

            // Check if end of generation
            if new_token_id == model.token_eos() || n_cur >= config.max_generate_tokens {
                break;
            }

            let token_str = crate::types::decode_token(&model, new_token_id, &mut decoder);
            generated_text.push_str(&token_str);

            batch.clear();
            batch.add(new_token_id, n_cur, &[0], true).unwrap();
            
            ctx.decode(&mut batch).expect("Failed to decode next token");
            n_cur += 1;
        }

        let trimmed_output = generated_text.trim();
        if !trimmed_output.is_empty() && !trimmed_output.contains("特になし") {
            // Rule of Silence: print only if output is notable
            println!("[Chunk {}]\n{}", task.index, trimmed_output);
            outputs.push((task.index, trimmed_output.to_string()));
        }
    }
    
    outputs
}
