use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::num::NonZeroU32;
use std::sync::Arc;

pub fn generate_meta_prompt(
    model: Arc<LlamaModel>,
    backend: Arc<LlamaBackend>,
    sample_text: String,
) -> String {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(8192).unwrap()))
        .with_n_batch(1024)
        .with_n_ubatch(1024);
    let mut ctx = model
        .new_context(backend.as_ref(), ctx_params)
        .expect("Failed to create meta prompt context");

    let prompt = format!(
        "<|startoftext|><|im_start|>system\nあなたは優秀なプロンプトエンジニアです。<|im_end|>\n<|im_start|>user\n以下のテキスト断片を分析し、元のテキストのジャンル（小説、技術論文、システムログ、議事録など）を判定してください。\nその後、このテキスト全体を最も美しく構造化して要約するための「AIへの指示書（システムプロンプト）」を作成してください。\n出力は150文字以内の「指示書」のみとし、解説や挨拶は一切含めないでください。\n【テキスト断片】\n{}<|im_end|>\n<|im_start|>assistant\n",
        sample_text
    );

    let tokens = model
        .str_to_token(&prompt, llama_cpp_2::model::AddBos::Always)
        .expect("Failed to tokenize meta prompt");

    let mut n_eval = 0;
    let batch_size_limit = 1024;
    let mut last_batch_tokens = 0;
    while n_eval < tokens.len() {
        let chunk_size = std::cmp::min(tokens.len() - n_eval, batch_size_limit);
        let mut batch = LlamaBatch::new(chunk_size, 1);
        for i in 0..chunk_size {
            let token = tokens[n_eval + i].clone();
            let is_last = (n_eval + i) == (tokens.len() - 1);
            batch.add(token, (n_eval + i) as i32, &[0], is_last).unwrap();
        }
        ctx.decode(&mut batch).expect("Failed to decode meta prompt");
        last_batch_tokens = chunk_size;
        n_eval += chunk_size;
    }

    let mut batch = LlamaBatch::new(1, 1);
    let mut n_cur = tokens.len() as i32;
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut generated_text = String::new();

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(0.2),
        LlamaSampler::top_k(50),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::penalties(32, 1.00, 0.05, 0.05),
        LlamaSampler::dist(1234),
    ]);
    // Isolate history: only penalize newly generated tokens, not the input prompt.
    // sampler.accept_many(&tokens);

    for _ in 0..150 {
        let candidates = ctx.candidates_ith(last_batch_tokens as i32 - 1);
        last_batch_tokens = 1;
        let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);
        
        candidates_p.apply_sampler(&mut sampler);
        let new_token_id = candidates_p.selected_token().expect("Failed to sample token");
        sampler.accept(new_token_id);

        if new_token_id == model.token_eos() || n_cur >= 32768 {
            break;
        }

        let token_str = crate::types::decode_token(&model, new_token_id, &mut decoder);
        generated_text.push_str(&token_str);

        batch.clear();
        batch.add(new_token_id, n_cur, &[0], true).unwrap();
        if ctx.decode(&mut batch).is_err() { break; }
        n_cur += 1;
    }
    
    generated_text.trim().to_string()
}
