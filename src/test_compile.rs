use llama_cpp_2::model::LlamaModel;

fn test(model: &LlamaModel) {
    let tokens = model.str_to_token_with_special("hello", llama_cpp_2::model::AddBos::Never, true);
}
