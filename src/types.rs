pub struct ChunkTask {
    pub index: usize,
    pub text: String,
}

pub fn decode_token(
    model: &llama_cpp_2::model::LlamaModel,
    token: llama_cpp_2::token::LlamaToken,
    decoder: &mut encoding_rs::Decoder,
) -> String {
    let bytes = model.token_to_piece_bytes(token, 32, false, None).unwrap_or_default();
    let mut output_piece = String::with_capacity(32);
    let _ = decoder.decode_to_string(&bytes, &mut output_piece, false);
    output_piece
}
