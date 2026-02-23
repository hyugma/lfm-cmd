mod cli;
mod types;
mod config;
mod prompts;
mod worker;
mod reducer;
mod chunker;

use clap::Parser;
use crossbeam_channel::bounded;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::fs;
use std::sync::Arc;
use std::thread;

use cli::Args;
use config::AppConfig;
use types::ChunkTask;
use worker::worker_loop;
use reducer::run_reducer;
use chunker::parse_and_chunk;

static EMBEDDED_MODEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"));

extern "C" fn void_log(
    _level: llama_cpp_sys_2::ggml_log_level,
    _text: *const std::ffi::c_char,
    _user_data: *mut std::ffi::c_void,
) {}

fn main() {
    // Suppress all llama-cpp logs (especially the noisy hardware logs on startup/decode)
    unsafe {
        llama_cpp_sys_2::llama_log_set(Some(void_log), std::ptr::null_mut());
    }

    let args = Args::parse();

    // Determine model path: extract embedded if not provided
    let model_path = if let Some(ref path) = args.model {
        path.clone()
    } else {
        let temp_dir = std::env::temp_dir();
        let filename = format!("lfm2.5-1.2b-instruct-q4-v{}.gguf", env!("CARGO_PKG_VERSION"));
        let extracted_path = temp_dir.join(&filename);
        if !extracted_path.exists() {
            fs::write(&extracted_path, EMBEDDED_MODEL)
                .expect("Failed to write embedded model to tmp directory");
        }
        extracted_path
    };

    // 1. Init backend (Metal enabled by build.rs and features)
    let backend = Arc::new(LlamaBackend::init().expect("Failed to initialize llama backend"));

    // 2. Load model
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .expect("Failed to load model from file");
    let model = Arc::new(model);

    // 2.5 Load AppConfig
    let app_config = if let Some(ref config_path) = args.config {
        let config_str = fs::read_to_string(config_path)
            .expect(&format!("Failed to read config file from {:?}", config_path));
        serde_json::from_str(&config_str)
            .expect("Failed to parse config file as JSON")
    } else {
        AppConfig::default()
    };
    let app_config = Arc::new(app_config);

    // 3. Set up the queue for workers and reducer
    let (worker_tx, worker_rx) = bounded::<ChunkTask>(args.workers * 2);
    let (reducer_tx, reducer_rx) = bounded::<(usize, String)>(args.workers * 2);

    let mut worker_handles = vec![];
    for id in 0..args.workers {
        let rx_clone = worker_rx.clone();
        let tx_clone = reducer_tx.clone();
        let model_clone = model.clone();
        let backend_clone = backend.clone();
        let system_prompt = args.prompt.clone();
        let config_clone = app_config.clone();
        worker_handles.push(thread::spawn(move || {
            let outputs = worker_loop(id, model_clone, backend_clone, rx_clone, system_prompt, config_clone);
            for out in outputs {
                let _ = tx_clone.send(out);
            }
        }));
    }

    // Spawn Reducer Thread
    let reducer_model = model.clone();
    let reducer_backend = backend.clone();
    let reducer_prompt = args.prompt.clone();
    let reducer_config = app_config.clone();
    let reducer_handle = thread::spawn(move || {
        run_reducer(reducer_model, reducer_backend, reducer_prompt, reducer_rx, reducer_config);
    });

    // Drop original receivers
    drop(worker_rx);

    // 4. Smart Chunking pipeline (stdin)
    parse_and_chunk(&model, worker_tx.clone(), args.tokens);

    // Close channel so workers finish their tasks and exit
    drop(worker_tx);

    for handle in worker_handles {
        let _ = handle.join();
    }
    
    // Close reducer channel when workers are done
    drop(reducer_tx);
    let _ = reducer_handle.join();
    
    // safe Drop: ARC unrefs and llama_model_free / llama_free are called automatically.
}
