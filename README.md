# lfm-cmd

A blazing fast, Apple Silicon (Metal) optimized CLI tool for streaming AI processing. Built entirely in Rust, it statically links and utilizes `llama.cpp`'s Metal backend for maximum parallel throughput and zero-copy performance on macOS M3 chips.

## Features

- **Apple Silicon Native**: Harnesses Metal GPU acceleration (`LLAMA_METAL=on`) to execute inference at lighting speeds.
- **Smart Token Chunking**: Reads `stdin` and uses binary search to safely dispatch semantic chunks up to 2,000 tokens cleanly aligned to sentence boundaries (`\n`, `。`, `.`).
- **Continuous Thread Pool Batching**: Dispatches chunks of text to parallel VRAM contexts over lock-free `crossbeam-channel` queues. 
- **Rule of Silence**: A core requirement—if the AI identifies "nothing special" or output contains "特になし", `lfm-cmd` stays entirely silent to maintain zero pollution of `stdout` in chained pipelines.
- **Zero-Copy Intent**: Optimized chunk reading minimizes GC jitter and runtime overhead.

## Requirements

You must be on macOS with an Apple Silicon processor (M1/M2/M3).
- **Xcode Command Line Tools**: Install via `xcode-select --install`
- **CMake**: Reuired to compile `llama.cpp` bindings. Install via `brew install cmake`
- **Rust Toolchain**: 1.80+ installed via `rustup`.

## Build Instructions

To build the tool natively with Metal support, run the following:

```bash
# 1. Clone the repository and cd into it (or just navigate to this workspace)
cd lfm-cmd

# 2. Build for release mode (crucial for performance)
cargo build --release
```

During compilation, `build.rs` will automatically:
- Trigger CMake to build the C/C++ libraries of `llama.cpp` using Metal support native to your Mac.
- Link the necessary Apple frameworks correctly (`Foundation`, `Metal`, `MetalPerformanceShaders`, `Accelerate`).

## Usage Synopsis

```bash
lfm-cmd [OPTIONS] -m <MODEL_PATH>
```

**Options:**
- `-m, --model <FILE>` : Path to the GGUF model file. If not provided, auto-extracts and loads the embedded LFM2.5 model.
- `-t, --tokens <COUNT>` : Target maximum tokens per chunk for semantic chunking (Default: `512`)
- `-w, --workers <VAL>` : Number of parallel workers/threads for context batching (Default: `2`)
- `-p, --prompt <TEXT>` : Custom system prompt to define the extraction logic.
- `-c, --config <FILE>` : Path to an advanced JSON configuration file to override hardcoded parameters.

### Example Pipeline

```bash
# Stream a large log file and process it with the standard prompt
cat large_server_logs.txt | ./target/release/lfm-cmd -w 4

# Only interesting anomalies found by the AI will be emitted:
# [Chunk 54]
# エラー: `connection timeout` が複数回発生。データベースへの接続が不安定です。
```

## Configuration (CLI vs Hardcoded)

`lfm-cmd` relies on two different configuration methods, depending on the performance impact and intended usage:

**1. Configurable via Command Line (Dynamic):**
These options can be tweaked at runtime using flags:
- `tokens`: Max tokens per chunk (`-t`, default: 512)
- `workers`: Number of parallel inference threads (`-w`, default: 2)
- `model`: GGUF model path (`-m`, default: embedded model)
- `prompt`: System prompt (`-p`)

**2. Advanced JSON Configuration (`--config`):**
For deeper control, you can pass a JSON configuration file via `--config configuration.json`. This overrides the application's default hyperparameters, without needing to recompile. Missing fields in your JSON will elegantly fallback to their native default values.

```json
{
    "meta_ctx_size": 8192,
    "main_ctx_size": 32768,
    "max_generate_tokens": 32768,
    "batch_size_limit": 4096,
    "sample_temp": 0.2,
    "sample_top_k": 50,
    "sample_top_p": 0.9,
    "penalty_repeat": 1.00,
    "penalty_last_n": 32
}
```

*Note: You can also override the inner prompt structures (`meta_prompt_template`, `worker_prompt_template`, `intermediate_reduce_prompt`, `final_reduce_prompt`) via this JSON.*
