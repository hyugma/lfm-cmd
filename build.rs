use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-env=LLAMA_METAL=on");

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = PathBuf::from(out_dir).join("LFM2.5-1.2B-Instruct-Q4_K_M.gguf");

    if !dest_path.exists() {
        println!("cargo:warning=Downloading LFM2.5 1.2B model (730MB) to OUT_DIR for embedding...");
        let status = Command::new("curl")
            .args(&[
                "-L",
                "-o",
                dest_path.to_str().unwrap(),
                "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
            ])
            .status()
            .expect("Failed to execute curl");

        if !status.success() {
            panic!("Failed to download model file.");
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
}
