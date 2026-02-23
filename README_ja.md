# lfm-cmd

驚異的な速さを誇る、Apple Silicon (Metal) に最適化されたストリーミングAI処理用のCLIツールです。完全にRustで構築されており、`llama.cpp` の Metal バックエンドを静的リンクして利用することで、macOS M3チップ上で最大の並列スループットとゼロコピー（Zero-copy）パフォーマンスを実現します。

## 機能・特徴

- **Apple Silicon ネイティブ**: Metal のGPUアクセラレーション (`LLAMA_METAL=on`) をフル活用し、光の速さで推論を実行します。
- **スマート・トークンチャンキング**: `stdin`（標準入力）を読み込み、バイナリ探索を用いて元の文の境界（`\n`, `。`, `.`）を崩さずに最大2,000トークンまでの意味的な固まり（チャンク）へ安全に分割します。
- **スレッドプールの連続バッチング**: 分割されたテキストチャンクを、ロックフリーな `crossbeam-channel` キューを通して複数のVRAMコンテキストへ並列にディスパッチします。
- **「無視」の原則 (Rule of Silence)**: 重要な設計要件として、もしAIが「特に書くことがない」と判断した場合や、出力に「特になし」が含まれる場合、`lfm-cmd` は**完全に沈黙**します。これにより、シェルパイプラインで繋いだ際に `stdout` が一切汚染されません。
- **ゼロコピー志向**: チャンク読み込みの最適化により、ガベージコレクションのジッターやランタイムのオーバーヘッドを最小限に抑えています。

## 動作要件

Apple Silicon プロセッサ (M1/M2/M3) を搭載した macOS 環境が必須です。
- **Xcode Command Line Tools**: `xcode-select --install` でインストール。
- **CMake**: `llama.cpp` のバインディングをコンパイルするために必要です。`brew install cmake` 等でインストールしてください。
- **Rust Toolchain**: `rustup` 経由でインストール済みの Rust 1.80 以上。

## ビルド方法

Metalをサポートしたネイティブビルドを行うには、以下のコマンドを実行します：

```bash
# 1. リポジトリをクローンし、ディレクトリに移動します
cd lfm-cmd

# 2. release モードでビルドします（パフォーマンスを発揮するために必須です）
cargo build --release
```

コンパイル中、`build.rs` が自動的に以下の処理を行います：
- Mac ネイティブの Metal サポートを有効にして、CMake 経由で `llama.cpp` の C/C++ ライブラリをビルドします。
- 必要な Apple フレームワーク（`Foundation`, `Metal`, `MetalPerformanceShaders`, `Accelerate`）を正しくリンクします。

## 使用方法 (Usage)

```bash
lfm-cmd [OPTIONS] -m <MODEL_PATH>
```

**オプション一覧:**
- `-m, --model <FILE>` : GGUFモデルファイルのパス。指定しない場合、バイナリ内に埋め込まれた LFM2.5 モデルを自動で抽出し読み込みます。
- `-t, --tokens <COUNT>` : 意味的チャンキングを行う際の、1チャンクあたりの最大トークン数（デフォルト: `512`）
- `-w, --workers <VAL>` : コンテキストバッチング処理を行う並列ワーカー/スレッドの数（デフォルト: `2`）
- `-p, --prompt <TEXT>` : 抽出・要約のロジックとして与えるカスタムのシステムプロンプト。
- `-c, --config <FILE>` : ハードコードされたパラメータを上書きするための、JSON構成ファイルのパス。

### 実行例 (パイプライン)

```bash
# 大きなサーバーログファイルを流し込み、標準のプロンプトで処理する
cat large_server_logs.txt | ./target/release/lfm-cmd -w 4

# AIが発見した「興味深い異常」のみが標準出力エミュレートされます：
# [Chunk 54]
# エラー: `connection timeout` が複数回発生。データベースへの接続が不安定です。
```

## 設定方法 (CLI vs ハードコード構成)

`lfm-cmd` はパフォーマンスへの影響や用途に合わせて、2つの異なる構成（コンフィグ）手法を採用しています：

**1. コマンドラインからの動的設定:**
以下のオプションは、実行時にフラグとして柔軟に変更できます。
- `tokens`: チャンクごとの最大トークン数 (`-t`, デフォルト: 512)
- `workers`: 並列推論スレッド数 (`-w`, デフォルト: 2)
- `model`: GGUFモデルのパス (`-m`, デフォルト: 内蔵モデル)
- `prompt`: システムプロンプト (`-p`)

**2. 拡張 JSON 構成ファイル (`--config`):**
より深い階層の設定を制御したい場合、`--config configuration.json` を通してJSONファイルを渡すことができます。これにより、アプリのデフォルトのハイパーパラメータを「再コンパイルなしで」上書きできます。JSON内で指定を省いた項目は、自動的にネイティブのデフォルト値（フォールバック）が適用されます。

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

*注: 内部のプロンプト構造体（`meta_prompt_template`, `worker_prompt_template`, `intermediate_reduce_prompt`, `final_reduce_prompt`）もこのJSONファイル経由で柔軟に上書き可能です。*
