# solverforge-gpt

A minimal, zero-dependency, CPU-trainable GPT-2-style transformer written entirely in Rust. It does one thing: takes a task description and returns a list of subtasks.

Three domain-specific models are trained independently -- **work**, **software engineering**, and **creative** -- with a TF-IDF router that automatically selects the right one at inference time.

Inspired by [Karpathy's microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), reimplemented from scratch in Rust with a custom autograd engine.

## Why zero dependencies?

The entire stack -- autograd, transformer, BPE tokenizer, Adam optimizer, RNG, binary serialization -- is built on top of Rust `std` and nothing else. The `[dependencies]` section in `Cargo.toml` is empty. This keeps the binary small, the build fast, and the code auditable.

## Architecture

Each domain model is a decoder-only transformer (~1.5M parameters, ~4.3 MB on disk):

- 4 layers, 4 attention heads, 128-dimensional embeddings
- RMSNorm (pre-norm), GELU MLP, residual connections
- Block size of 512 tokens
- KV cache for efficient autoregressive inference
- Custom BPE tokenizer per domain (~1000 merges)

The domain router uses TF-IDF cosine similarity to classify incoming tasks.

## Quick start

Pre-trained weights for all three domains ship in `weights/`. To run inference immediately:

```bash
# build
cargo build --release

# single task
cargo run --bin infer --release -- --weights weights/ --task "Build a REST API for user authentication"

# interactive REPL
cargo run --bin infer --release -- --weights weights/ --interactive

# pipe from stdin
echo "Plan a team offsite" | cargo run --bin infer --release -- --weights weights/
```

Or use Make:

```bash
make all      # build release binaries
make infer    # launch interactive REPL
```

## Training

Train all three domains in parallel, then build the combined router:

```bash
make train
```

Train a single domain:

```bash
make train-swe
make train-work
make train-creative
```

Rebuild the router after training:

```bash
make router
```

### Training parameters

All configurable via environment variables or Make arguments:

| Variable | Default | Description |
|---|---|---|
| `STEPS` | 30000 | Training steps per domain |
| `VOCAB_SIZE` | 1000 | BPE vocabulary size |
| `LR` | 5e-4 | Learning rate |
| `EVAL_EVERY` | 500 | Evaluate every N steps |
| `WEIGHTS` | weights | Output directory |
| `DATA` | data | Input directory |

```bash
make train STEPS=50000 LR=3e-4
```

## Training data format

Training data lives in `data/` as plain text files (`swe.txt`, `work.txt`, `creative.txt`). Each example follows this format:

```
DOMAIN: swe
TASK: Build a REST API for user authentication
SUB: Define API schema and authentication endpoints
SUB: Implement JWT token generation and validation
SUB: Write integration tests for auth flow

DOMAIN: swe
TASK: Migrate database from PostgreSQL to MySQL
SUB: Audit existing schema and queries for compatibility
SUB: Rewrite PostgreSQL-specific SQL to MySQL dialect
```

Examples are separated by blank lines.

## Library usage

The crate exposes `Decomposer` (single domain) and `MultiDecomposer` (auto-routing) for use as a library:

```rust
use microgpt::{MultiDecomposer, Domain};

let md = MultiDecomposer::load("weights/").unwrap();

// auto-detect domain
let subtasks = md.split("Build a login page with OAuth").unwrap();
for s in &subtasks {
    println!("- {}", s);
}

// explicit domain
let subtasks = md.split_in_domain("Write a sonnet about rain", Domain::Creative).unwrap();
```

## Project structure

```
src/
  lib.rs          Public API (Decomposer, MultiDecomposer)
  model.rs        Transformer model (training + inference with KV cache)
  tensor.rs       Autograd engine (reverse-mode autodiff, Adam optimizer)
  tokenizer.rs    BPE tokenizer (training + encode/decode)
  router.rs       TF-IDF domain classifier
  rng.rs          xorshift64 PRNG with Box-Muller sampling
  bin/
    train.rs      Training CLI
    infer.rs      Inference CLI (single / interactive / pipe)
    build_router.rs  Router construction CLI
data/             Training datasets (swe, work, creative)
weights/          Pre-trained model weights and tokenizers
scripts/          Per-domain training shell scripts
```

## Make targets

```
make all              Build release binaries
make train            Train all domains + build router
make train-swe        Train software engineering domain
make train-work       Train work/business domain
make train-creative   Train creative domain
make router           Build domain router from training data
make infer            Interactive inference REPL
make clean            Delete all weights and tokenizer files
make help             Show all targets and variables
```
