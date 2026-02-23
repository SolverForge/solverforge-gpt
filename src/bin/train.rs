/*!
Training binary for domain-specific task decomposition models.

Usage:
  cargo run --bin train --release -- \
    --domain swe \
    --data data/swe.txt \
    --out weights/ \
    --steps 5000 \
    --vocab-size 2000

This will produce:
  weights/swe.bin    (model weights)
  weights/swe.tok    (tokenizer)
  weights/router.bin (domain router, updated if multiple domains trained)

Data format (data/swe.txt):
  DOMAIN: swe
  TASK: Build a REST API for user authentication
  SUB: Define API schema and authentication endpoints
  SUB: Implement JWT token generation and validation
  SUB: Add middleware for protected route authorization
  SUB: Write integration tests for auth flows

  DOMAIN: swe
  TASK: Set up CI/CD pipeline for a monorepo
  SUB: Choose CI platform and configure repository access
  ...
  (blank line separates examples)
*/

use microgpt::tensor::{Adam, Tensor};
use microgpt::{Config, Domain, Example, Model, Rng, Router, Tokenizer, parse_training_data};
use std::io::Write;
use std::time::Instant;

struct Args {
    domain: Domain,
    data_path: String,
    out_dir: String,
    steps: usize,
    vocab_size: usize,
    lr: f64,
    eval_every: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut domain = Domain::Software;
    let mut data_path = String::new();
    let mut out_dir = "weights".to_string();
    let mut steps = 3000usize;
    let mut vocab_size = 2000usize;
    let mut lr = 3e-4f64;
    let mut eval_every = 100usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--domain" => {
                i += 1;
                domain = Domain::from_str(&args[i]).unwrap_or_else(|| {
                    eprintln!("Unknown domain '{}'. Valid: work, swe, creative", args[i]);
                    std::process::exit(1);
                });
            }
            "--data" => {
                i += 1;
                data_path = args[i].clone();
            }
            "--out" => {
                i += 1;
                out_dir = args[i].clone();
            }
            "--steps" => {
                i += 1;
                steps = args[i].parse().expect("--steps must be int");
            }
            "--vocab-size" => {
                i += 1;
                vocab_size = args[i].parse().expect("--vocab-size must be int");
            }
            "--lr" => {
                i += 1;
                lr = args[i].parse().expect("--lr must be float");
            }
            "--eval-every" => {
                i += 1;
                eval_every = args[i].parse().expect("--eval-every must be int");
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if data_path.is_empty() {
        eprintln!("Error: --data <path> is required");
        print_usage();
        std::process::exit(1);
    }

    Args {
        domain,
        data_path,
        out_dir,
        steps,
        vocab_size,
        lr,
        eval_every,
    }
}

fn print_usage() {
    eprintln!(
        "Usage: train --domain <work|swe|creative> --data <path> [--out <dir>] [--steps N] [--vocab-size N] [--lr F]"
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = parse_args();

    println!("=== MicroGPT Task Decomposer — Training ===");
    println!("Domain:     {:?}", args.domain);
    println!("Data:       {}", args.data_path);
    println!("Out dir:    {}", args.out_dir);
    println!("Steps:      {}", args.steps);
    println!("Vocab size: {}", args.vocab_size);
    println!("LR:         {}", args.lr);

    std::fs::create_dir_all(&args.out_dir).expect("Could not create output directory");

    let raw = std::fs::read_to_string(&args.data_path).unwrap_or_else(|e| {
        eprintln!("Cannot read {}: {}", args.data_path, e);
        std::process::exit(1);
    });

    let examples = parse_training_data(&raw);
    println!("Examples loaded: {}", examples.len());

    if examples.is_empty() {
        eprintln!(
            "No examples found in {}. Check format (DOMAIN:/TASK:/SUB: lines).",
            args.data_path
        );
        std::process::exit(1);
    }

    // Train/val split
    let mut rng = Rng::new(42);
    let mut indices: Vec<usize> = (0..examples.len()).collect();
    rng.shuffle(&mut indices);

    let val_size = (examples.len() / 10).max(1).min(200);
    let val_indices: Vec<usize> = indices[..val_size].to_vec();
    let mut train_indices: Vec<usize> = indices[val_size..].to_vec();

    println!("Train: {}  Val: {}", train_indices.len(), val_indices.len());

    // Build corpus for BPE training (all task + subtask text)
    println!(
        "\n[1/4] Training BPE tokenizer (vocab_size={})...",
        args.vocab_size
    );
    let corpus: String = examples
        .iter()
        .map(|e| {
            let mut s = e.task.clone();
            for sub in &e.subtasks {
                s.push('\n');
                s.push_str(sub);
            }
            s
        })
        .collect::<Vec<_>>()
        .join("\n");

    let tok = Tokenizer::train(&corpus, args.vocab_size);
    println!("  Vocab size: {}", tok.vocab_size());

    // Save tokenizer
    let tok_path = format!("{}/{}.tok", args.out_dir, args.domain.name());
    tok.save(&tok_path).expect("Could not save tokenizer");
    println!("  Saved: {}", tok_path);

    // Encode all training examples
    println!("\n[2/4] Encoding training data...");
    let encoded: Vec<(Vec<u32>, Vec<bool>)> =
        examples.iter().map(|e| encode_example(e, &tok)).collect();

    let max_len = encoded.iter().map(|(ids, _)| ids.len()).max().unwrap_or(0);
    println!("  Max sequence length: {}", max_len);

    // Initialize model
    println!("\n[3/4] Initializing model...");
    let cfg = Config::task_decomposer(tok.vocab_size());
    let model = Model::new(cfg.clone(), &mut rng);
    println!("  Params: {}", model.num_params());
    println!("  Config: {:?}", cfg);

    // Optimizer — total param count for Adam buffer
    let total_params: usize = model.params().iter().map(|p| p.len()).sum();
    let mut adam = Adam::new(total_params, args.lr);

    // Training loop
    println!("\n[4/4] Training for {} steps...", args.steps);
    let start = Instant::now();
    let mut train_step = 0usize;
    let mut best_val_loss = f64::INFINITY;

    for step in 0..args.steps {
        // Pick a training example
        let idx = train_indices[train_step % train_indices.len()];
        train_step += 1;
        if train_step % train_indices.len() == 0 {
            // Reshuffle at epoch boundary
            rng.shuffle(&mut train_indices);
        }

        let (tokens, mask) = &encoded[idx];
        if tokens.len() < 2 {
            continue;
        }

        // Forward
        let loss = model.forward_train(tokens, mask);
        let loss_val = loss.item();

        // Backward
        loss.backward();

        // Optimizer step with linear LR decay
        let lr_scale = 1.0 - step as f64 / args.steps as f64;
        let lr_scale = lr_scale.max(0.1); // don't decay below 10% of initial LR
        adam.step_params(&model.params(), lr_scale);

        // Print progress
        print!(
            "\r  step {:5}/{} | loss {:.4} | lr {:.2e} | elapsed {:.0}s",
            step + 1,
            args.steps,
            loss_val,
            args.lr * lr_scale,
            start.elapsed().as_secs_f64()
        );
        std::io::stdout().flush().unwrap();

        // Validation
        if (step + 1) % args.eval_every == 0 || step == args.steps - 1 {
            let val_loss = evaluate(&model, &encoded, &val_indices);
            println!();
            println!(
                "  [eval] step {} | train_loss {:.4} | val_loss {:.4}",
                step + 1,
                loss_val,
                val_loss
            );

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                let model_path = format!("{}/{}.bin", args.out_dir, args.domain.name());
                model.save(&model_path).expect("Could not save model");
                println!("  [saved] {} (val_loss={:.4})", model_path, best_val_loss);
            }

            // Sample to show qualitative progress
            if (step + 1) % (args.eval_every * 5) == 0 {
                sample_and_print(&model, &tok, &examples, &mut rng);
            }
        }
    }

    println!("\n\nTraining complete. Best val loss: {:.4}", best_val_loss);

    println!("\nUpdating domain router...");
    update_router(&args.out_dir, &examples, args.domain);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn encode_example(example: &Example, tok: &Tokenizer) -> (Vec<u32>, Vec<bool>) {
    let tokens = tok.encode_training_example(&example.task, &example.subtasks);

    // Label mask: compute loss only on subtask tokens (not on task input)
    // Find where </task> appears -- everything after that is fair game
    let ctask = microgpt::tokenizer::CTASK_ID;

    let task_end = tokens.iter().position(|&t| t == ctask).unwrap_or(0);

    // mask[i] = true means: compute loss at position i (predicting tokens[i+1])
    let mask: Vec<bool> = (0..tokens.len().saturating_sub(1))
        .map(|i| i >= task_end) // only predict output tokens
        .collect();

    (tokens, mask)
}

fn evaluate(model: &Model, encoded: &[(Vec<u32>, Vec<bool>)], val_indices: &[usize]) -> f64 {
    let mut total = 0.0;
    let mut count = 0;

    for &idx in val_indices.iter().take(50) {
        // limit for speed
        let (tokens, mask) = &encoded[idx];
        if tokens.len() < 2 {
            continue;
        }
        let loss = model.forward_train(tokens, mask);
        total += loss.item();
        Tensor::clear_tape(); // Free the eval computation graph (no backward needed)
        count += 1;
    }

    if count == 0 {
        f64::INFINITY
    } else {
        total / count as f64
    }
}

fn sample_and_print(model: &Model, tok: &Tokenizer, examples: &[Example], rng: &mut Rng) {
    println!("\n  --- Sample outputs ---");
    // Pick 2 examples from the set and run inference
    for example in examples.iter().take(2) {
        let prompt = tok.encode_prompt(&example.task);
        let generated = model.generate(&prompt, 150, 0.7, rng);

        println!("  TASK: {}", example.task);

        // Parse subtasks from generated tokens
        let mut in_sub = true; // prompt ends with <sub>
        let mut current: Vec<u32> = vec![];
        let mut subtask_count = 0;

        for &t in &generated {
            if t == microgpt::tokenizer::EOS_ID {
                break;
            }
            if t == microgpt::tokenizer::OSUB_ID {
                in_sub = true;
                current.clear();
            } else if t == microgpt::tokenizer::CSUB_ID {
                if in_sub && !current.is_empty() {
                    subtask_count += 1;
                    println!("    {}. {}", subtask_count, tok.decode(&current).trim());
                }
                in_sub = false;
                current.clear();
            } else if in_sub {
                current.push(t);
            }
        }

        if subtask_count == 0 {
            // Model hasn't learned the format yet -- show raw tokens
            println!(
                "    (no structured output yet -- {} tokens generated)",
                generated.len()
            );
        }
        println!();
    }
    println!("  ---");
}

fn update_router(out_dir: &str, examples: &[Example], domain: Domain) {
    // Collect all available domain data
    let training_pairs: Vec<(String, Domain)> =
        examples.iter().map(|e| (e.task.clone(), domain)).collect();

    // If router already exists, we'd ideally merge -- for now just rebuild from current data
    let router = Router::train(&training_pairs);
    let router_path = format!("{}/router.bin", out_dir);
    router.save(&router_path).expect("Could not save router");
    println!("  Saved router: {}", router_path);
}
