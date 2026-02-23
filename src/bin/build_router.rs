/*!
Build the domain router from all three training data files.

This must be run after training all three domain models, so the router
knows about all domains. Running training in parallel means each training
run only has one domain's data and writes a single-domain router — this
binary fixes that by reading all three and producing a combined router.bin.

Usage:
  cargo run --bin build_router --release -- \
    --data-dir data/ \
    --out weights/

Or with custom file names:
  cargo run --bin build_router --release -- \
    --swe  data/swe.txt \
    --work data/work.txt \
    --creative data/creative.txt \
    --out weights/
*/

use microgpt::{Domain, Router, parse_training_data};

struct Args {
    swe_path: String,
    work_path: String,
    creative_path: String,
    out_dir: String,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut swe_path = String::new();
    let mut work_path = String::new();
    let mut creative_path = String::new();
    let mut out_dir = "weights".to_string();
    let mut data_dir = String::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--swe" => {
                i += 1;
                swe_path = args[i].clone();
            }
            "--work" => {
                i += 1;
                work_path = args[i].clone();
            }
            "--creative" => {
                i += 1;
                creative_path = args[i].clone();
            }
            "--out" => {
                i += 1;
                out_dir = args[i].clone();
            }
            "--data-dir" => {
                i += 1;
                data_dir = args[i].clone();
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                eprintln!(
                    "Usage: build_router [--data-dir <dir>] [--swe <path>] [--work <path>] [--creative <path>] [--out <dir>]"
                );
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // If --data-dir given, fill in defaults for any missing paths
    if !data_dir.is_empty() {
        if swe_path.is_empty() {
            swe_path = format!("{}/swe.txt", data_dir);
        }
        if work_path.is_empty() {
            work_path = format!("{}/work.txt", data_dir);
        }
        if creative_path.is_empty() {
            creative_path = format!("{}/creative.txt", data_dir);
        }
    }

    Args {
        swe_path,
        work_path,
        creative_path,
        out_dir,
    }
}

fn load_domain(path: &str, domain: Domain, pairs: &mut Vec<(String, Domain)>) {
    if path.is_empty() {
        eprintln!("  Skipping {:?}: no path provided", domain);
        return;
    }
    match std::fs::read_to_string(path) {
        Ok(raw) => {
            let examples = parse_training_data(&raw);
            println!("  {:?}: {} examples from {}", domain, examples.len(), path);
            for e in examples {
                pairs.push((e.task, domain));
            }
        }
        Err(e) => {
            eprintln!("  Skipping {:?}: cannot read {}: {}", domain, path, e);
        }
    }
}

fn main() {
    let args = parse_args();

    println!("=== MicroGPT — Building Domain Router ===");
    println!("Output dir: {}", args.out_dir);

    std::fs::create_dir_all(&args.out_dir).expect("Could not create output directory");

    let mut pairs: Vec<(String, Domain)> = vec![];
    load_domain(&args.swe_path, Domain::Software, &mut pairs);
    load_domain(&args.work_path, Domain::Work, &mut pairs);
    load_domain(&args.creative_path, Domain::Creative, &mut pairs);

    if pairs.is_empty() {
        eprintln!("No training examples loaded. Provide at least one data file.");
        std::process::exit(1);
    }

    println!("\nTotal examples: {}", pairs.len());
    println!("Training router...");

    let router = Router::train(&pairs);

    let router_path = format!("{}/router.bin", args.out_dir);
    router.save(&router_path).expect("Could not save router");
    println!("Saved: {}", router_path);

    // Quick sanity check
    println!("\nSanity check:");
    let tests = [
        ("implement a REST API with authentication", Domain::Software),
        ("plan the quarterly OKR review meeting", Domain::Work),
        ("write a short story about a robot", Domain::Creative),
    ];
    for (task, expected) in tests {
        let predicted = router.predict(task);
        let ok = if predicted == expected {
            "OK"
        } else {
            "MISMATCH"
        };
        println!(
            "  [{ok}] \"{task}\" => {:?} (expected {:?})",
            predicted, expected
        );
    }
}
