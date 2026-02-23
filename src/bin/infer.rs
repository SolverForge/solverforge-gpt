/*!
Inference CLI for the solverforge-gpt MicroGPT Task Decomposer.

Loads all trained domain models and either auto-routes the task to the
best domain, or uses the domain you specify.

Usage:
  # Auto-detect domain
  cargo run --bin infer --release -- --weights weights/ --task "Build a login system"

  # Force a specific domain
  cargo run --bin infer --release -- --weights weights/ --domain swe --task "Build a login system"

  # Read task from stdin (pipe or interactive)
  echo "Plan a team offsite" | cargo run --bin infer --release -- --weights weights/

  # Interactive mode (prompts for input)
  cargo run --bin infer --release -- --weights weights/ --interactive
*/

use microgpt::{Domain, MultiDecomposer};
use std::io::{self, BufRead, Write};

struct Args {
    weights_dir: String,
    task: Option<String>,
    domain: Option<Domain>,
    interactive: bool,
    show_domain: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut weights_dir = "weights".to_string();
    let mut task = None;
    let mut domain = None;
    let mut interactive = false;
    let mut show_domain = true;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" | "-w" => {
                i += 1;
                weights_dir = args[i].clone();
            }
            "--task" | "-t" => {
                i += 1;
                task = Some(args[i].clone());
            }
            "--domain" | "-d" => {
                i += 1;
                domain = Some(Domain::from_str(&args[i]).unwrap_or_else(|| {
                    eprintln!("Unknown domain '{}'. Valid: work, swe, creative", args[i]);
                    std::process::exit(1);
                }));
            }
            "--interactive" | "-i" => {
                interactive = true;
            }
            "--no-domain-info" => {
                show_domain = false;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Args {
        weights_dir,
        task,
        domain,
        interactive,
        show_domain,
    }
}

fn print_help() {
    eprintln!(
        "Usage: infer [--weights <dir>] [--task <text>] [--domain <work|swe|creative>] [--interactive]"
    );
    eprintln!();
    eprintln!("  --weights / -w     Directory with trained weights (default: weights/)");
    eprintln!("  --task    / -t     Task description (or pipe via stdin)");
    eprintln!("  --domain  / -d     Force domain instead of auto-detecting");
    eprintln!("  --interactive / -i Prompt for tasks repeatedly until Ctrl+D");
    eprintln!("  --no-domain-info   Suppress domain routing info in output");
}

fn run_task(md: &MultiDecomposer, task: &str, forced_domain: Option<Domain>, show_domain: bool) {
    let detected = match forced_domain {
        Some(d) => d,
        None => md.detect_domain(task),
    };

    if show_domain {
        println!("Domain: {:?}", detected);
    }

    let result = if forced_domain.is_some() {
        md.split_in_domain(task, detected)
    } else {
        md.split(task)
    };

    match result {
        Ok(subtasks) => {
            for (i, sub) in subtasks.iter().enumerate() {
                println!("  {}. {}", i + 1, sub);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("(Model may need more training steps, or the task phrasing is unusual)");
        }
    }
}

fn main() {
    let args = parse_args();

    eprint!("Loading models from {}/ ... ", args.weights_dir);
    io::stderr().flush().unwrap();

    let md = MultiDecomposer::load(&args.weights_dir).unwrap_or_else(|e| {
        eprintln!("\nFailed to load models: {}", e);
        eprintln!("Run training first: cargo run --bin train --release -- --domain <d> --data data/<d>.txt");
        std::process::exit(1);
    });

    eprintln!("ready.");

    if args.interactive {
        // Interactive REPL
        println!("MicroGPT Task Decomposer — interactive mode. Ctrl+D to exit.\n");
        let stdin = io::stdin();
        loop {
            print!("Task> ");
            io::stdout().flush().unwrap();
            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    let task = line.trim();
                    if task.is_empty() {
                        continue;
                    }
                    run_task(&md, task, args.domain, args.show_domain);
                    println!();
                }
                Err(e) => {
                    eprintln!("Read error: {}", e);
                    break;
                }
            }
        }
    } else if let Some(ref task) = args.task {
        // Single task from --task flag
        run_task(&md, task, args.domain, args.show_domain);
    } else {
        // Read from stdin (pipe or single line)
        let stdin = io::stdin();
        let mut any = false;
        for line in stdin.lock().lines() {
            let task = line.expect("stdin read error");
            let task = task.trim().to_string();
            if task.is_empty() {
                continue;
            }
            run_task(&md, &task, args.domain, args.show_domain);
            any = true;
        }
        if !any {
            eprintln!(
                "No task provided. Use --task \"...\" or pipe via stdin, or --interactive for REPL."
            );
            print_help();
            std::process::exit(1);
        }
    }
}
