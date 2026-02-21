/*!
MicroGPT Task Decomposer

A minimal, zero-dependency, CPU-trainable GPT that does one thing:
takes a task description and returns required subtasks.

Public API:
  split_task(task: &str) -> Vec<String>
  split_task_in_domain(task: &str, domain: Domain) -> Vec<String>

Models are trained per-domain (work, software, creative) and loaded lazily.
The domain router automatically selects the best model if not specified.

@karpathy (original Python GPT), extended to task decomposition in Rust.
*/

pub mod model;
pub mod rng;
pub mod router;
pub mod tensor;
pub mod tokenizer;

pub use model::Config;
pub use model::Model;
pub use rng::Rng;
pub use router::Domain;
pub use router::Router;
pub use tokenizer::{parse_training_data, Example, Tokenizer};

use crate::model::Model as M;
use crate::router::Router as R;
use crate::tokenizer::Tokenizer as Tok;

// ---------------------------------------------------------------------------
// Simple non-static API: caller manages model instances
// ---------------------------------------------------------------------------

/// A loaded task decomposer for a single domain.
pub struct Decomposer {
    pub domain: Domain,
    tokenizer: Tok,
    model: M,
}

impl Decomposer {
    /// Load a decomposer from weights directory.
    /// Expects `{dir}/{domain}.bin` and `{dir}/{domain}.tok`.
    pub fn load(dir: &str, domain: Domain) -> Result<Self, String> {
        let name = domain.name();
        let tok_path = format!("{}/{}.tok", dir, name);
        let tok = Tok::load(&tok_path)
            .map_err(|e| format!("Failed to load tokenizer {}: {}", tok_path, e))?;

        let model_path = format!("{}/{}.bin", dir, name);
        let mut rng = Rng::new(0);
        let model = M::load(&model_path, &mut rng)
            .map_err(|e| format!("Failed to load model {}: {}", model_path, e))?;

        Ok(Decomposer {
            domain,
            tokenizer: tok,
            model,
        })
    }

    /// Split a task description into subtasks.
    pub fn split(&self, task: &str) -> Result<Vec<String>, String> {
        let mut rng = Rng::new(42);
        let prompt = self.tokenizer.encode_prompt(task);
        let generated = self.model.generate(&prompt, 200, 0.7, &mut rng);
        parse_subtasks(&generated, &self.tokenizer)
    }
}

/// A multi-domain task decomposer with automatic routing.
pub struct MultiDecomposer {
    decomposers: Vec<Decomposer>,
    router: Option<R>,
}

impl MultiDecomposer {
    /// Load all available domain models from a directory.
    pub fn load(dir: &str) -> Result<Self, String> {
        let mut decomposers = vec![];
        for &domain in Domain::all() {
            match Decomposer::load(dir, domain) {
                Ok(d) => decomposers.push(d),
                Err(e) => eprintln!("Skipping domain {:?}: {}", domain, e),
            }
        }
        if decomposers.is_empty() {
            return Err("No domain models found".to_string());
        }

        let router_path = format!("{}/router.bin", dir);
        let router = R::load(&router_path).ok();

        Ok(MultiDecomposer {
            decomposers,
            router,
        })
    }

    /// Split a task, auto-detecting the domain.
    pub fn split(&self, task: &str) -> Result<Vec<String>, String> {
        let domain = self.detect_domain(task);
        self.split_in_domain(task, domain)
    }

    /// Split a task in a specific domain.
    pub fn split_in_domain(&self, task: &str, domain: Domain) -> Result<Vec<String>, String> {
        let decomposer = self
            .decomposers
            .iter()
            .find(|d| d.domain == domain)
            .or_else(|| self.decomposers.first())
            .ok_or("No decomposers loaded")?;
        decomposer.split(task)
    }

    /// Detect domain, falling back to the first available.
    pub fn detect_domain(&self, task: &str) -> Domain {
        if let Some(r) = &self.router {
            // Only return domains we actually have loaded
            let scores = r.predict_with_scores(task);
            for (d, _) in scores {
                if self.decomposers.iter().any(|dec| dec.domain == d) {
                    return d;
                }
            }
        }
        self.decomposers[0].domain
    }
}

// ---------------------------------------------------------------------------
// Subtask parser
// ---------------------------------------------------------------------------

fn parse_subtasks(tokens: &[u32], tok: &Tok) -> Result<Vec<String>, String> {
    use crate::tokenizer::{CSUB_ID, EOS_ID, OSUB_ID};

    let mut subtasks = vec![];
    let mut current: Vec<u32> = vec![];
    let mut in_sub = false;

    for &t in tokens {
        if t == EOS_ID {
            break;
        }

        if t == OSUB_ID {
            in_sub = true;
            current.clear();
        } else if t == CSUB_ID {
            if in_sub && !current.is_empty() {
                let text = tok.decode(&current).trim().to_string();
                if !text.is_empty() {
                    subtasks.push(text);
                }
            }
            in_sub = false;
            current.clear();
        } else if in_sub {
            current.push(t);
        }
    }

    if subtasks.is_empty() {
        return Err("Model generated no subtasks".to_string());
    }

    Ok(subtasks)
}
