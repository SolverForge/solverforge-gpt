/*!
TF-IDF domain classifier.

Given a task string, returns the most likely Domain. This is a simple
bag-of-words cosine similarity classifier -- no neural network, no autograd.
Each domain has a TF-IDF weight vector over a shared vocabulary.

The classifier is trained from the same data as the generative models.
Training output is a small binary file (~50-100KB) that ships with the app.

Zero external dependencies — pure Rust std only.
*/

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Domain enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Domain {
    Work,
    Software,
    Creative,
}

impl Domain {
    pub fn name(&self) -> &'static str {
        match self {
            Domain::Work => "work",
            Domain::Software => "swe",
            Domain::Creative => "creative",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "work" | "professional" | "business" => Some(Domain::Work),
            "swe" | "software" | "engineering" | "tech" => Some(Domain::Software),
            "creative" | "content" | "media" => Some(Domain::Creative),
            _ => None,
        }
    }

    pub fn all() -> &'static [Domain] {
        &[Domain::Work, Domain::Software, Domain::Creative]
    }
}

// ---------------------------------------------------------------------------
// TF-IDF Router
// ---------------------------------------------------------------------------

pub struct Router {
    // vocabulary: word -> index
    vocab: HashMap<String, usize>,
    // per-domain TF-IDF weight vectors (indexed by vocab word index)
    domain_vecs: Vec<(Domain, Vec<f64>)>,
}

impl Router {
    /// Train from a list of (text, domain) pairs.
    pub fn train(examples: &[(String, Domain)]) -> Self {
        // Build vocabulary from all words
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for (text, _) in examples {
            for word in tokenize_words(text) {
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }

        // Keep words that appear at least twice (prune noise)
        let vocab: HashMap<String, usize> = word_counts
            .iter()
            .filter(|(_, c)| **c >= 2)
            .enumerate()
            .map(|(i, (w, _))| (w.clone(), i))
            .collect();

        let vsize = vocab.len();
        let n_docs = examples.len() as f64;

        // Document frequency per word
        let mut df = vec![0.0f64; vsize];
        for (text, _) in examples {
            let words: std::collections::HashSet<String> =
                tokenize_words(text).into_iter().collect();
            for word in words {
                if let Some(&idx) = vocab.get(&word) {
                    df[idx] += 1.0;
                }
            }
        }

        // IDF
        let idf: Vec<f64> = df
            .iter()
            .map(|&d| {
                if d > 0.0 {
                    (n_docs / d).ln() + 1.0
                } else {
                    0.0
                }
            })
            .collect();

        // Group examples by domain
        let mut domain_docs: HashMap<Domain, Vec<&String>> = HashMap::new();
        for (text, domain) in examples {
            domain_docs.entry(*domain).or_default().push(text);
        }

        // Build TF-IDF vector per domain (centroid of all docs in that domain)
        let mut domain_vecs = vec![];
        for &domain in Domain::all() {
            let docs = match domain_docs.get(&domain) {
                Some(d) => d,
                None => continue,
            };

            let mut centroid = vec![0.0f64; vsize];

            for text in docs {
                // Term frequency for this doc
                let mut tf: HashMap<usize, f64> = HashMap::new();
                let words = tokenize_words(text);
                let n = words.len() as f64;
                for word in words {
                    if let Some(&idx) = vocab.get(&word) {
                        *tf.entry(idx).or_insert(0.0) += 1.0 / n;
                    }
                }
                // TF-IDF
                for (idx, tf_val) in tf {
                    centroid[idx] += tf_val * idf[idx];
                }
            }

            // Normalize centroid by number of docs
            let nd = docs.len() as f64;
            centroid.iter_mut().for_each(|v| *v /= nd);

            // L2 normalize
            l2_normalize(&mut centroid);
            domain_vecs.push((domain, centroid));
        }

        Router { vocab, domain_vecs }
    }

    /// Predict the domain for a given task string.
    pub fn predict(&self, task: &str) -> Domain {
        let vec = self.vectorize(task);
        self.domain_vecs
            .iter()
            .map(|(d, dv)| (*d, cosine(&vec, dv)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(d, _)| d)
            .unwrap_or(Domain::Work)
    }

    /// Predict with confidence scores for all domains
    pub fn predict_with_scores(&self, task: &str) -> Vec<(Domain, f64)> {
        let vec = self.vectorize(task);
        let mut scores: Vec<(Domain, f64)> = self
            .domain_vecs
            .iter()
            .map(|(d, dv)| (*d, cosine(&vec, dv)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores
    }

    fn vectorize(&self, text: &str) -> Vec<f64> {
        let vsize = self.vocab.len();
        let mut vec = vec![0.0f64; vsize];
        let words = tokenize_words(text);
        let n = words.len() as f64;
        if n == 0.0 {
            return vec;
        }
        for word in words {
            if let Some(&idx) = self.vocab.get(&word) {
                vec[idx] += 1.0 / n;
            }
        }
        l2_normalize(&mut vec);
        vec
    }

    /// Serialize to bytes
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = vec![];

        // vocab: count + (len, bytes) entries
        let vc = self.vocab.len() as u32;
        out.extend_from_slice(&vc.to_le_bytes());
        // Sort by index for deterministic order
        let mut vocab_sorted: Vec<(&String, &usize)> = self.vocab.iter().collect();
        vocab_sorted.sort_by_key(|&(_, i)| i);
        for (word, _) in vocab_sorted {
            let b = word.as_bytes();
            out.extend_from_slice(&(b.len() as u16).to_le_bytes());
            out.extend_from_slice(b);
        }

        // domain vectors
        let dc = self.domain_vecs.len() as u32;
        out.extend_from_slice(&dc.to_le_bytes());
        for (domain, vec) in &self.domain_vecs {
            let name = domain.name().as_bytes();
            out.extend_from_slice(&(name.len() as u8).to_le_bytes());
            out.extend_from_slice(name);
            for &v in vec {
                out.extend_from_slice(&(v as f32).to_le_bytes());
            }
        }

        out
    }

    pub fn deserialize(data: &[u8]) -> Self {
        let mut pos = 0;

        let vc = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let mut vocab = HashMap::with_capacity(vc);
        for i in 0..vc {
            let len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            let word = String::from_utf8(data[pos..pos + len].to_vec()).unwrap();
            pos += len;
            vocab.insert(word, i);
        }

        let vsize = vc;
        let dc = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let mut domain_vecs = vec![];
        for _ in 0..dc {
            let nlen = data[pos] as usize;
            pos += 1;
            let name = std::str::from_utf8(&data[pos..pos + nlen]).unwrap();
            pos += nlen;
            let domain = Domain::from_str(name).unwrap();
            let mut vec = vec![0.0f64; vsize];
            for v in vec.iter_mut() {
                *v = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as f64;
                pos += 4;
            }
            domain_vecs.push((domain, vec));
        }

        Router { vocab, domain_vecs }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.serialize())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        Ok(Self::deserialize(&data))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tokenize_words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_string())
        .collect()
}

fn l2_normalize(v: &mut Vec<f64>) {
    let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}
