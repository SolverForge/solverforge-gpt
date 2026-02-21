/*!
Byte-Pair Encoding (BPE) tokenizer.

Trained on a corpus in a single pass using the standard BPE algorithm:
1. Start with byte-level vocabulary (256 tokens)
2. Count all adjacent pair frequencies
3. Merge the most frequent pair into a new token
4. Repeat until target vocab size is reached

The trained vocabulary and merge rules can be serialized to/from a simple
binary format for embedding in the inference binary.

Special tokens:
  <pad> = 0
  <bos> = 1  (begin of sequence / task start)
  <eos> = 2  (end of sequence)
  <sub> = 3  (subtask marker open)
  </sub> = 4 (subtask marker close)
  <task> = 5
  </task> = 6

Zero external dependencies — pure Rust std only.
*/

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Special token ids (fixed, always at the top of vocab)
// ---------------------------------------------------------------------------

pub const PAD_ID: u32 = 0;
pub const BOS_ID: u32 = 1;
pub const EOS_ID: u32 = 2;
pub const OSUB_ID: u32 = 3; // <sub>
pub const CSUB_ID: u32 = 4; // </sub>
pub const OTASK_ID: u32 = 5; // <task>
pub const CTASK_ID: u32 = 6; // </task>
pub const NUM_SPECIAL: usize = 7;

// ---------------------------------------------------------------------------
// BPE Tokenizer
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Tokenizer {
    // vocab: id -> string representation (for decode)
    pub vocab: Vec<String>,
    // merge rules: (a, b) -> new_id, in priority order
    pub merges: Vec<(u32, u32, u32)>,
    // encode: string -> id
    str_to_id: HashMap<String, u32>,
}

impl Tokenizer {
    /// Build a new tokenizer from a corpus string, targeting `vocab_size` tokens.
    pub fn train(corpus: &str, vocab_size: usize) -> Self {
        assert!(
            vocab_size > NUM_SPECIAL + 256,
            "vocab_size must be > {} (special + byte vocab)",
            NUM_SPECIAL + 256
        );

        // Initialize vocabulary with special tokens + 256 byte tokens
        let mut vocab: Vec<String> = Vec::with_capacity(vocab_size);
        vocab.push("<pad>".to_string());
        vocab.push("<bos>".to_string());
        vocab.push("<eos>".to_string());
        vocab.push("<sub>".to_string());
        vocab.push("</sub>".to_string());
        vocab.push("<task>".to_string());
        vocab.push("</task>".to_string());

        // Byte-level base vocab
        for b in 0u8..=255 {
            vocab.push(format!("b{}", b));
        }

        let mut str_to_id: HashMap<String, u32> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();

        // Tokenize corpus at byte level (skip special token sequences)
        // We work with the actual text bytes, excluding lines that are metadata
        let mut words: Vec<Vec<u32>> = corpus
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|line| {
                // Encode line as byte ids (offset by NUM_SPECIAL)
                let mut ids: Vec<u32> = line
                    .bytes()
                    .map(|b| NUM_SPECIAL as u32 + b as u32)
                    .collect();
                // Add a space-separation token between words conceptually
                // by treating each line as a separate word sequence
                ids.push(NUM_SPECIAL as u32 + b' ' as u32); // space separator
                ids
            })
            .collect();

        let mut merges: Vec<(u32, u32, u32)> = vec![];
        let num_merges = vocab_size - vocab.len();

        for merge_idx in 0..num_merges {
            // Count all adjacent pairs
            let mut pair_counts: HashMap<(u32, u32), u64> = HashMap::new();
            for word in &words {
                for pair in word.windows(2) {
                    *pair_counts.entry((pair[0], pair[1])).or_insert(0) += 1;
                }
            }

            if pair_counts.is_empty() {
                break;
            }

            // Find most frequent pair (tie-break by pair value for determinism)
            let best = pair_counts
                .iter()
                .max_by_key(|&(k, &v)| (v, std::cmp::Reverse(*k)))
                .map(|(&k, _)| k)
                .unwrap();

            let new_id = vocab.len() as u32;
            let new_token = format!("{}{}", vocab[best.0 as usize], vocab[best.1 as usize]);
            str_to_id.insert(new_token.clone(), new_id);
            vocab.push(new_token);
            merges.push((best.0, best.1, new_id));

            // Apply merge to all words
            for word in words.iter_mut() {
                let mut i = 0;
                let mut new_word = Vec::with_capacity(word.len());
                while i < word.len() {
                    if i + 1 < word.len() && word[i] == best.0 && word[i + 1] == best.1 {
                        new_word.push(new_id);
                        i += 2;
                    } else {
                        new_word.push(word[i]);
                        i += 1;
                    }
                }
                *word = new_word;
            }

            if (merge_idx + 1) % 100 == 0 {
                eprintln!(
                    "  BPE merge {}/{}: vocab size = {}",
                    merge_idx + 1,
                    num_merges,
                    vocab.len()
                );
            }
        }

        Tokenizer {
            vocab,
            merges,
            str_to_id,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encode a raw text string into token ids (byte-level then apply merges)
    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        // Start with byte-level encoding
        let mut ids: Vec<u32> = text
            .bytes()
            .map(|b| NUM_SPECIAL as u32 + b as u32)
            .collect();

        // Apply merges in order
        for &(a, b, new_id) in &self.merges {
            let mut i = 0;
            let mut new_ids = Vec::with_capacity(ids.len());
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == a && ids[i + 1] == b {
                    new_ids.push(new_id);
                    i += 2;
                } else {
                    new_ids.push(ids[i]);
                    i += 1;
                }
            }
            ids = new_ids;
        }

        ids
    }

    /// Encode a task string into the full training/inference format:
    /// BOS <task> [task tokens] </task> <sub> [subtask1] </sub> ... EOS
    pub fn encode_training_example(&self, task: &str, subtasks: &[String]) -> Vec<u32> {
        let mut ids = vec![BOS_ID, OTASK_ID];
        ids.extend(self.encode_raw(task));
        ids.push(CTASK_ID);

        for sub in subtasks {
            ids.push(OSUB_ID);
            ids.extend(self.encode_raw(sub));
            ids.push(CSUB_ID);
        }

        ids.push(EOS_ID);
        ids
    }

    /// Encode just the prompt prefix for inference (task input side only)
    pub fn encode_prompt(&self, task: &str) -> Vec<u32> {
        let mut ids = vec![BOS_ID, OTASK_ID];
        ids.extend(self.encode_raw(task));
        ids.push(CTASK_ID);
        // Append <sub> to kick off generation of first subtask
        ids.push(OSUB_ID);
        ids
    }

    /// Decode a token id to its string representation
    pub fn decode_token(&self, id: u32) -> &str {
        &self.vocab[id as usize]
    }

    /// Decode a sequence of ids back to a string, stripping byte encoding
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = vec![];
        for &id in ids {
            let s = &self.vocab[id as usize];
            // Byte tokens are stored as "b{N}" where N is 0..255
            if let Some(rest) = s.strip_prefix('b') {
                if let Ok(n) = rest.parse::<u8>() {
                    bytes.push(n);
                    continue;
                }
            }
            // For merged tokens, decode by recursively expanding -- but since
            // merged tokens store the concatenated string representation, we
            // can just extract the bytes directly from the string content.
            // The simplest correct approach: re-encode the string chars as bytes.
            for b in s.bytes() {
                bytes.push(b);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Serialize to bytes for embedding/saving
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = vec![];

        // vocab size
        let vs = self.vocab.len() as u32;
        out.extend_from_slice(&vs.to_le_bytes());

        // vocab entries: u16 length + bytes
        for s in &self.vocab {
            let b = s.as_bytes();
            out.extend_from_slice(&(b.len() as u16).to_le_bytes());
            out.extend_from_slice(b);
        }

        // merges: count + (u32, u32, u32) triples
        let mc = self.merges.len() as u32;
        out.extend_from_slice(&mc.to_le_bytes());
        for &(a, b, c) in &self.merges {
            out.extend_from_slice(&a.to_le_bytes());
            out.extend_from_slice(&b.to_le_bytes());
            out.extend_from_slice(&c.to_le_bytes());
        }

        out
    }

    /// Deserialize from bytes
    pub fn deserialize(data: &[u8]) -> Self {
        let mut pos = 0;

        let vs = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let mut vocab = Vec::with_capacity(vs);
        for _ in 0..vs {
            let len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            let s = String::from_utf8(data[pos..pos + len].to_vec()).unwrap();
            pos += len;
            vocab.push(s);
        }

        let mc = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let mut merges = Vec::with_capacity(mc);
        for _ in 0..mc {
            let a = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let b = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let c = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            merges.push((a, b, c));
        }

        let str_to_id: HashMap<String, u32> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();

        Tokenizer {
            vocab,
            merges,
            str_to_id,
        }
    }

    /// Save to file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.serialize())
    }

    /// Load from file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        Ok(Self::deserialize(&data))
    }
}

// ---------------------------------------------------------------------------
// Training data parser
// ---------------------------------------------------------------------------

/// A single training example
#[derive(Clone, Debug)]
pub struct Example {
    pub task: String,
    pub subtasks: Vec<String>,
    pub domain: String,
}

/// Parse training data file. Each example is:
///   DOMAIN: <domain name>
///   TASK: <task description>
///   SUB: <subtask 1>
///   SUB: <subtask 2>
///   ...
///   (blank line separates examples)
pub fn parse_training_data(text: &str) -> Vec<Example> {
    let mut examples = vec![];
    let mut current_domain = String::new();
    let mut current_task: Option<String> = None;
    let mut current_subs: Vec<String> = vec![];

    let flush =
        |domain: &str, task: &Option<String>, subs: &Vec<String>, out: &mut Vec<Example>| {
            if let Some(t) = task {
                if !subs.is_empty() {
                    out.push(Example {
                        task: t.clone(),
                        subtasks: subs.clone(),
                        domain: domain.to_string(),
                    });
                }
            }
        };

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            flush(&current_domain, &current_task, &current_subs, &mut examples);
            current_task = None;
            current_subs.clear();
        } else if let Some(rest) = line.strip_prefix("DOMAIN:") {
            flush(&current_domain, &current_task, &current_subs, &mut examples);
            current_task = None;
            current_subs.clear();
            current_domain = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("TASK:") {
            current_task = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("SUB:") {
            current_subs.push(rest.trim().to_string());
        }
    }

    // flush last example
    flush(&current_domain, &current_task, &current_subs, &mut examples);

    examples
}
