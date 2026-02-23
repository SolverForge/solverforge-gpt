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
    // merge lookup: (a, b) -> (priority, new_id) for O(1) encode
    merge_rank: HashMap<(u32, u32), (usize, u32)>,
}

impl Tokenizer {
    pub fn train(corpus: &str, vocab_size: usize) -> Self {
        assert!(
            vocab_size > NUM_SPECIAL + 256,
            "vocab_size must be > {} (special + byte vocab)",
            NUM_SPECIAL + 256
        );

        let mut vocab: Vec<String> = Vec::with_capacity(vocab_size);
        vocab.push("<pad>".to_string());
        vocab.push("<bos>".to_string());
        vocab.push("<eos>".to_string());
        vocab.push("<sub>".to_string());
        vocab.push("</sub>".to_string());
        vocab.push("<task>".to_string());
        vocab.push("</task>".to_string());

        for b in 0u8..=255 {
            vocab.push(format!("b{}", b));
        }

        let mut words: Vec<Vec<u32>> = corpus
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|line| {
                let mut ids: Vec<u32> = line
                    .bytes()
                    .map(|b| NUM_SPECIAL as u32 + b as u32)
                    .collect();
                ids.push(NUM_SPECIAL as u32 + b' ' as u32); // space separator
                ids
            })
            .collect();

        let mut merges: Vec<(u32, u32, u32)> = vec![];
        let num_merges = vocab_size - vocab.len();

        for merge_idx in 0..num_merges {
            let mut pair_counts: HashMap<(u32, u32), u64> = HashMap::new();
            for word in &words {
                for pair in word.windows(2) {
                    *pair_counts.entry((pair[0], pair[1])).or_insert(0) += 1;
                }
            }

            if pair_counts.is_empty() {
                break;
            }

            let best = pair_counts
                .iter()
                .max_by_key(|&(k, &v)| (v, std::cmp::Reverse(*k)))
                .map(|(&k, _)| k)
                .unwrap();

            let new_id = vocab.len() as u32;
            let new_token = format!("{}{}", vocab[best.0 as usize], vocab[best.1 as usize]);
            vocab.push(new_token);
            merges.push((best.0, best.1, new_id));

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

        let merge_rank = merges
            .iter()
            .enumerate()
            .map(|(i, &(a, b, new_id))| ((a, b), (i, new_id)))
            .collect();

        Tokenizer {
            vocab,
            merges,
            merge_rank,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut ids: Vec<u32> = text
            .bytes()
            .map(|b| NUM_SPECIAL as u32 + b as u32)
            .collect();

        if ids.len() < 2 {
            return ids;
        }

        let merge_rank = &self.merge_rank;

        let n = ids.len();
        let mut prev: Vec<usize> = (0..n)
            .map(|i| if i == 0 { usize::MAX } else { i - 1 })
            .collect();
        let mut next: Vec<usize> = (0..n)
            .map(|i| if i == n - 1 { usize::MAX } else { i + 1 })
            .collect();

        let mut heap: BinaryHeap<(Reverse<usize>, u32, u32, usize)> = BinaryHeap::new();

        let push_pair = |heap: &mut BinaryHeap<(Reverse<usize>, u32, u32, usize)>,
                         merge_rank: &HashMap<(u32, u32), (usize, u32)>,
                         ids: &[u32],
                         left: usize,
                         right: usize| {
            if right == usize::MAX {
                return;
            }
            let pair = (ids[left], ids[right]);
            if let Some(&(rank, _)) = merge_rank.get(&pair) {
                heap.push((Reverse(rank), ids[left], ids[right], left));
            }
        };

        for i in 0..n - 1 {
            push_pair(&mut heap, &merge_rank, &ids, i, next[i]);
        }

        while let Some((Reverse(rank), expected_a, expected_b, left)) = heap.pop() {
            let right = next[left];
            if right == usize::MAX {
                continue;
            }

            if ids[left] != expected_a || ids[right] != expected_b {
                continue;
            }
            let pair = (ids[left], ids[right]);

            if let Some(&(current_rank, new_id)) = merge_rank.get(&pair) {
                if current_rank != rank {
                    continue; // stale
                }

                ids[left] = new_id;

                let right_next = next[right];
                next[left] = right_next;

                if right_next != usize::MAX {
                    prev[right_next] = left;
                }

                push_pair(&mut heap, &merge_rank, &ids, left, next[left]);

                if prev[left] != usize::MAX {
                    push_pair(&mut heap, &merge_rank, &ids, prev[left], left);
                }
            }
        }

        let mut result = Vec::with_capacity(ids.len());
        let mut cur = 0;
        loop {
            result.push(ids[cur]);
            if next[cur] == usize::MAX {
                break;
            }
            cur = next[cur];
        }
        result
    }

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

    pub fn encode_prompt(&self, task: &str) -> Vec<u32> {
        let mut ids = vec![BOS_ID, OTASK_ID];
        ids.extend(self.encode_raw(task));
        ids.push(CTASK_ID);
        // Append <sub> to kick off generation of first subtask
        ids.push(OSUB_ID);
        ids
    }

    pub fn decode_token(&self, id: u32) -> &str {
        &self.vocab[id as usize]
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = vec![];
        for &id in ids {
            let s = &self.vocab[id as usize];
            Self::decode_token_str(s, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    fn decode_token_str(s: &str, out: &mut Vec<u8>) {
        if s.starts_with('<') {
            return;
        }

        let mut chars = s.char_indices().peekable();
        let mut parsed_any = false;
        let mut all_valid = true;

        let mut tmp: Vec<u8> = vec![];

        while let Some((i, ch)) = chars.next() {
            if ch == 'b' {
                let start = i + 1;
                let mut end = start;
                while let Some(&(j, c)) = chars.peek() {
                    if c.is_ascii_digit() {
                        end = j + 1;
                        chars.next();
                    } else {
                        break;
                    }
                }
                if end > start {
                    let num_str = &s[start..end];
                    if let Ok(n) = num_str.parse::<u8>() {
                        tmp.push(n);
                        parsed_any = true;
                        continue;
                    }
                }
                all_valid = false;
                break;
            } else {
                all_valid = false;
                break;
            }
        }

        if parsed_any && all_valid {
            out.extend_from_slice(&tmp);
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = vec![];

        let vs = self.vocab.len() as u32;
        out.extend_from_slice(&vs.to_le_bytes());

        for s in &self.vocab {
            let b = s.as_bytes();
            out.extend_from_slice(&(b.len() as u16).to_le_bytes());
            out.extend_from_slice(b);
        }

        let mc = self.merges.len() as u32;
        out.extend_from_slice(&mc.to_le_bytes());
        for &(a, b, c) in &self.merges {
            out.extend_from_slice(&a.to_le_bytes());
            out.extend_from_slice(&b.to_le_bytes());
            out.extend_from_slice(&c.to_le_bytes());
        }

        out
    }

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

        let merge_rank = merges
            .iter()
            .enumerate()
            .map(|(i, &(a, b, new_id))| ((a, b), (i, new_id)))
            .collect();

        Tokenizer {
            vocab,
            merges,
            merge_rank,
        }
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

    flush(&current_domain, &current_task, &current_subs, &mut examples);

    examples
}
