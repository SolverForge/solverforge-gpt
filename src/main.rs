/*!
The most atomic way to train and run inference for a GPT in pure, dependency-free Rust.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy (original Python), ported to Rust
*/

use std::cell::RefCell;
use std::collections::HashSet;
use std::fs;
use std::io::{self, Write};
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Autograd: scalar Value node with reverse-mode autodiff
// ---------------------------------------------------------------------------

// Each node in the computation graph. Children and their local gradients are
// stored alongside the node so backward() can traverse the DAG in reverse
// topological order and accumulate gradients into the leaves (parameters).
struct ValueInner {
    data: f64,
    grad: f64,
    // (child, local_grad) pairs. Cloning the Rc keeps the graph alive.
    children: Vec<(Value, f64)>,
}

#[derive(Clone)]
struct Value(Rc<RefCell<ValueInner>>);

impl Value {
    fn new(data: f64) -> Self {
        Value(Rc::new(RefCell::new(ValueInner {
            data,
            grad: 0.0,
            children: vec![],
        })))
    }

    fn with_children(data: f64, children: Vec<(Value, f64)>) -> Self {
        Value(Rc::new(RefCell::new(ValueInner {
            data,
            grad: 0.0,
            children,
        })))
    }

    fn data(&self) -> f64 {
        self.0.borrow().data
    }
    fn grad(&self) -> f64 {
        self.0.borrow().grad
    }
    fn set_data(&self, d: f64) {
        self.0.borrow_mut().data = d;
    }
    fn set_grad(&self, g: f64) {
        self.0.borrow_mut().grad = g;
    }
    fn add_grad(&self, g: f64) {
        self.0.borrow_mut().grad += g;
    }
    fn zero_grad(&self) {
        self.0.borrow_mut().grad = 0.0;
    }

    // ---- ops ---------------------------------------------------------------

    fn add(&self, other: &Value) -> Value {
        Value::with_children(
            self.data() + other.data(),
            vec![(self.clone(), 1.0), (other.clone(), 1.0)],
        )
    }

    fn mul(&self, other: &Value) -> Value {
        let (sd, od) = (self.data(), other.data());
        Value::with_children(sd * od, vec![(self.clone(), od), (other.clone(), sd)])
    }

    fn pow_f64(&self, exp: f64) -> Value {
        let d = self.data();
        Value::with_children(d.powf(exp), vec![(self.clone(), exp * d.powf(exp - 1.0))])
    }

    fn log(&self) -> Value {
        let d = self.data();
        Value::with_children(d.ln(), vec![(self.clone(), 1.0 / d)])
    }

    fn exp(&self) -> Value {
        let d = self.data();
        let e = d.exp();
        Value::with_children(e, vec![(self.clone(), e)])
    }

    fn relu(&self) -> Value {
        let d = self.data();
        Value::with_children(
            d.max(0.0),
            vec![(self.clone(), if d > 0.0 { 1.0 } else { 0.0 })],
        )
    }

    fn neg(&self) -> Value {
        self.mul_f64(-1.0)
    }
    fn sub(&self, o: &Value) -> Value {
        self.add(&o.neg())
    }
    fn div(&self, o: &Value) -> Value {
        self.mul(&o.pow_f64(-1.0))
    }

    fn mul_f64(&self, s: f64) -> Value {
        Value::with_children(self.data() * s, vec![(self.clone(), s)])
    }

    fn add_f64(&self, s: f64) -> Value {
        self.add(&Value::new(s))
    }

    // ---- backward ----------------------------------------------------------

    fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<*const RefCell<ValueInner>> = HashSet::new();

        fn build(
            v: &Value,
            topo: &mut Vec<Value>,
            visited: &mut HashSet<*const RefCell<ValueInner>>,
        ) {
            let ptr = Rc::as_ptr(&v.0);
            if visited.insert(ptr) {
                // Collect children first to avoid holding borrow across recursion
                let children: Vec<Value> =
                    v.0.borrow()
                        .children
                        .iter()
                        .map(|(c, _)| c.clone())
                        .collect();
                for child in &children {
                    build(child, topo, visited);
                }
                topo.push(v.clone());
            }
        }

        build(self, &mut topo, &mut visited);

        self.set_grad(1.0);
        for v in topo.iter().rev() {
            let vg = v.grad();
            let children: Vec<(Value, f64)> = v.0.borrow().children.clone();
            for (child, local_grad) in &children {
                child.add_grad(local_grad * vg);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Random number generation
// ---------------------------------------------------------------------------
// Python uses Mersenne Twister; we use xorshift64 + Box-Muller for the same
// statistical properties (Gaussian weights, uniform sampling, shuffle).
// Bit-exact reproduction of Python's stream is not required.

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng {
            state: seed ^ 0x123456789abcdef,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    // Uniform [0, 1)
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    // Standard normal via Box-Muller
    fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mu + sigma * z
    }

    // Fisher-Yates shuffle
    fn shuffle<T>(&mut self, v: &mut Vec<T>) {
        let n = v.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            v.swap(i, j);
        }
    }

    // Weighted categorical sample; returns index
    fn choices(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut r = self.uniform() * total;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }
}

// ---------------------------------------------------------------------------
// Dataset & Tokenizer
// ---------------------------------------------------------------------------

fn load_docs(path: &str) -> Vec<String> {
    fs::read_to_string(path)
        .expect("could not read input.txt")
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

// Sorted unique characters across all docs  →  token ids 0..n-1
fn build_vocab(docs: &[String]) -> Vec<char> {
    let mut set: std::collections::BTreeSet<char> = std::collections::BTreeSet::new();
    for d in docs {
        for c in d.chars() {
            set.insert(c);
        }
    }
    set.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Model parameters
// ---------------------------------------------------------------------------

type Matrix = Vec<Vec<Value>>;

fn make_matrix(nout: usize, nin: usize, std: f64, rng: &mut Rng) -> Matrix {
    (0..nout)
        .map(|_| (0..nin).map(|_| Value::new(rng.gauss(0.0, std))).collect())
        .collect()
}

struct LayerWeights {
    attn_wq: Matrix,
    attn_wk: Matrix,
    attn_wv: Matrix,
    attn_wo: Matrix,
    mlp_fc1: Matrix,
    mlp_fc2: Matrix,
}

struct StateDict {
    wte: Matrix,
    wpe: Matrix,
    lm_head: Matrix,
    layers: Vec<LayerWeights>,
}

impl StateDict {
    fn new(
        vocab_size: usize,
        n_embd: usize,
        block_size: usize,
        n_layer: usize,
        rng: &mut Rng,
    ) -> Self {
        let std = 0.08;
        // Mirror Python init order: wte, wpe, lm_head, then layer weights
        let wte = make_matrix(vocab_size, n_embd, std, rng);
        let wpe = make_matrix(block_size, n_embd, std, rng);
        let lm_head = make_matrix(vocab_size, n_embd, std, rng);
        let layers = (0..n_layer)
            .map(|_| LayerWeights {
                attn_wq: make_matrix(n_embd, n_embd, std, rng),
                attn_wk: make_matrix(n_embd, n_embd, std, rng),
                attn_wv: make_matrix(n_embd, n_embd, std, rng),
                attn_wo: make_matrix(n_embd, n_embd, std, rng),
                mlp_fc1: make_matrix(4 * n_embd, n_embd, std, rng),
                mlp_fc2: make_matrix(n_embd, 4 * n_embd, std, rng),
            })
            .collect();
        StateDict {
            wte,
            wpe,
            lm_head,
            layers,
        }
    }

    // Flat list of all parameter Values (same order as Python's state_dict iteration)
    fn params(&self) -> Vec<Value> {
        let mut ps = vec![];
        let mut add = |m: &Matrix| {
            for row in m {
                for p in row {
                    ps.push(p.clone());
                }
            }
        };
        add(&self.wte);
        add(&self.wpe);
        add(&self.lm_head);
        for l in &self.layers {
            add(&l.attn_wq);
            add(&l.attn_wk);
            add(&l.attn_wv);
            add(&l.attn_wo);
            add(&l.mlp_fc1);
            add(&l.mlp_fc2);
        }
        ps
    }
}

// ---------------------------------------------------------------------------
// Neural network primitives
// ---------------------------------------------------------------------------

fn linear(x: &[Value], w: &Matrix) -> Vec<Value> {
    w.iter()
        .map(|row| {
            row.iter()
                .zip(x.iter())
                .map(|(wi, xi)| wi.mul(xi))
                .reduce(|a, b| a.add(&b))
                .unwrap()
        })
        .collect()
}

fn softmax(logits: &[Value]) -> Vec<Value> {
    let max_val = logits
        .iter()
        .map(|v| v.data())
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<Value> = logits
        .iter()
        .map(|v| v.sub(&Value::new(max_val)).exp())
        .collect();
    // Sum all exps
    let total = exps.iter().skip(1).fold(exps[0].clone(), |a, b| a.add(b));
    exps.iter().map(|e| e.div(&total)).collect()
}

fn rmsnorm(x: &[Value]) -> Vec<Value> {
    let n = x.len() as f64;
    let ms = x
        .iter()
        .map(|xi| xi.mul(xi))
        .reduce(|a, b| a.add(&b))
        .unwrap()
        .mul_f64(1.0 / n);
    let scale = ms.add_f64(1e-5).pow_f64(-0.5);
    x.iter().map(|xi| xi.mul(&scale)).collect()
}

// ---------------------------------------------------------------------------
// GPT forward pass — one token at a time with explicit KV cache
// ---------------------------------------------------------------------------

fn gpt(
    token_id: usize,
    pos_id: usize,
    keys: &mut Vec<Vec<Vec<Value>>>, // [layer][pos][dim]
    vals: &mut Vec<Vec<Vec<Value>>>, // [layer][pos][dim]
    sd: &StateDict,
    n_head: usize,
    head_dim: usize,
) -> Vec<Value> {
    // Token + position embeddings
    let tok_emb = &sd.wte[token_id];
    let pos_emb = &sd.wpe[pos_id];
    let mut x: Vec<Value> = tok_emb.iter().zip(pos_emb).map(|(t, p)| t.add(p)).collect();
    x = rmsnorm(&x);

    for li in 0..sd.layers.len() {
        // --- Multi-head self-attention ---
        let x_res = x.clone();
        x = rmsnorm(&x);
        let q = linear(&x, &sd.layers[li].attn_wq);
        let k = linear(&x, &sd.layers[li].attn_wk);
        let v = linear(&x, &sd.layers[li].attn_wv);
        keys[li].push(k);
        vals[li].push(v);

        let seq_len = keys[li].len();
        let scale = (head_dim as f64).sqrt();
        let mut x_attn: Vec<Value> = Vec::with_capacity(n_head * head_dim);

        for h in 0..n_head {
            let hs = h * head_dim;
            let q_h = &q[hs..hs + head_dim];

            let attn_logits: Vec<Value> = (0..seq_len)
                .map(|t| {
                    let dot = q_h
                        .iter()
                        .zip(&keys[li][t][hs..hs + head_dim])
                        .map(|(qi, ki)| qi.mul(ki))
                        .reduce(|a, b| a.add(&b))
                        .unwrap();
                    dot.mul_f64(1.0 / scale)
                })
                .collect();

            let attn_w = softmax(&attn_logits);

            for j in 0..head_dim {
                let out = (0..seq_len)
                    .map(|t| attn_w[t].mul(&vals[li][t][hs + j]))
                    .reduce(|a, b| a.add(&b))
                    .unwrap();
                x_attn.push(out);
            }
        }

        x = linear(&x_attn, &sd.layers[li].attn_wo);
        x = x.iter().zip(&x_res).map(|(a, b)| a.add(b)).collect();

        // --- MLP ---
        let x_res = x.clone();
        x = rmsnorm(&x);
        x = linear(&x, &sd.layers[li].mlp_fc1);
        x = x.iter().map(|xi| xi.relu()).collect();
        x = linear(&x, &sd.layers[li].mlp_fc2);
        x = x.iter().zip(&x_res).map(|(a, b)| a.add(b)).collect();
    }

    linear(&x, &sd.lm_head)
}

// ---------------------------------------------------------------------------
// Main: dataset → train → inference
// ---------------------------------------------------------------------------

fn main() {
    let input_path = "input.txt";
    if !std::path::Path::new(input_path).exists() {
        eprintln!("input.txt not found. Download it with:");
        eprintln!("  curl -o input.txt https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt");
        std::process::exit(1);
    }

    let mut rng = Rng::new(42);

    // Dataset
    let mut docs = load_docs(input_path);
    let uchars = build_vocab(&docs);
    let bos = uchars.len(); // BOS token id  (= vocab_size - 1)
    let vocab_size = uchars.len() + 1;

    rng.shuffle(&mut docs);
    println!("num docs: {}", docs.len());
    println!("vocab size: {}", vocab_size);

    // Hyperparameters (mirror Python)
    let n_layer: usize = 1;
    let n_embd: usize = 16;
    let block_size: usize = 16;
    let n_head: usize = 4;
    let head_dim = n_embd / n_head;

    // Parameters
    let sd = StateDict::new(vocab_size, n_embd, block_size, n_layer, &mut rng);
    let params = sd.params();
    println!("num params: {}", params.len());

    // Adam buffers
    let lr = 0.01_f64;
    let beta1 = 0.85_f64;
    let beta2 = 0.99_f64;
    let eps = 1e-8_f64;
    let mut m_buf = vec![0.0_f64; params.len()];
    let mut v_buf = vec![0.0_f64; params.len()];

    // Training loop
    let num_steps = 1000_usize;
    for step in 0..num_steps {
        let doc = &docs[step % docs.len()];

        // Tokenise: BOS + char-ids + BOS
        let mut tokens: Vec<usize> = vec![bos];
        for ch in doc.chars() {
            tokens.push(uchars.iter().position(|&c| c == ch).unwrap());
        }
        tokens.push(bos);

        let n = block_size.min(tokens.len() - 1);

        // Forward pass
        let mut keys: Vec<Vec<Vec<Value>>> = vec![vec![]; n_layer];
        let mut vals: Vec<Vec<Vec<Value>>> = vec![vec![]; n_layer];
        let mut losses: Vec<Value> = Vec::with_capacity(n);

        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];
            let logits = gpt(
                token_id, pos_id, &mut keys, &mut vals, &sd, n_head, head_dim,
            );
            let probs = softmax(&logits);
            losses.push(probs[target_id].log().neg());
        }

        let loss = losses
            .iter()
            .skip(1)
            .fold(losses[0].clone(), |a, b| a.add(b))
            .mul_f64(1.0 / n as f64);

        // Backward
        loss.backward();

        // Adam update with linear LR decay
        let lr_t = lr * (1.0 - step as f64 / num_steps as f64);
        let step1 = (step + 1) as f64;
        for (i, p) in params.iter().enumerate() {
            let g = p.grad();
            m_buf[i] = beta1 * m_buf[i] + (1.0 - beta1) * g;
            v_buf[i] = beta2 * v_buf[i] + (1.0 - beta2) * g * g;
            let m_hat = m_buf[i] / (1.0 - beta1.powf(step1));
            let v_hat = v_buf[i] / (1.0 - beta2.powf(step1));
            p.set_data(p.data() - lr_t * m_hat / (v_hat.sqrt() + eps));
            p.zero_grad();
        }

        print!(
            "\rstep {:4} / {:4} | loss {:.4}",
            step + 1,
            num_steps,
            loss.data()
        );
        io::stdout().flush().unwrap();
    }

    println!();

    // Inference
    let temperature = 0.5_f64;
    println!("--- inference (new, hallucinated names) ---");
    for sample_idx in 0..20 {
        let mut keys: Vec<Vec<Vec<Value>>> = vec![vec![]; n_layer];
        let mut vals: Vec<Vec<Vec<Value>>> = vec![vec![]; n_layer];
        let mut token_id = bos;
        let mut sample: Vec<char> = vec![];

        for pos_id in 0..block_size {
            let logits = gpt(
                token_id, pos_id, &mut keys, &mut vals, &sd, n_head, head_dim,
            );
            let scaled: Vec<Value> = logits
                .iter()
                .map(|l| l.mul_f64(1.0 / temperature))
                .collect();
            let probs = softmax(&scaled);
            let weights: Vec<f64> = probs.iter().map(|p| p.data()).collect();
            token_id = rng.choices(&weights);
            if token_id == bos {
                break;
            }
            sample.push(uchars[token_id]);
        }

        println!(
            "sample {:2}: {}",
            sample_idx + 1,
            sample.iter().collect::<String>()
        );
    }
}
