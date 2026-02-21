/*!
GPT-style transformer model using the tensor autograd engine.

Architecture:
  - Decoder-only transformer
  - Token + positional embeddings
  - N layers of: RMSNorm -> multi-head attention (with KV cache) -> residual
                 RMSNorm -> MLP (linear -> ReLU -> linear) -> residual
  - Final linear projection to vocab logits

Hyperparameters for task-decomposition models:
  n_layer = 4, n_embd = 128, n_head = 4, block_size = 256
  ~1.5M parameters per domain model.

Zero external dependencies — pure Rust std only.
*/

use crate::rng::Rng;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Config {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub block_size: usize,
}

impl Config {
    pub fn task_decomposer(vocab_size: usize) -> Self {
        Config {
            vocab_size,
            n_embd: 128,
            n_head: 4,
            n_layer: 4,
            block_size: 256,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

// ---------------------------------------------------------------------------
// Model parameters
// ---------------------------------------------------------------------------

fn make_param(rows: usize, cols: usize, std: f64, rng: &mut Rng) -> Tensor {
    let data: Vec<f64> = (0..rows * cols).map(|_| rng.gauss(0.0, std)).collect();
    Tensor::new(data, [rows, cols])
}

struct LayerWeights {
    attn_wq: Tensor,
    attn_wk: Tensor,
    attn_wv: Tensor,
    attn_wo: Tensor,
    mlp_fc1: Tensor,
    mlp_fc2: Tensor,
}

pub struct Model {
    pub cfg: Config,
    wte: Tensor,
    wpe: Tensor,
    lm_head: Tensor,
    layers: Vec<LayerWeights>,
}

impl Model {
    pub fn new(cfg: Config, rng: &mut Rng) -> Self {
        let std = (2.0 / (cfg.n_embd as f64)).sqrt();
        let e = cfg.n_embd;
        let vs = cfg.vocab_size;
        let bs = cfg.block_size;

        let wte = make_param(vs, e, std, rng);
        let wpe = make_param(bs, e, std, rng);
        let lm_head = make_param(vs, e, std, rng);

        let layers = (0..cfg.n_layer)
            .map(|_| LayerWeights {
                attn_wq: make_param(e, e, std, rng),
                attn_wk: make_param(e, e, std, rng),
                attn_wv: make_param(e, e, std, rng),
                attn_wo: make_param(e, e, std, rng),
                mlp_fc1: make_param(4 * e, e, std, rng),
                mlp_fc2: make_param(e, 4 * e, std, rng),
            })
            .collect();

        Model {
            cfg,
            wte,
            wpe,
            lm_head,
            layers,
        }
    }

    pub fn params(&self) -> Vec<Tensor> {
        let mut ps = vec![self.wte.clone(), self.wpe.clone(), self.lm_head.clone()];
        for l in &self.layers {
            ps.push(l.attn_wq.clone());
            ps.push(l.attn_wk.clone());
            ps.push(l.attn_wv.clone());
            ps.push(l.attn_wo.clone());
            ps.push(l.mlp_fc1.clone());
            ps.push(l.mlp_fc2.clone());
        }
        ps
    }

    pub fn num_params(&self) -> usize {
        self.params().iter().map(|p| p.len()).sum()
    }

    pub fn zero_grad(&self) {
        for p in self.params() {
            p.zero_grad();
        }
    }

    // ---- training forward pass ---------------------------------------------

    /// Full causal forward pass over a token sequence.
    /// Returns the scalar cross-entropy loss averaged over labeled positions.
    /// label_mask[i] = true means compute loss predicting tokens[i+1] from position i.
    pub fn forward_train(&self, tokens: &[u32], label_mask: &[bool]) -> Tensor {
        let t = tokens.len();
        assert!(t >= 2);
        assert_eq!(label_mask.len(), t - 1);

        let cfg = &self.cfg;
        let e = cfg.n_embd;

        // Embed each position: [1, e]
        let mut x: Vec<Tensor> = (0..t)
            .map(|pos| {
                let tok = self.wte.row(tokens[pos] as usize);
                let pe = self.wpe.row(pos.min(cfg.block_size - 1));
                tok.tensor_add(&pe)
            })
            .collect();

        // Transformer layers
        for li in 0..cfg.n_layer {
            let lw = &self.layers[li];

            // Norm + Q/K/V projections for all positions
            let normed: Vec<Tensor> = x.iter().map(|r| r.rmsnorm()).collect();
            // W is [n_embd, n_embd], x is [1, n_embd], output [1, n_embd]
            let qs: Vec<Tensor> = normed.iter().map(|r| r.matmul_t(&lw.attn_wq)).collect();
            let ks: Vec<Tensor> = normed.iter().map(|r| r.matmul_t(&lw.attn_wk)).collect();
            let vs: Vec<Tensor> = normed.iter().map(|r| r.matmul_t(&lw.attn_wv)).collect();

            let head_dim = cfg.head_dim();
            let scale = (head_dim as f64).sqrt();

            // Causal self-attention for each position
            let mut attn_outs: Vec<Tensor> = Vec::with_capacity(t);
            for pos in 0..t {
                let seq = pos + 1;
                let mut heads: Vec<Tensor> = Vec::with_capacity(cfg.n_head);

                for h in 0..cfg.n_head {
                    let hs = h * head_dim;
                    let he = hs + head_dim;

                    let q_h = qs[pos].slice_cols(hs, he); // [1, head_dim]

                    // Attention scores: dot q with each past k
                    let mut score_data = vec![0.0f64; seq];
                    let q_data = q_h.data();
                    for t2 in 0..seq {
                        let k_data = ks[t2].data();
                        let mut s = 0.0;
                        for j in 0..head_dim {
                            s += q_data[j] * k_data[hs + j];
                        }
                        score_data[t2] = s / scale;
                    }
                    let scores = Tensor::new(score_data, [1, seq]).softmax();
                    let sw = scores.data();

                    // Weighted sum of values
                    let mut val_data = vec![0.0f64; head_dim];
                    for t2 in 0..seq {
                        let v_data = vs[t2].data();
                        for j in 0..head_dim {
                            val_data[j] += sw[t2] * v_data[hs + j];
                        }
                    }
                    heads.push(Tensor::new(val_data, [1, head_dim]));
                }

                // Concatenate heads [1, e]
                let concat_data: Vec<f64> = heads.iter().flat_map(|h| h.data()).collect();
                attn_outs.push(Tensor::new(concat_data, [1, e]));
            }

            // Output projection + residual
            let mut new_x: Vec<Tensor> = Vec::with_capacity(t);
            for pos in 0..t {
                let proj = attn_outs[pos].matmul_t(&lw.attn_wo);
                let after_attn = proj.tensor_add(&x[pos]);

                // MLP
                let normed2 = after_attn.rmsnorm();
                let fc1 = normed2.matmul_t(&lw.mlp_fc1);
                let act = fc1.relu();
                let fc2 = act.matmul_t(&lw.mlp_fc2);
                let after_mlp = fc2.tensor_add(&after_attn);
                new_x.push(after_mlp);
            }
            x = new_x;
        }

        // Compute cross-entropy loss over labeled positions
        let mut losses: Vec<Tensor> = vec![];
        for pos in 0..t - 1 {
            if !label_mask[pos] {
                continue;
            }
            let target = tokens[pos + 1] as usize;
            let logits = x[pos].matmul_t(&self.lm_head); // [1, vocab_size]
            let loss = logits.cross_entropy(target);
            losses.push(loss);
        }

        if losses.is_empty() {
            return Tensor::scalar(0.0);
        }

        let n = losses.len() as f64;
        let total = losses
            .iter()
            .skip(1)
            .fold(losses[0].clone(), |a, b| a.tensor_add(&b));
        total.mul_scalar(1.0 / n)
    }

    // ---- inference (autoregressive, no autograd) ----------------------------

    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f64,
        rng: &mut Rng,
    ) -> Vec<u32> {
        let cfg = &self.cfg;
        let e = cfg.n_embd;
        let hd = cfg.head_dim();

        // KV cache: [layer][pos] -> Vec<f64> of length e
        let mut k_cache: Vec<Vec<Vec<f64>>> = vec![vec![]; cfg.n_layer];
        let mut v_cache: Vec<Vec<Vec<f64>>> = vec![vec![]; cfg.n_layer];

        let mut generated: Vec<u32> = vec![];

        // Process prompt
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            self.forward_infer(tok, pos, &mut k_cache, &mut v_cache, e, hd, cfg);
        }

        // Generate
        for _ in 0..max_new_tokens {
            let pos = prompt_tokens.len() + generated.len() - 1;
            let last = if generated.is_empty() {
                *prompt_tokens.last().unwrap()
            } else {
                *generated.last().unwrap()
            };

            let logits = self.forward_infer(last, pos, &mut k_cache, &mut v_cache, e, hd, cfg);
            let next = sample_logits(&logits, temperature, rng);
            generated.push(next);

            if next == crate::tokenizer::EOS_ID {
                break;
            }
        }

        generated
    }

    fn forward_infer(
        &self,
        token: u32,
        pos: usize,
        k_cache: &mut Vec<Vec<Vec<f64>>>,
        v_cache: &mut Vec<Vec<Vec<f64>>>,
        e: usize,
        head_dim: usize,
        cfg: &Config,
    ) -> Vec<f64> {
        let tok_emb = self.wte.row_data(token as usize);
        let pos_emb = self.wpe.row_data(pos.min(cfg.block_size - 1));
        let mut x: Vec<f64> = tok_emb.iter().zip(&pos_emb).map(|(a, b)| a + b).collect();

        for li in 0..cfg.n_layer {
            let lw = &self.layers[li];
            let xn = rmsnorm_vec(&x);

            let q = matvec(&lw.attn_wq.data(), e, e, &xn);
            let k = matvec(&lw.attn_wk.data(), e, e, &xn);
            let v = matvec(&lw.attn_wv.data(), e, e, &xn);

            k_cache[li].push(k);
            v_cache[li].push(v);

            let seq = k_cache[li].len();
            let scale = (head_dim as f64).sqrt();
            let mut attn_out = vec![0.0f64; e];

            for h in 0..cfg.n_head {
                let hs = h * head_dim;
                let q_h = &q[hs..hs + head_dim];

                let mut scores: Vec<f64> = (0..seq)
                    .map(|t2| {
                        let k_h = &k_cache[li][t2][hs..hs + head_dim];
                        dot(q_h, k_h) / scale
                    })
                    .collect();
                softmax_inplace(&mut scores);

                for j in 0..head_dim {
                    let val: f64 = (0..seq)
                        .map(|t2| scores[t2] * v_cache[li][t2][hs + j])
                        .sum();
                    attn_out[hs + j] = val;
                }
            }

            let proj = matvec(&lw.attn_wo.data(), e, e, &attn_out);
            x = x.iter().zip(&proj).map(|(a, b)| a + b).collect();

            let xn2 = rmsnorm_vec(&x);
            let fc1 = matvec(&lw.mlp_fc1.data(), 4 * e, e, &xn2);
            let act: Vec<f64> = fc1.iter().map(|&v| v.max(0.0)).collect();
            let fc2 = matvec(&lw.mlp_fc2.data(), e, 4 * e, &act);
            x = x.iter().zip(&fc2).map(|(a, b)| a + b).collect();
        }

        matvec(&self.lm_head.data(), cfg.vocab_size, e, &x)
    }

    // ---- serialization ------------------------------------------------------

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut bytes = vec![];
        for v in &[
            self.cfg.vocab_size,
            self.cfg.n_embd,
            self.cfg.n_head,
            self.cfg.n_layer,
            self.cfg.block_size,
        ] {
            bytes.extend_from_slice(&(*v as u32).to_le_bytes());
        }
        for p in self.params() {
            for &v in &p.data() {
                bytes.extend_from_slice(&(v as f32).to_le_bytes());
            }
        }
        std::fs::write(path, bytes)
    }

    pub fn load(path: &str, rng: &mut Rng) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let mut pos = 0;

        macro_rules! ru32 {
            () => {{
                let v = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                v
            }};
        }

        let cfg = Config {
            vocab_size: ru32!(),
            n_embd: ru32!(),
            n_head: ru32!(),
            n_layer: ru32!(),
            block_size: ru32!(),
        };
        let model = Model::new(cfg, rng);

        for p in model.params() {
            let n = p.len();
            let mut data = vec![0.0f64; n];
            for i in 0..n {
                data[i] = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as f64;
                pos += 4;
            }
            p.set_data_vec(data);
        }

        Ok(model)
    }
}

// ---------------------------------------------------------------------------
// Tensor helpers (in tensor.rs impl, not duplicating add)
// ---------------------------------------------------------------------------

impl Tensor {
    /// Multiply self [1, K] by W^T where W is [N, K] -> result [1, N]
    /// Equivalent to x @ W.T
    pub fn matmul_t(&self, w: &Tensor) -> Tensor {
        // w is [rows=N, cols=K], self is [1, K]
        // result[i] = dot(self, w[i])
        let k = self.cols();
        let n = w.rows();
        assert_eq!(w.cols(), k);
        let x = self.data();
        let wdata = w.data();
        let mut out_data = vec![0.0f64; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..k {
                s += x[j] * wdata[i * k + j];
            }
            out_data[i] = s;
        }
        // Forward only (no autograd for now in attention score path)
        // For training path we need grad. Use matmul with explicit transpose.
        // This is used only in the training forward pass where we need grad.
        // We route through the existing matmul + transpose ops.
        let out = Tensor::new(out_data, [1, n]);

        // backward: d_self = out_grad @ W, d_W += self^T @ out_grad
        let a = self.clone();
        let b = w.clone();
        let out_c = out.clone();
        out.set_backward(move || {
            let og = out_c.grad(); // [n]
            let xdata = a.data(); // [k]
            let wdata = b.data(); // [n, k]

            // d_a: sum_i og[i] * w[i, j]
            let mut da = vec![0.0f64; k];
            for i in 0..n {
                for j in 0..k {
                    da[j] += og[i] * wdata[i * k + j];
                }
            }
            a.add_grad_vec(&da);

            // d_W: og[i] * x[j]
            let mut dw = vec![0.0f64; n * k];
            for i in 0..n {
                for j in 0..k {
                    dw[i * k + j] += og[i] * xdata[j];
                }
            }
            b.add_grad_vec(&dw);
        });

        out
    }

    /// Element-wise add (non-consuming, for use in training loops)
    pub fn tensor_add(&self, other: &Tensor) -> Tensor {
        self.add(other)
    }

    /// Get a single row as raw f64 Vec (no grad, for inference)
    pub fn row_data(&self, idx: usize) -> Vec<f64> {
        let n = self.cols();
        let data = self.data();
        data[idx * n..(idx + 1) * n].to_vec()
    }

    /// Slice columns [start, end) -> [rows, end-start]
    pub fn slice_cols(&self, start: usize, end: usize) -> Tensor {
        let (m, n) = (self.rows(), self.cols());
        let nc = end - start;
        let data = self.data();
        let mut out_data = vec![0.0f64; m * nc];
        for i in 0..m {
            for j in 0..nc {
                out_data[i * nc + j] = data[i * n + start + j];
            }
        }
        let out = Tensor::new(out_data, [m, nc]);
        let a = self.clone();
        let out_c = out.clone();
        out.set_backward(move || {
            let og = out_c.grad();
            let mut da = vec![0.0f64; m * n];
            for i in 0..m {
                for j in 0..nc {
                    da[i * n + start + j] += og[i * nc + j];
                }
            }
            a.add_grad_vec(&da);
        });
        out
    }
}

// ---------------------------------------------------------------------------
// Pure f64 inference helpers (no autograd)
// ---------------------------------------------------------------------------

fn rmsnorm_vec(x: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let ms = x.iter().map(|&v| v * v).sum::<f64>() / n;
    let scale = (ms + 1e-5).sqrt();
    x.iter().map(|&v| v / scale).collect()
}

fn matvec(w: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    (0..rows)
        .map(|i| (0..cols).map(|j| w[i * cols + j] * x[j]).sum::<f64>())
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn softmax_inplace(x: &mut Vec<f64>) {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    x.iter_mut().for_each(|v| *v = (*v - max).exp());
    let sum: f64 = x.iter().sum();
    x.iter_mut().for_each(|v| *v /= sum);
}

fn sample_logits(logits: &[f64], temperature: f64, rng: &mut Rng) -> u32 {
    let mut probs = logits.to_vec();
    if temperature > 0.0 {
        probs.iter_mut().for_each(|v| *v /= temperature);
    }
    softmax_inplace(&mut probs);
    rng.choices(&probs) as u32
}
