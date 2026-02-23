/*!
Tensor autograd engine.

Replaces the scalar Value engine with matrix-level operations so that
training is tractable. Every Tensor owns its data and gradient as flat
Vec<f64> with a 2-D shape [rows, cols].

All ops that participate in the computation graph return a new Tensor
and record the backward closure needed to propagate gradients. Backward
is triggered by calling `.backward()` on the loss scalar (a 1×1 Tensor).

Zero external dependencies — pure Rust std only.
*/

use std::cell::RefCell;
use std::rc::Rc;

// Global tape: records every Tensor created, in creation order.
// backward() drains this to walk the graph in reverse and break Rc cycles.
thread_local! {
    static TAPE: RefCell<Vec<Tensor>> = RefCell::new(Vec::new());
}

// ---------------------------------------------------------------------------
// Core Tensor type
// ---------------------------------------------------------------------------

struct TensorInner {
    data: Vec<f64>,
    grad: Vec<f64>,
    shape: [usize; 2], // [rows, cols]
    // backward closure: accumulate gradients into inputs
    backward: Option<Box<dyn Fn()>>,
}

#[derive(Clone)]
pub struct Tensor(Rc<RefCell<TensorInner>>);

impl Tensor {
    // ---- CONSTRUCTORS -------------------------------------------------------

    pub fn new(data: Vec<f64>, shape: [usize; 2]) -> Self {
        assert_eq!(data.len(), shape[0] * shape[1], "data len != shape product");
        let t = Tensor(Rc::new(RefCell::new(TensorInner {
            grad: vec![0.0; data.len()],
            data,
            shape,
            backward: None,
        })));
        TAPE.with(|tape| tape.borrow_mut().push(t.clone()));
        t
    }

    pub fn zeros(shape: [usize; 2]) -> Self {
        let n = shape[0] * shape[1];
        Self::new(vec![0.0; n], shape)
    }

    pub fn scalar(v: f64) -> Self {
        Self::new(vec![v], [1, 1])
    }

    // ---- ACCESSORS ----------------------------------------------------------

    pub fn shape(&self) -> [usize; 2] {
        self.0.borrow().shape
    }
    pub fn rows(&self) -> usize {
        self.0.borrow().shape[0]
    }
    pub fn cols(&self) -> usize {
        self.0.borrow().shape[1]
    }
    pub fn len(&self) -> usize {
        let s = self.shape();
        s[0] * s[1]
    }

    pub fn data(&self) -> Vec<f64> {
        self.0.borrow().data.clone()
    }
    pub fn grad(&self) -> Vec<f64> {
        self.0.borrow().grad.clone()
    }

    pub fn item(&self) -> f64 {
        assert_eq!(self.len(), 1, "item() on non-scalar tensor");
        self.0.borrow().data[0]
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        let cols = self.cols();
        self.0.borrow().data[r * cols + c]
    }

    pub fn set_data_vec(&self, d: Vec<f64>) {
        self.0.borrow_mut().data = d;
    }

    pub fn zero_grad(&self) {
        let mut inner = self.0.borrow_mut();
        inner.grad.iter_mut().for_each(|g| *g = 0.0);
    }

    pub fn add_grad_vec(&self, dg: &[f64]) {
        let mut inner = self.0.borrow_mut();
        for (g, &d) in inner.grad.iter_mut().zip(dg.iter()) {
            *g += d;
        }
    }

    pub fn set_backward(&self, f: impl Fn() + 'static) {
        self.0.borrow_mut().backward = Some(Box::new(f));
    }

    // ---- BACKWARD -----------------------------------------------------------

    /// Reverse-mode autodiff. Call on the scalar loss tensor.
    ///
    /// Drains the global tape, propagates gradients in reverse creation order,
    /// then drops all backward closures to break `Rc` reference cycles so the
    /// computation graph is freed.
    pub fn backward(&self) {
        assert_eq!(self.len(), 1, "backward() requires scalar tensor");

        // Drain the tape — every tensor created since the last backward/clear.
        let tape = TAPE.with(|t| std::mem::take(&mut *t.borrow_mut()));

        // Seed the gradient of the loss
        self.0.borrow_mut().grad[0] = 1.0;

        // Propagate in reverse creation order (== valid reverse topological
        // order because every op output is created after its inputs).
        for t in tape.iter().rev() {
            let has_bwd = t.0.borrow().backward.is_some();
            if has_bwd {
                let f = {
                    let inner = t.0.borrow();
                    inner.backward.as_ref().unwrap() as *const Box<dyn Fn()>
                };
                // SAFETY: The closure is owned by the Rc and lives as long as
                // the TensorInner. We borrow it only for the duration of this
                // call and do not drop the Tensor until after.
                unsafe { (*f)() };
            }
        }

        // Break Rc cycles: every backward closure captures Rc<RefCell<TensorInner>>
        // references (including to its own output tensor). Dropping the closures
        // lets Rc reference counts reach zero so the graph is freed.
        for t in tape.iter() {
            t.0.borrow_mut().backward = None;
        }
    }

    /// Drain the tape and drop all backward closures without propagating
    /// gradients. Use after forward-only passes (e.g. evaluation) to free
    /// the computation graph.
    pub fn clear_tape() {
        let tape = TAPE.with(|t| std::mem::take(&mut *t.borrow_mut()));
        for t in tape.iter() {
            t.0.borrow_mut().backward = None;
        }
    }

    // ---- OPS ----------------------------------------------------------------

    /// Matrix multiply: [M, K] x [K, N] -> [M, N]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (m, k) = (self.rows(), self.cols());
        let (k2, n) = (other.rows(), other.cols());
        assert_eq!(k, k2, "matmul shape mismatch: [{m},{k}] x [{k2},{n}]");

        let a_data = self.data();
        let b_data = other.data();
        let mut c_data = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for l in 0..k {
                    s += a_data[i * k + l] * b_data[l * n + j];
                }
                c_data[i * n + j] = s;
            }
        }

        let out = Tensor::new(c_data, [m, n]);

        let a = self.clone();
        let b = other.clone();
        let out_c = out.clone();
        out.set_backward(move || {
            let out_grad = out_c.0.borrow().grad.clone();
            let a_data = a.0.borrow().data.clone();
            let b_data = b.0.borrow().data.clone();

            // dA = dOut @ B^T
            let mut da = vec![0.0; m * k];
            for i in 0..m {
                for l in 0..k {
                    let mut s = 0.0;
                    for j in 0..n {
                        s += out_grad[i * n + j] * b_data[l * n + j];
                    }
                    da[i * k + l] += s;
                }
            }
            a.add_grad_vec(&da);

            // dB = A^T @ dOut
            let mut db = vec![0.0; k * n];
            for l in 0..k {
                for j in 0..n {
                    let mut s = 0.0;
                    for i in 0..m {
                        s += a_data[i * k + l] * out_grad[i * n + j];
                    }
                    db[l * n + j] += s;
                }
            }
            b.add_grad_vec(&db);
        });

        out
    }

    /// Element-wise addition. Supports broadcasting: [M,N] + [1,N] or [M,N] + [M,N]
    pub fn add(&self, other: &Tensor) -> Tensor {
        let (m, n) = (self.rows(), self.cols());
        let a_data = self.data();
        let b_data = other.data();

        let broadcast_b = other.rows() == 1 && other.cols() == n;
        let same_shape = other.rows() == m && other.cols() == n;
        assert!(
            broadcast_b || same_shape,
            "add shape mismatch: [{m},{n}] + [{},{}]",
            other.rows(),
            other.cols()
        );

        let out_data: Vec<f64> = if same_shape {
            a_data
                .iter()
                .zip(b_data.iter())
                .map(|(a, b)| a + b)
                .collect()
        } else {
            let mut v = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    v[i * n + j] = a_data[i * n + j] + b_data[j];
                }
            }
            v
        };

        let out = Tensor::new(out_data, [m, n]);
        let a = self.clone();
        let b = other.clone();
        let out_c = out.clone();

        out.set_backward(move || {
            let og = out_c.0.borrow().grad.clone();
            a.add_grad_vec(&og);
            if same_shape {
                b.add_grad_vec(&og);
            } else {
                // Sum gradient over rows for broadcast dim
                let mut db = vec![0.0; n];
                for i in 0..m {
                    for j in 0..n {
                        db[j] += og[i * n + j];
                    }
                }
                b.add_grad_vec(&db);
            }
        });

        out
    }

    /// Scalar multiply
    pub fn mul_scalar(&self, s: f64) -> Tensor {
        let data: Vec<f64> = self.data().iter().map(|&x| x * s).collect();
        let out = Tensor::new(data, self.shape());
        let a = self.clone();
        let out_c = out.clone();
        out.set_backward(move || {
            let og: Vec<f64> = out_c.0.borrow().grad.iter().map(|&g| g * s).collect();
            a.add_grad_vec(&og);
        });
        out
    }

    /// Element-wise ReLU
    pub fn relu(&self) -> Tensor {
        let data: Vec<f64> = self.data().iter().map(|&x| x.max(0.0)).collect();
        let out = Tensor::new(data, self.shape());
        let a = self.clone();
        let a_data = self.data();
        let out_c = out.clone();
        out.set_backward(move || {
            let og = out_c.0.borrow().grad.clone();
            let dg: Vec<f64> = og
                .iter()
                .zip(a_data.iter())
                .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
                .collect();
            a.add_grad_vec(&dg);
        });
        out
    }

    /// Softmax over last dimension (each row independently)
    pub fn softmax(&self) -> Tensor {
        let (m, n) = (self.rows(), self.cols());
        let data = self.data();
        let mut out_data = vec![0.0; m * n];

        for i in 0..m {
            let row = &data[i * n..(i + 1) * n];
            let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = row.iter().map(|&x| (x - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for j in 0..n {
                out_data[i * n + j] = exps[j] / sum;
            }
        }

        let out = Tensor::new(out_data, [m, n]);
        let a = self.clone();
        let out_c = out.clone();

        out.set_backward(move || {
            let p = out_c.data();
            let og = out_c.0.borrow().grad.clone();
            // dL/dx_i = p_i * (dL/dy_i - sum_j(dL/dy_j * p_j))
            let mut da = vec![0.0; m * n];
            for i in 0..m {
                let mut dot = 0.0;
                for j in 0..n {
                    dot += og[i * n + j] * p[i * n + j];
                }
                for j in 0..n {
                    da[i * n + j] = p[i * n + j] * (og[i * n + j] - dot);
                }
            }
            a.add_grad_vec(&da);
        });

        out
    }

    /// Log of each element
    pub fn log(&self) -> Tensor {
        let data: Vec<f64> = self.data().iter().map(|&x| x.ln()).collect();
        let out = Tensor::new(data, self.shape());
        let a = self.clone();
        let a_data = self.data();
        let out_c = out.clone();
        out.set_backward(move || {
            let og = out_c.0.borrow().grad.clone();
            let dg: Vec<f64> = og.iter().zip(a_data.iter()).map(|(&g, &x)| g / x).collect();
            a.add_grad_vec(&dg);
        });
        out
    }

    /// Neg
    pub fn neg(&self) -> Tensor {
        self.mul_scalar(-1.0)
    }

    /// RMSNorm over last dimension (each row independently), no learned scale
    pub fn rmsnorm(&self) -> Tensor {
        let (m, n) = (self.rows(), self.cols());
        let data = self.data();
        let mut out_data = vec![0.0; m * n];
        let mut scales = vec![0.0; m]; // rms per row

        for i in 0..m {
            let row = &data[i * n..(i + 1) * n];
            let ms: f64 = row.iter().map(|&x| x * x).sum::<f64>() / n as f64;
            let scale = (ms + 1e-5).sqrt();
            scales[i] = scale;
            for j in 0..n {
                out_data[i * n + j] = data[i * n + j] / scale;
            }
        }

        let out = Tensor::new(out_data, [m, n]);
        let a = self.clone();
        let out_c = out.clone();

        out.set_backward(move || {
            let og = out_c.0.borrow().grad.clone();
            let x = a.data();
            let mut da = vec![0.0; m * n];

            for i in 0..m {
                let s = scales[i];
                // sum of og * y (where y = x/s)
                let mut dot = 0.0;
                for j in 0..n {
                    dot += og[i * n + j] * out_c.0.borrow().data[i * n + j];
                }
                for j in 0..n {
                    // d/dx_j (x_j/s) = 1/s - x_j^2 / (n * s^3)
                    // full: da[j] = (1/s) * og[j] - (x[j]/(n*s^3)) * sum(og*y)
                    da[i * n + j] = og[i * n + j] / s - x[i * n + j] * dot / (n as f64 * s * s * s);
                }
            }
            a.add_grad_vec(&da);
        });

        out
    }

    /// Cross-entropy loss: input is [1, vocab_size] logits, target is class index.
    /// Returns a scalar [1,1] tensor.
    pub fn cross_entropy(&self, target: usize) -> Tensor {
        assert_eq!(self.rows(), 1);
        let n = self.cols();
        let logits = self.data();

        // Numerically stable softmax
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let probs: Vec<f64> = exps.iter().map(|&e| e / sum).collect();

        let loss = -probs[target].ln();
        let out = Tensor::scalar(loss);

        let a = self.clone();
        let out_c = out.clone();

        out.set_backward(move || {
            let og = out_c.0.borrow().grad[0];
            let mut da = vec![0.0; n];
            for j in 0..n {
                da[j] = og * (probs[j] - if j == target { 1.0 } else { 0.0 });
            }
            a.add_grad_vec(&da);
        });

        out
    }

    /// Sum all elements into a scalar [1,1] tensor
    pub fn sum(&self) -> Tensor {
        let total: f64 = self.data().iter().sum();
        let out = Tensor::scalar(total);
        let a = self.clone();
        let n = self.len();
        let out_c = out.clone();
        out.set_backward(move || {
            let og = out_c.0.borrow().grad[0];
            a.add_grad_vec(&vec![og; n]);
        });
        out
    }

    /// Extract a single row as a [1, cols] tensor (shares no grad; used for embedding lookup)
    pub fn row(&self, idx: usize) -> Tensor {
        let n = self.cols();
        let data = self.data();
        let row_data = data[idx * n..(idx + 1) * n].to_vec();
        let out = Tensor::new(row_data, [1, n]);

        let a = self.clone();
        let out_c = out.clone();
        out.set_backward(move || {
            let og = out_c.0.borrow().grad.clone();
            let mut da = vec![0.0; a.len()];
            for j in 0..n {
                da[idx * n + j] += og[j];
            }
            a.add_grad_vec(&da);
        });

        out
    }

    /// Concatenate rows: [A, n] + [B, n] -> [A+B, n]
    pub fn vcat(tensors: &[Tensor]) -> Tensor {
        if tensors.is_empty() {
            panic!("vcat: empty slice");
        }
        let n = tensors[0].cols();
        let m: usize = tensors.iter().map(|t| t.rows()).sum();
        let data: Vec<f64> = tensors.iter().flat_map(|t| t.data()).collect();
        let out = Tensor::new(data, [m, n]);

        let ts: Vec<Tensor> = tensors.to_vec();
        let out_c = out.clone();
        out.set_backward(move || {
            let og = out_c.0.borrow().grad.clone();
            let mut offset = 0;
            for t in &ts {
                let rows = t.rows();
                let slice = og[offset * n..(offset + rows) * n].to_vec();
                t.add_grad_vec(&slice);
                offset += rows;
            }
        });

        out
    }

    /// Slice rows [start, end)
    pub fn slice_rows(&self, start: usize, end: usize) -> Tensor {
        let n = self.cols();
        let data = self.data();
        let slice_data = data[start * n..end * n].to_vec();
        let out = Tensor::new(slice_data, [end - start, n]);

        let a = self.clone();
        let total_rows = self.rows();
        let out_c = out.clone();
        out.set_backward(move || {
            let og = out_c.0.borrow().grad.clone();
            let mut da = vec![0.0; total_rows * n];
            let len = (end - start) * n;
            da[start * n..start * n + len].copy_from_slice(&og);
            a.add_grad_vec(&da);
        });

        out
    }
}

// ---------------------------------------------------------------------------
// Adam optimizer state
// ---------------------------------------------------------------------------

pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub step: usize,
    m: Vec<f64>,
    v: Vec<f64>,
}

impl Adam {
    pub fn new(n_params: usize, lr: f64) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            step: 0,
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
        }
    }

    pub fn step_params(&mut self, params: &[Tensor], lr_scale: f64) {
        self.step += 1;
        let t = self.step as f64;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        let mut offset = 0;
        for p in params {
            let mut inner = p.0.borrow_mut();
            for i in 0..inner.data.len() {
                let g = inner.grad[i];
                self.m[offset + i] = self.beta1 * self.m[offset + i] + (1.0 - self.beta1) * g;
                self.v[offset + i] = self.beta2 * self.v[offset + i] + (1.0 - self.beta2) * g * g;
                let m_hat = self.m[offset + i] / bc1;
                let v_hat = self.v[offset + i] / bc2;
                inner.data[i] -= lr_scale * self.lr * m_hat / (v_hat.sqrt() + self.eps);
                inner.grad[i] = 0.0;
            }
            offset += inner.data.len();
        }
    }
}
