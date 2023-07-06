use anyhow::Result;
use candle::{safetensors::SafeTensors, DType, Device, Shape, Tensor, D};
use std::collections::HashMap;

pub struct VarBuilder<'a> {
    safetensors: Option<(HashMap<String, usize>, Vec<SafeTensors<'a>>)>,
    dtype: DType,
    device: Device,
}

impl<'a> VarBuilder<'a> {
    pub fn from_safetensors(
        safetensors: Vec<SafeTensors<'a>>,
        dtype: DType,
        device: &Device,
    ) -> Self {
        let mut routing = HashMap::new();
        for (index, sf) in safetensors.iter().enumerate() {
            for k in sf.names() {
                routing.insert(k.to_string(), index);
            }
        }
        Self {
            safetensors: Some((routing, safetensors)),
            device: device.clone(),
            dtype,
        }
    }

    pub fn zeros(dtype: DType, device: &Device) -> Self {
        Self {
            safetensors: None,
            device: device.clone(),
            dtype,
        }
    }

    pub fn get<S: Into<Shape>>(&self, s: S, tensor_name: &str) -> candle::Result<Tensor> {
        let s: Shape = s.into();
        match &self.safetensors {
            None => Tensor::zeros(s, self.dtype, &self.device),
            Some((routing, safetensors)) => {
                // Unwrap or 0  just to let the proper error flow.
                let index = routing.get(tensor_name).unwrap_or(&0);
                let tensor = safetensors[*index]
                    .tensor(tensor_name, &self.device)?
                    .to_dtype(self.dtype)?;
                if *tensor.shape() != s {
                    let msg = format!("shape mismatch for {tensor_name}");
                    Err(candle::Error::UnexpectedShape {
                        msg,
                        expected: s,
                        got: tensor.shape().clone(),
                    })?
                }
                Ok(tensor)
            }
        }
    }
}

#[derive(Debug)]
struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    fn load(size1: usize, size2: usize, p: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get((size2, size1), &format!("{p}.weight"))?;
        let bias = vb.get(size2, &format!("{p}.bias"))?;
        Ok(Self {
            weight,
            bias: Some(bias),
        })
    }

    fn load_no_bias(size1: usize, size2: usize, p: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get((size2, size1), &format!("{p}.weight"))?;
        Ok(Self { weight, bias: None })
    }

    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let (bsize, _, _) = x.shape().r3()?;
        let w = self.weight.broadcast_left(bsize)?.t()?;
        let x = x.matmul(&w)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

#[derive(Debug)]
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self { weight, bias, eps }
    }

    fn load(size: usize, eps: f64, p: &str, vb: &VarBuilder) -> Result<Self> {
        let (weight, bias) = match (
            vb.get(size, &format!("{p}.weight")),
            vb.get(size, &format!("{p}.bias")),
        ) {
            (Ok(weight), Ok(bias)) => (weight, bias),
            (Err(err), _) | (_, Err(err)) => {
                if let (Ok(weight), Ok(bias)) = (
                    vb.get(size, &format!("{p}.gamma")),
                    vb.get(size, &format!("{p}.beta")),
                ) {
                    (weight, bias)
                } else {
                    return Err(err.into());
                }
            }
        };
        Ok(Self { weight, bias, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_bsize, _seq_len, hidden_size) = x.shape().r3()?;
        let mean_x = (x.sum(&[2])? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = ((&x * &x)?.sum(&[2])? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?;
        Ok(x)
    }
}

#[derive(Debug)]
struct Dropout {
    pr: f64,
}

impl Dropout {
    fn new(pr: f64) -> Self {
        Self { pr }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(x.clone())
    }
}

#[derive(Debug)]
struct Embedding {
    embeddings: Tensor,
    hidden_size: usize,
}

impl Embedding {
    fn new(embeddings: Tensor, hidden_size: usize) -> Self {
        Self {
            embeddings,
            hidden_size,
        }
    }

    fn load(vocab_size: usize, hidden_size: usize, p: &str, vb: &VarBuilder) -> Result<Self> {
        let embeddings = vb.get((vocab_size, hidden_size), &format!("{p}.weight"))?;
        Ok(Self::new(embeddings, hidden_size))
    }

    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = Tensor::embedding(&indexes, &self.embeddings)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}

// https://raw.githubusercontent.com/huggingface/transformers/030c863aaa0165e98352b61697430bf69bf33755/src/transformers/models/falcon/configuration_falcon.py
#[derive(Debug)]
pub struct Config {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    layer_norm_epsilon: f64,
    initializer_range: f64,
    use_cache: bool,
    bos_token_id: u32,
    eos_token_id: u32,
    hidden_dropout: f64,
    attention_dropout: f64,
    n_head_kv: Option<usize>,
    alibi: bool,
    multi_query: bool,
    parallel_attn: bool,
    bias: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 65024,
            hidden_size: 4544,
            num_hidden_layers: 32,
            num_attention_heads: 71,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            use_cache: true,
            bos_token_id: 11,
            eos_token_id: 11,
            hidden_dropout: 0.0,
            attention_dropout: 0.0,
            n_head_kv: None,
            alibi: false,
            multi_query: true,
            parallel_attn: true,
            bias: false,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.alibi {
            anyhow::bail!("alibi is not supported");
        }
        if self.n_head_kv.is_some() {
            anyhow::bail!("n_head_kv is not supported");
        }
        Ok(())
    }

    // https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json
    pub fn falcon7b() -> Self {
        // This is currently on par with the defaults, the defaults come from the Python default
        // arguments for the config initialization whereas the following come from the json config.
        Self {
            vocab_size: 65024,
            hidden_size: 4544,
            num_hidden_layers: 32,
            num_attention_heads: 71,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            use_cache: true,
            bos_token_id: 11,
            eos_token_id: 11,
            hidden_dropout: 0.,
            attention_dropout: 0.,
            n_head_kv: None,
            alibi: false,
            multi_query: true,
            parallel_attn: true,
            bias: false,
        }
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn rotary(&self) -> bool {
        !self.alibi
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let l = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, l / 2)?;
    let x2 = x.narrow(D::Minus1, l / 2, l - l / 2)?;
    let x21 = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    Ok(x21)
}

#[derive(Debug)]
struct FalconRotaryEmbedding {
    inv_freq: Tensor,
}

impl FalconRotaryEmbedding {
    fn load(p: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let inv_freq = vb.get((1, cfg.head_dim()), &format!("{p}.inv_freq"))?;
        Ok(Self { inv_freq })
    }

    fn cos_sin(
        &mut self,
        seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        // TODO: Add the cache.
        let t: Vec<_> = (0..seq_len).map(|c| c as u32).collect();
        let t = Tensor::new(t.as_slice(), device)?.to_dtype(dtype)?;
        let freqs = t.matmul(&self.inv_freq)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        Ok((cos, sin))
    }

    fn forward(&mut self, query: &Tensor, key: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_batch, seq_len, _head_dim) = query.shape().r3()?;
        let (cos, sin) = self.cos_sin(seq_len, &query.device(), query.dtype())?;
        let qs = ((query * &cos)? + (&rotate_half(query)? * &sin)?)?;
        let ks = ((key * &cos)? + (&rotate_half(key)? * &sin)?)?;
        Ok((qs, ks))
    }
}

#[derive(Debug)]
struct FalconAttention {
    maybe_rotary: Option<FalconRotaryEmbedding>,
}

impl FalconAttention {
    fn load(p: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let maybe_rotary = if cfg.rotary() {
            let rotary = FalconRotaryEmbedding::load(&format!("{p}.maybe_rotary"), vb, cfg)?;
            Some(rotary)
        } else {
            None
        };
        Ok(Self { maybe_rotary })
    }

    fn forward(&self, _x: Tensor) -> Result<Tensor> {
        todo!()
    }
}

#[derive(Debug)]
struct FalconMlp {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
    dropout: Dropout,
}

impl FalconMlp {
    fn load(p: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let dense_h_to_4h = Linear::load(h, 4 * h, &format!("{p}.dense_h_to_4h"), vb)?;
        let dense_4h_to_h = Linear::load(4 * h, h, &format!("{p}.dense_4h_to_h"), vb)?;
        let dropout = Dropout::new(cfg.hidden_dropout);
        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
            dropout,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.dense_4h_to_h.forward(x)?.gelu()?;
        let x = self.dense_h_to_4h.forward(&x)?;
        Ok(x)
    }
}

#[derive(Debug)]
struct FalconDecoderLayer {
    inp_layernorm: LayerNorm,
    self_attention: FalconAttention,
    post_attention_layernorm: Option<LayerNorm>,
    mlp: FalconMlp,
}

impl FalconDecoderLayer {
    fn load(p: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let mlp = FalconMlp::load(&format!("{p}.mlp"), vb, cfg)?;
        let inp_layernorm = LayerNorm::load(
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
            &format!("{p}.input_layernorm"),
            vb,
        )?;
        let self_attention = FalconAttention::load(&format!("{p}.self_attention"), vb, cfg)?;
        let post_attention_layernorm = if cfg.parallel_attn {
            None
        } else {
            let ln = LayerNorm::load(
                cfg.hidden_size,
                cfg.layer_norm_epsilon,
                &format!("{p}.post_attention_layernorm"),
                vb,
            )?;
            Some(ln)
        };
        Ok(Self {
            inp_layernorm,
            self_attention,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

#[derive(Debug)]
pub struct Falcon {
    word_embeddings: Embedding,
    h: Vec<FalconDecoderLayer>,
    ln_f: LayerNorm,
    config: Config,
}

impl Falcon {
    pub fn load(vb: &VarBuilder, cfg: Config) -> Result<Self> {
        let word_embeddings =
            Embedding::load(cfg.vocab_size, cfg.hidden_size, "word_embeddings", vb)?;
        let h = (0..cfg.num_hidden_layers)
            .map(|i| FalconDecoderLayer::load(&format!("h.{i}"), vb, &cfg))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = LayerNorm::load(cfg.hidden_size, cfg.layer_norm_epsilon, "ln_f", vb)?;
        Ok(Self {
            word_embeddings,
            h,
            ln_f,
            config: cfg,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_bsize, _seq_len) = input_ids.shape().r2()?;
        let mut hidden_state = self.word_embeddings.forward(input_ids)?;
        for block in self.h.iter() {
            hidden_state = block.forward(&hidden_state)?;
        }
        let hidden_state = self.ln_f.forward(&hidden_state)?;
        Ok(hidden_state)
    }
}
