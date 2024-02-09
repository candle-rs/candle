#![allow(unused)]
use crate::models::with_tracing::Linear;
use candle::{Module, Result, Tensor, D};
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct Config {
    pub num_layers: usize,
    pub padded_vocab_size: usize,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub kv_channels: usize,
    pub num_attention_heads: usize,
    pub seq_length: usize,
    pub layernorm_epsilon: f64,
    pub rmsnorm: bool,
    pub apply_residual_connection_post_layernorm: bool,
    pub post_layer_norm: bool,
    pub add_bias_linear: bool,
    pub add_qkv_bias: bool,
    pub bias_dropout_fusion: bool,
    pub multi_query_attention: bool,
    pub multi_query_group_num: usize,
    pub apply_query_key_layer_scaling: bool,
    pub attention_softmax_in_fp32: bool,
    pub fp32_residual_connection: bool,
}

impl Config {
    fn glm3_6b() -> Self {
        Self {
            num_layers: 28,
            padded_vocab_size: 65024,
            hidden_size: 4096,
            ffn_hidden_size: 13696,
            kv_channels: 128,
            num_attention_heads: 32,
            seq_length: 8192,
            layernorm_epsilon: 1e-5,
            rmsnorm: true,
            apply_residual_connection_post_layernorm: false,
            post_layer_norm: true,
            add_bias_linear: false,
            add_qkv_bias: true,
            bias_dropout_fusion: true,
            multi_query_attention: true,
            multi_query_group_num: 2,
            apply_query_key_layer_scaling: true,
            attention_softmax_in_fp32: true,
            fp32_residual_connection: false,
        }
    }
}

fn linear(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    if bias {
        crate::models::with_tracing::linear(in_dim, out_dim, vb)
    } else {
        crate::models::with_tracing::linear_no_bias(in_dim, out_dim, vb)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cache: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let dev = vb.device();
        let rotary_dim = cfg.kv_channels;
        let n_elem = rotary_dim / 2;
        let inv_freq: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / 10_000f64.powf(i as f64 / n_elem as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, cfg.seq_length as u32, dev)?
            .to_dtype(dtype)?
            .reshape((cfg.seq_length, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cache = Tensor::cat(&[&freqs.cos()?, &freqs.sin()?], D::Minus1)?;
        Ok(Self { cache })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct CoreAttention {
    coeff: Option<f64>,
    norm_factor: f64,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

impl CoreAttention {
    fn new(layer_number: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let norm_factor = (cfg.kv_channels as f64).sqrt();
        let (norm_factor, coeff) = if cfg.apply_query_key_layer_scaling {
            let coeff = f64::max(1.0, layer_number as f64);
            (norm_factor * coeff, Some(coeff))
        } else {
            (norm_factor, None)
        };
        Ok(Self { coeff, norm_factor })
    }

    fn forward(
        &self,
        query_layer: &Tensor,
        key_layer: &Tensor,
        value_layer: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let output_size = (
            query_layer.dim(1)?,
            query_layer.dim(2)?,
            query_layer.dim(0)?,
            key_layer.dim(0)?,
        );
        let query_layer =
            query_layer.reshape((output_size.2, output_size.0 * output_size.1, ()))?;
        let key_layer = key_layer.reshape((output_size.3, output_size.0 * output_size.1, ()))?;
        let matmul_result = Tensor::matmul(
            &query_layer.transpose(0, 1)?,
            &key_layer.transpose(0, 1)?.transpose(1, 2)?,
        )?;
        let matmul_result = (matmul_result / self.norm_factor)?.reshape(output_size)?;
        let attention_scores = masked_fill(&matmul_result, attention_mask, f32::NEG_INFINITY)?;
        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;

        let output_size = (
            value_layer.dim(1)?,
            value_layer.dim(2)?,
            query_layer.dim(0)?,
            value_layer.dim(3)?,
        );
        let value_layer =
            value_layer.reshape((value_layer.dim(0)?, output_size.0 * output_size.1, ()))?;
        let attention_probs =
            attention_probs.reshape((output_size.0 * output_size.1, output_size.2, ()))?;
        let context_layer = Tensor::matmul(&attention_probs, &value_layer.transpose(0, 1)?)?;
        let context_layer = context_layer.reshape(output_size)?;
        let context_layer = context_layer.permute((2, 0, 1, 3))?.contiguous()?;
        context_layer.flatten_from(D::Minus2)
    }
}

#[derive(Debug, Clone)]
struct SelfAttention {
    query_key_value: Linear,
    core_attention: CoreAttention,
    dense: Linear,
    multi_query_attention: bool,
    num_attention_heads_per_partition: usize,
    num_multi_query_groups_per_partition: usize,
    hidden_size_per_attention_head: usize,
}

impl SelfAttention {
    fn new(layer_number: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let projection_size = cfg.kv_channels * cfg.num_attention_heads;
        let hidden_size_per_attention_head = projection_size / cfg.num_attention_heads;
        let qkv_hidden_size = if cfg.multi_query_attention {
            projection_size + 2 * hidden_size_per_attention_head * cfg.multi_query_group_num
        } else {
            3 * projection_size
        };
        let query_key_value = linear(
            cfg.hidden_size,
            qkv_hidden_size,
            cfg.add_bias_linear || cfg.add_qkv_bias,
            vb.pp("query_key_value"),
        )?;
        let core_attention = CoreAttention::new(layer_number, cfg, vb.pp("core_attention"))?;
        let dense = linear(
            cfg.hidden_size,
            cfg.hidden_size,
            cfg.add_bias_linear,
            vb.pp("dense"),
        )?;
        Ok(Self {
            query_key_value,
            core_attention,
            dense,
            multi_query_attention: cfg.multi_query_attention,
            num_attention_heads_per_partition: cfg.num_attention_heads,
            num_multi_query_groups_per_partition: cfg.multi_query_group_num,
            hidden_size_per_attention_head: cfg.kv_channels,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mixed_x_layer = xs.apply(&self.query_key_value)?;
        if !self.multi_query_attention {
            candle::bail!("only multi_query_attention=true is supported")
        }
        let hpa = self.hidden_size_per_attention_head;
        let query_layer =
            mixed_x_layer.narrow(D::Minus1, 0, self.num_attention_heads_per_partition * hpa)?;
        let key_layer = mixed_x_layer.narrow(
            D::Minus1,
            self.num_attention_heads_per_partition * hpa,
            self.num_multi_query_groups_per_partition * hpa,
        )?;
        let value_layer = mixed_x_layer.narrow(
            D::Minus1,
            self.num_attention_heads_per_partition * hpa
                + self.num_multi_query_groups_per_partition * hpa,
            self.num_multi_query_groups_per_partition * hpa,
        )?;

        let context_layer =
            self.core_attention
                .forward(&query_layer, &key_layer, &value_layer, attention_mask)?;
        let output = context_layer.apply(&self.dense)?;
        Ok(output)
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
struct MLP {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense_h_to_4h = linear(
            cfg.hidden_size,
            cfg.ffn_hidden_size * 2,
            cfg.add_bias_linear,
            vb.pp("dense_h_to_4h"),
        )?;
        let dense_4h_to_h = linear(
            cfg.ffn_hidden_size * 2,
            cfg.hidden_size,
            cfg.add_bias_linear,
            vb.pp("dense_h_to_4h"),
        )?;
        Ok(Self {
            dense_4h_to_h,
            dense_h_to_4h,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.dense_h_to_4h)?
            .apply(&candle_nn::Activation::Swiglu)?
            .apply(&self.dense_4h_to_h)
    }
}

#[derive(Debug, Clone)]
struct Block {
    input_layernorm: candle_nn::LayerNorm,
    self_attention: SelfAttention,
    post_attention_layernorm: candle_nn::LayerNorm,
    mlp: MLP,
    apply_residual_connection_post_layernorm: bool,
}

impl Block {
    fn new(layer_number: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let input_layernorm = if cfg.rmsnorm {
            candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.layernorm_epsilon,
                vb.pp("input_layernorm"),
            )?
            .into_inner()
        } else {
            candle_nn::layer_norm(
                cfg.hidden_size,
                cfg.layernorm_epsilon,
                vb.pp("input_layernorm"),
            )?
        };
        let post_attention_layernorm = if cfg.rmsnorm {
            candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.layernorm_epsilon,
                vb.pp("post_attention_layernorm"),
            )?
            .into_inner()
        } else {
            candle_nn::layer_norm(
                cfg.hidden_size,
                cfg.layernorm_epsilon,
                vb.pp("post_attention_layernorm"),
            )?
        };
        let self_attention = SelfAttention::new(layer_number, cfg, vb.pp("self_attention"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        Ok(Self {
            input_layernorm,
            self_attention,
            post_attention_layernorm,
            mlp,
            apply_residual_connection_post_layernorm: cfg.apply_residual_connection_post_layernorm,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let layernorm_output = xs.apply(&self.input_layernorm)?;
        let attention_output = self
            .self_attention
            .forward(&layernorm_output, attention_mask)?;
        let residual = if self.apply_residual_connection_post_layernorm {
            &layernorm_output
        } else {
            xs
        };
        let layernorm_input = (residual + attention_output)?;
        let layernorm_output = layernorm_input.apply(&self.post_attention_layernorm)?;
        let mlp_output = layernorm_output.apply(&self.mlp)?;
        let residual = if self.apply_residual_connection_post_layernorm {
            &layernorm_output
        } else {
            &layernorm_input
        };
        mlp_output + residual
    }
}

#[derive(Debug, Clone)]
struct Transformer {
    layers: Vec<Block>,
    final_layernorm: Option<candle_nn::LayerNorm>,
}

impl Transformer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_l = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for layer_index in 0..cfg.num_layers {
            let block = Block::new(layer_index + 1, cfg, vb_l.pp(layer_index))?;
            layers.push(block)
        }
        let final_layernorm = if cfg.post_layer_norm {
            let ln = if cfg.rmsnorm {
                candle_nn::rms_norm(
                    cfg.hidden_size,
                    cfg.layernorm_epsilon,
                    vb.pp("final_layernorm"),
                )?
                .into_inner()
            } else {
                candle_nn::layer_norm(
                    cfg.hidden_size,
                    cfg.layernorm_epsilon,
                    vb.pp("final_layernorm"),
                )?
            };
            Some(ln)
        } else {
            None
        };
        Ok(Self {
            layers,
            final_layernorm,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for block in self.layers.iter() {
            xs = block.forward(&xs, attention_mask)?
        }
        match self.final_layernorm.as_ref() {
            None => Ok(xs),
            Some(ln) => xs.apply(ln),
        }
    }
}

#[derive(Debug, Clone)]
struct Embedding {
    word_embeddings: candle_nn::Embedding,
    fp32_residual_connection: bool,
}

impl Embedding {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let word_embeddings = candle_nn::embedding(
            cfg.padded_vocab_size,
            cfg.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        Ok(Self {
            word_embeddings,
            fp32_residual_connection: cfg.fp32_residual_connection,
        })
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.word_embeddings.forward(xs)?.transpose(0, 1)?; // b,s,h -> s,b,h
        if self.fp32_residual_connection {
            xs.to_dtype(candle::DType::F32)
        } else {
            xs.contiguous()
        }
    }
}

#[derive(Debug, Clone)]
struct Model {
    embedding: Embedding,
    encoder: Transformer,
    output_layer: Linear,
}

impl Model {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("transformer");
        let embedding = Embedding::new(cfg, vb.pp("embedding"))?;
        let encoder = Transformer::new(cfg, vb.pp("encoder"))?;
        let output_layer = linear(
            cfg.hidden_size,
            cfg.padded_vocab_size,
            false,
            vb.pp("output_layer"),
        )?;
        Ok(Self {
            embedding,
            encoder,
            output_layer,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}
