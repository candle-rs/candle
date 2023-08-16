#![allow(dead_code)]
use clap::Parser;
use std::collections::HashMap;

use candle::quantized::ggml_file::Content;
use candle::quantized::{QMatMul, QTensor};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::Embedding;

struct RmsNorm {
    scale: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(scale: QTensor) -> Result<Self> {
        let scale = scale.dequantize(&Device::Cpu)?;
        Ok(Self { scale, eps: 1e-5 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let norm_x = norm_x.broadcast_as((b_sz, seq_len, hidden_size))?;
        let x_normed = (x / (norm_x + self.eps)?.sqrt()?)?;
        let size = self.scale.dims1()?;
        let scale = self
            .scale
            .to_dtype(DType::F32)?
            .broadcast_as((b_sz, seq_len, size))?;
        let x = (scale * x_normed)?;
        Ok(x)
    }
}

struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
    ffn_norm: RmsNorm,
}

struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
}

struct WeightMap(HashMap<String, QTensor>);
impl WeightMap {
    fn get(&mut self, name: &str) -> Result<QTensor> {
        match self.0.remove(name) {
            None => candle::bail!("cannot find tensor with name '{name}'"),
            Some(tensor) => Ok(tensor),
        }
    }
}

impl ModelWeights {
    fn new(mut ct: Content) -> Result<Self> {
        let cpu = &Device::Cpu;
        let tok_embeddings = ct.remove("tok_embeddings.weight")?;
        let tok_embeddings = tok_embeddings.dequantize(cpu)?;
        let norm = RmsNorm::new(ct.remove("norm.weight")?)?;
        let output = QMatMul::from_qtensor(ct.remove("output.weight")?);
        let mut layers = Vec::with_capacity(ct.hparams.n_layer as usize);
        for layer_idx in 0..ct.hparams.n_layer {
            let prefix = format!("layers.{layer_idx}");
            let attention_wq = ct.remove(&format!("layers.{layer_idx}.attention.wq.weight"))?;
            let attention_wk = ct.remove(&format!("{prefix}.attention.wk.weight"))?;
            let attention_wv = ct.remove(&format!("{prefix}.attention.wv.weight"))?;
            let attention_wo = ct.remove(&format!("{prefix}.attention.wo.weight"))?;
            let feed_forward_w1 = ct.remove(&format!("{prefix}.feed_forward.w1.weight"))?;
            let feed_forward_w2 = ct.remove(&format!("{prefix}.feed_forward.w2.weight"))?;
            let feed_forward_w3 = ct.remove(&format!("{prefix}.feed_forward.w3.weight"))?;
            let attention_norm = ct.remove(&format!("{prefix}.attention_norm.weight"))?;
            let ffn_norm = ct.remove(&format!("{prefix}.ffn_norm.weight"))?;
            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq),
                attention_wk: QMatMul::from_qtensor(attention_wk),
                attention_wv: QMatMul::from_qtensor(attention_wv),
                attention_wo: QMatMul::from_qtensor(attention_wo),
                attention_norm: RmsNorm::new(attention_norm)?,
                feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1),
                feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2),
                feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3),
                ffn_norm: RmsNorm::new(ffn_norm)?,
            })
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, ct.hparams.n_vocab as usize),
            layers,
            norm,
            output,
        })
    }

    fn forward(&self, x: &Tensor, _index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mut layer_in = self.tok_embeddings.forward(x)?;
        for (_layer_idx, layer) in self.layers.iter().enumerate() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            // TODO: implement the attention bit.
            let attn = x.clone();
            let x = (attn + residual)?;

            // MLP
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let w1 = layer.feed_forward_w1.forward(&x)?;
            let w3 = layer.feed_forward_w3.forward(&x)?;
            let mlp = layer
                .feed_forward_w2
                .forward(&(candle_nn::ops::silu(&w1)? * w3)?)?;
            layer_in = (mlp + residual)?;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        self.output.forward(&x)
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// GGML file to load, typically a .bin file generated by the quantize command from llama.cpp
    #[arg(long)]
    model: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut file = std::fs::File::open(args.model)?;
    let start = std::time::Instant::now();
    let model = Content::read(&mut file)?;

    let mut total_size_in_bytes = 0;
    for (_, tensor) in model.tensors.iter() {
        let elem_count = tensor.shape().elem_count();
        total_size_in_bytes += elem_count * tensor.dtype().type_size() / tensor.dtype().blck_size();
    }
    let total_size = if total_size_in_bytes < 1_000 {
        format!("{}B", total_size_in_bytes)
    } else if total_size_in_bytes < 1_000_000 {
        format!("{:.2}KB", total_size_in_bytes as f64 / 1e3)
    } else if total_size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", total_size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", total_size_in_bytes as f64 / 1e9)
    };

    println!(
        "loaded {:?} tensors ({}) in {:.2}s",
        model.tensors.len(),
        total_size,
        start.elapsed().as_secs_f32(),
    );
    println!("params: {:?}", model.hparams);
    let _model = ModelWeights::new(model);
    Ok(())
}
