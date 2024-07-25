use candle::{D, IndexOp, Result, Tensor};
use candle::D::Minus1;
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

const IMG_SIZE: usize = 518;
const PATCH_SIZE: usize = 14;
const NUM_CLASSES: usize = 1000;

fn linear(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Linear> {
    if bias {
        candle_nn::linear(in_dim, out_dim, vb)
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)
    }
}

#[derive(Debug)]
struct Attention {
    qkv: Linear,
    output: Linear,
    num_heads: usize,
    scale: f64,
}

impl Attention {
    fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        proj_bias: bool,
    ) -> Result<Self> {
        let qkv = linear(vb.pp("qkv"), dim, 3 *dim, qkv_bias)?;

        let output = linear(vb.pp("proj"), dim, dim, proj_bias)?;
        let scale = 1. / ((dim / num_heads) as f64).sqrt();
        Ok(Self {
            qkv,
            output,
            num_heads,
            scale,
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, n, 3, self.num_heads, c / self.num_heads))?
            .transpose(1, 2)? // 02134
            .transpose(0, 1)? // 20134
            .transpose(2, 3)?; // 20314
        let q = (qkv.i(0)? * self.scale)?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;
        let attn = candle_nn::ops::softmax(&q.matmul(&k.t()?)?, Minus1)?;
        let attn = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, n, c))?;
        self.output.forward(&attn)
    }
}

#[derive(Debug)]
struct LayerScale {
    lambda: Tensor,
}

impl LayerScale {
    fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self { lambda: gamma })
    }
}

impl Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.lambda)
    }
}

#[derive(Debug)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(vb: VarBuilder, in_features: usize, hidden_features: usize, bias: bool) -> Result<Self> {
        let out_features = in_features;
        let fc1 = linear(vb.pp("fc1"), in_features, hidden_features, bias)?;
        let fc2 = linear(vb.pp("fc2"), hidden_features, out_features, bias)?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?.gelu()?;
        self.fc2.forward(&xs)
    }
}

#[derive(Debug)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    ls1: LayerScale,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
}

impl Block {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let norm1 = layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let attn = Attention::new(vb.pp("attn"), dim, num_heads, true, true)?;
        let ls1 = LayerScale::new(vb.pp("ls1"), dim)?;
        let norm2 = layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        let mlp = Mlp::new(vb.pp("mlp"), dim, dim * 4, true)?;
        let ls2 = LayerScale::new(vb.pp("ls2"), dim)?;
        Ok(Self {
            norm1,
            attn,
            ls1,
            norm2,
            mlp,
            ls2,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self
            .ls1
            .forward(&self.attn.forward(&self.norm1.forward(xs)?)?)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .ls2
            .forward(&self.mlp.forward(&self.norm2.forward(&xs)?)?)?;
        xs + residual
    }
}

#[derive(Debug)]
struct PatchEmbed {
    proj: candle_nn::Conv2d,
    patch_size: (usize, usize),
    num_patches: usize,
}

impl PatchEmbed {
    fn new(
        vb: VarBuilder,
        img_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
    ) -> Result<Self> {
        let config = candle_nn::Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(in_chans, embed_dim, patch_size, config, vb.pp("proj"))?;
        let num_patches = (img_size / patch_size) * (img_size / patch_size);
        Ok(Self {
            proj,
            patch_size: (patch_size, patch_size),
            num_patches,
        })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = xs.dims4()?;
        let (patch_h, patch_w) = self.patch_size;
        if (h % patch_h) != 0 {
            candle::bail!("image height {h} is not a multiple of patch height {patch_h}")
        }
        if (w % patch_w) != 0 {
            candle::bail!("image width {w} is not a multiple of patch width {patch_w}")
        }
        let xs = self.proj.forward(xs)?;
        let (b, c, h, w) = xs.dims4()?;
        // flatten embeddings.
        xs.reshape((b, c, h * w))?.transpose(1, 2)
    }
}

#[derive(Debug)]
pub struct DinoVisionTransformer {
    patch_embed: PatchEmbed,
    cls_token: Tensor,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    norm: LayerNorm,
    head: Option<Linear>,
}

impl DinoVisionTransformer {
    pub fn new(
        vb: VarBuilder,
        vb_head: Option<VarBuilder>,
        depth: usize,
        embed_dim: usize,
        num_heads: usize,
    ) -> Result<Self> {
        let patch_embed = PatchEmbed::new(
            vb.pp("patch_embed"),
            IMG_SIZE,
            PATCH_SIZE,
            3,
            embed_dim,
        )?;
        let cls_token = vb.get((1, 1, embed_dim), "cls_token")?;
        let num_tokens = 1;
        let pos_embed = vb.get(
            (1, patch_embed.num_patches + num_tokens, embed_dim),
            "pos_embed",
        )?;
        let head = match vb_head {
            Some(vb_head) => Some(linear(vb_head, 2 * embed_dim, NUM_CLASSES, true)?),
            None => None,
        };
        let norm = layer_norm(embed_dim, 1e-5, vb.pp("norm"))?;
        let vb_layer = vb.pp("blocks");
        let blocks = (0..depth)
            .map(|i| Block::new(vb_layer.pp(&i.to_string()), embed_dim, num_heads))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            norm,
            head,
        })
    }

    fn interpolate_pos_encoding(&self, xs: &Tensor, w: usize, h: usize) -> Result<Tensor> {
        let npatch = xs.dim(1)? - 1;
        let n = self.pos_embed.dim(1)? - 1;
        let sqrt_n = (n as f64).sqrt();
        if npatch == n && w == h {
            return Ok(xs.clone());
        }
        let class_pos_embed = self.pos_embed.i((.., ..1))?;
        let patch_pos_embed = self.pos_embed.i((.., 1..))?;
        let dim = xs.dim(D::Minus1)?;
        let (w0, h0) = ((w / PATCH_SIZE) as f64 + 0.1, (h / PATCH_SIZE) as f64 + 0.1);
        let patch_pos_embed = patch_pos_embed
            .reshape((1, sqrt_n as usize, sqrt_n as usize, dim))?
            .transpose(2, 3)?
            .transpose(1, 2)?;

        // This uses bicubic interpolation in the original implementation.
        let patch_pos_embed = patch_pos_embed.upsample_nearest2d(h0 as usize, w0 as usize)?;
        let el_count = patch_pos_embed.shape().elem_count();
        let patch_pos_embed =
            patch_pos_embed
                .transpose(1, 2)?
                .transpose(2, 3)?
                .reshape((1, el_count / dim, dim))?;
        Tensor::cat(&[&class_pos_embed, &patch_pos_embed], 1)
    }

    fn prepare_tokens_with_mask(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _nc, w, h) = xs.dims4()?;
        let xs = self.patch_embed.forward(xs)?;
        let xs = Tensor::cat(&[&self.cls_token, &xs], 1)?;
        &xs + &self.interpolate_pos_encoding(&xs, w, h)?
    }

    fn get_intermediate_layers_not_chunked(
        &self,
        xs: &Tensor,
        blocks_to_take: &[usize],
    ) -> Result<Vec<Tensor>> {

        let mut xs = self.prepare_tokens_with_mask(xs)?;
        let mut output = Vec::new();
        for (i, blk) in self.blocks.iter().enumerate() {
            xs = blk.forward(&xs)?;
            if blocks_to_take.contains(&i) {
                output.push(xs.clone());
            }
        }
        if output.len() != blocks_to_take.len() {
            candle::bail!(
                "only {} / {} blocks found",
                output.len(),
                blocks_to_take.len()
            );
        }
        Ok(output)
    }

    pub fn get_intermediate_layers(
        &self,
        xs: &Tensor,
        blocks_to_take: &[usize],
        reshape: bool,
        return_class_token: bool,
        norm: bool,
    ) -> Result<Tensor> {
        let outputs = self.get_intermediate_layers_not_chunked(xs, blocks_to_take)?;
        let outputs = if norm {
            outputs
                .iter()
                .map(|out| self.norm.forward(out))
                .collect::<Result<Vec<Tensor>>>()?
        } else {
            outputs
        };

        let class_tokens = outputs
            .iter()
            .map(|out| out.i((.., 0)))
            .collect::<Result<Vec<_>>>()?;
        let outputs = outputs
            .iter()
            .map(|out| out.i((.., 1..)))
            .collect::<Result<Vec<_>>>()?;

        let outputs = if reshape {
            let (b, _c, w, h) = xs.dims4()?;
            let patch_size = self.patch_embed.patch_size.0;
            let num_channels = outputs[0].elem_count() / (b * (w / patch_size) * (h / patch_size));
            outputs
                .iter()
                .map(|out| {
                    out.reshape((b, w / patch_size, h / patch_size, num_channels))?
                        .transpose(2, 3)?
                        .transpose(1, 2)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            outputs
        };

        let outputs = if return_class_token {
            outputs
                .iter()
                .zip(class_tokens.iter())
                .map(|(out, class_token)| Tensor::stack(&[out, class_token], 0))
                .collect::<Result<Vec<_>>>()?
        } else {
            outputs
        };

        Tensor::stack(&outputs[..], 0)
    }
}

impl Module for DinoVisionTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.prepare_tokens_with_mask(xs)?;
        for blk in self.blocks.iter() {
            xs = blk.forward(&xs)?
        }
        let xs = self.norm.forward(&xs)?;
        let xs_norm_clstoken = xs.i((.., 0))?;
        let xs_norm_patchtokens = xs.i((.., 1..))?.mean(1)?;
        let xs = Tensor::cat(&[xs_norm_clstoken, xs_norm_patchtokens], D::Minus1)?;

        match &self.head {
            Some(head) => head.forward(&xs),
            None => Ok(xs),
        }
    }
}

pub fn vit_small(vb: VarBuilder, vb_head: Option<VarBuilder>) -> Result<DinoVisionTransformer> {
    DinoVisionTransformer::new(vb, vb_head, 12, 384, 6)
}

pub fn vit_base(vb: VarBuilder, vb_head: Option<VarBuilder>) -> Result<DinoVisionTransformer> {
    DinoVisionTransformer::new(vb, vb_head, 12, 768, 12)
}

pub fn vit_large(vb: VarBuilder, vb_head: Option<VarBuilder>) -> Result<DinoVisionTransformer> {
    DinoVisionTransformer::new(vb, vb_head, 24, 1024, 16)
}

pub fn vit_giant(vb: VarBuilder, vb_head: Option<VarBuilder>) -> Result<DinoVisionTransformer> {
    DinoVisionTransformer::new(vb, vb_head, 40, 1536, 24)
}
