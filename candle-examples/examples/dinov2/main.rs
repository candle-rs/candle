//! DINOv2: Learning Robust Visual Features without Supervision
//! https://github.com/facebookresearch/dinov2

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use clap::Parser;

use candle::DType::F32;
use candle::{DType, IndexOp, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::dinov2;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    head: Option<String>,

    #[arg(long)]
    image: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;

    let image = candle_examples::imagenet::load_image224(args.image)?.to_device(&device)?;
    println!("loaded image {image:?}");

    let dinov2_model_file = match args.model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("facebook/dinov2-small".into());
            api.get("model.safetensors")?
        }
        Some(dinov2_model) => dinov2_model.into(),
    };
    println!("Using Dinov2 file {:?}", dinov2_model_file);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dinov2_model_file], F32, &device)? };

    let dinov2_head_file = match args.head {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("jeroenvlek/dinov2-linear-heads-safetensors".into());
            api.get("dinov2_vits14_linear_head.safetensors")?
        }
        Some(dinov2_head) => dinov2_head.into(),
    };
    println!("Using Dinov2 head file {:?}", dinov2_head_file);

    let vb_head =
        unsafe { VarBuilder::from_mmaped_safetensors(&[dinov2_head_file], F32, &device)? };

    let model = dinov2::vit_small(vb, Some(vb_head))?;
    println!("DinoV2 model built");

    let logits = model.forward(&image.unsqueeze(0)?)?;
    let prs = candle_nn::ops::softmax(&logits, D::Minus1)?
        .i(0)?
        .to_vec1::<f32>()?;
    let mut prs = prs.iter().enumerate().collect::<Vec<_>>();
    prs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for &(category_idx, pr) in prs.iter().take(5) {
        println!(
            "{:24}: {:.2}%",
            candle_examples::imagenet::CLASSES[category_idx],
            100. * pr
        );
    }
    Ok(())
}
