#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::Module;
use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::swin_transformer::{SwinConfig, SwinForImageClassification};
use clap::Parser;
use hf_hub::api::sync::Api;

#[derive(Parser, Debug)]
#[clap(author, version, about = "Swin Transformer", long_about = None)]
struct Args {
    #[arg(help = "Image to be processed.")]
    image: String,
    #[arg(
        long,
        default_value = "microsoft/swin-tiny-patch4-window7-224",
        help = "Model repo. See all Swin models at https://huggingface.co/models?filter=swin"
    )]
    model_repo: String,
    #[arg(long, default_value_t = false, help = "Run on CPU rather than on GPU.")]
    cpu: bool,
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let repo = Api::new()?.repo(hf_hub::Repo::new(args.model_repo, hf_hub::RepoType::Model));

    let config: SwinConfig = {
        let path = repo.get("config.json")?;
        let config = std::fs::read_to_string(path)?;
        serde_json::from_str(&config)?
    };
    println!("config: {:?}", config);
    let model = repo.get("model.safetensors")?;
    println!("model: {:?}", model);
    let vb = { unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? } };
    let model = SwinForImageClassification::new(&config, 1000, vb)?;
    println!("model {:?}", model);
    let tensor = Tensor::zeros((1, 3, 224, 224), DType::F32, &device)?;
    let output = model.forward(&tensor)?;
    println!("output {:?}", output);
    Ok(())
}
