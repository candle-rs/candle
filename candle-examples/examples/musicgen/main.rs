#![allow(dead_code)]
// https://huggingface.co/facebook/musicgen-small/tree/main
// https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/models/musicgen/modeling_musicgen.py
// TODO: Add an offline mode.
// TODO: Add a KV cache.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod musicgen_model;

use std::borrow::BorrowMut;

use musicgen_model::{GenConfig, MusicgenForConditionalGeneration};

use anyhow::{Error as E, Result};
use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};

const DTYPE: DType = DType::F32;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The model weight file, in safetensor format.
    #[arg(long)]
    model: Option<String>,

    /// The tokenizer config.
    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(
        long,
        default_value = "90s rock song with loud guitars and heavy drums."
    )]
    prompt: String,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let tokenizer = match args.tokenizer {
        Some(tokenizer) => std::path::PathBuf::from(tokenizer),
        None => Api::new()?
            .model("facebook/musicgen-small".to_string())
            .get("tokenizer.json")?,
    };
    let mut tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;

    let model = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => Api::new()?
            .repo(Repo::with_revision(
                "facebook/musicgen-small".to_string(),
                RepoType::Model,
                "refs/pr/13".to_string(),
            ))
            .get("model.safetensors")?,
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DTYPE, &device)? };
    let config = GenConfig::small();
    let mut model = MusicgenForConditionalGeneration::load(vb, config)?;

    let tokens = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    println!("tokens: {tokens:?}");
    let tokens = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
    println!("{tokens:?}");

    let token_type_ids = tokens.ones_like()?;
    println!("{token_type_ids}");

    //let embeds = model.text_encoder.forward(&tokens)?;

    //println!("{embeds}");
    let musicgen_config = musicgen_model::GenConfig::small().musicgen;
    let pad_token_id = musicgen_config.pad_token_id as f64;
    let num_codebooks = musicgen_config.num_codebooks;
    let decoder_input_ids =
        (Tensor::ones((num_codebooks, 1), DType::I64, &device)? * pad_token_id)?;
    let output = model.forward(&tokens, &decoder_input_ids, Some(&token_type_ids))?;
    println!("{output}");
    //let audio_encoder_outputs = model.audio_encoder.encode(&embeds)?;
    //println!("{audio_encoder_outputs}");

    Ok(())
}
