// TODO: Add an offline mode.

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use candle_transformers::models::falcon::{Config, Falcon};

struct TextGeneration {
    model: Falcon,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

struct GenerationOptions {
    temp: Option<f64>,
    top_p: Option<f64>,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    fn new(
        model: Falcon,
        tokenizer: Tokenizer,
        generation_options: GenerationOptions,
        seed: u64,
        device: &Device,
    ) -> Self {
        let logits_processor =
            LogitsProcessor::new(seed, generation_options.temp, generation_options.top_p);
        let repeat_penalty = generation_options.repeat_penalty;
        let repeat_last_n = generation_options.repeat_last_n;
        Self {
            model,
            tokenizer,
            logits_processor,
            device: device.clone(),
            repeat_penalty,
            repeat_last_n,
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        println!("starting the inference loop");
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let mut new_tokens = vec![];
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let start_gen = std::time::Instant::now();
            let context_size = if self.model.config().use_cache && index > 0 {
                1
            } else {
                tokens.len()
            };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            new_tokens.push(next_token);
            println!("> {:?}", start_gen.elapsed());
            println!(
                "{} token: {} '{}'",
                index + 1,
                next_token,
                self.tokenizer.decode(&[next_token], true).map_err(E::msg)?
            );
        }
        let dt = start_gen.elapsed();
        println!(
            "{sample_len} tokens generated ({} token/s)\n----\n{}\n----",
            sample_len as f64 / dt.as_secs_f64(),
            self.tokenizer.decode(&new_tokens, true).map_err(E::msg)?
        );
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    prompt: String,

    /// Use f32 computations rather than bf16.
    #[arg(long)]
    use_f32: bool,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    #[arg(long, default_value = "tiiuae/falcon-7b")]
    model_id: String,

    #[arg(long, default_value = "refs/pr/43")]
    revision: String,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = candle_examples::device(args.cpu)?;
    let start = std::time::Instant::now();
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let filenames = candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?;
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let dtype = if args.use_f32 {
        DType::F32
    } else if device.support_native_bf16()? {
        DType::BF16
    } else {
        DType::F16
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let config = Config::falcon7b();
    config.validate()?;
    let model = Falcon::load(vb, config)?;
    println!("loaded the model in {:?}", start.elapsed());

    let generation_options = GenerationOptions {
        temp: args.temperature,
        top_p: args.top_p,
        repeat_penalty: args.repeat_penalty,
        repeat_last_n: args.repeat_last_n,
    };
    let mut pipeline =
        TextGeneration::new(model, tokenizer, generation_options, args.seed, &device);
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}
