use std::path::PathBuf;

use clap::{Parser, ValueEnum};

#[derive(ValueEnum, Clone, Debug)]
pub enum ModelSize {
    S,
    B,
    L,
    G,
}

#[derive(Parser)]
pub struct Args {
    #[arg(long)]
    pub dinov2_model: Option<PathBuf>,

    #[arg(long)]
    pub dinov2_head: Option<PathBuf>,

    #[arg(long)]
    pub depth_anything_v2_model: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = ModelSize::B)]
    pub size: ModelSize,

    #[arg(long)]
    pub image: PathBuf,

    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    #[arg(long)]
    pub cpu: bool,

    #[arg(long)]
    pub color_map: bool,
}
