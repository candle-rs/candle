use anyhow::{Ok, Result};
use candle::{DType, Tensor};

use candle_transformers::models::flux;
use candle_transformers::models::mmdit::model::MMDiT; // for the get_noise function

pub fn euler_sample(
    mmdit: &MMDiT,
    y: &Tensor,
    context: &Tensor,
    num_inference_steps: usize,
    height: usize,
    width: usize,
) -> Result<Tensor> {
    let mut x = flux::sampling::get_noise(1, height, width, &y.device())?.to_dtype(DType::F16)?;
    let sigmas = (0..=num_inference_steps)
        .map(|x| x as f64 / num_inference_steps as f64)
        .rev()
        .map(|x| time_snr_shift(3.0, x as f64))
        .collect::<Vec<f64>>();

    for window in sigmas.windows(2) {
        let (t_curr, t_prev) = match window {
            [a, b] => (a, b),
            _ => continue,
        };
        let noise_pred = mmdit.forward(
            &Tensor::cat(&[x.clone(), x.clone()], 0)?,
            &Tensor::full((*t_curr) * 1000.0, (2,), &x.device())?.contiguous()?,
            &y,
            &context,
        )?;
        let cfg_scale = 4.0;
        x = (x + (apply_cfg(cfg_scale, &noise_pred)? * (*t_prev as f64 - *t_curr as f64))?)?;
    }
    Ok(x)
}

// The "Resolution-dependent shifting of timestep schedules" recommended in the SD3 tech report paper
// https://arxiv.org/pdf/2403.03206
// Following the implementation in ComfyUI:
// https://github.com/comfyanonymous/ComfyUI/blob/3c60ecd7a83da43d694e26a77ca6b93106891251/
// comfy/model_sampling.py#L181
fn time_snr_shift(alpha: f64, t: f64) -> f64 {
    alpha * t / (1.0 + (alpha - 1.0) * t)
}

fn apply_cfg(cfg_scale: f64, noise_pred: &Tensor) -> Result<Tensor> {
    Ok(((cfg_scale * noise_pred.narrow(0, 0, 1)?)?
        - ((cfg_scale - 1.0) * noise_pred.narrow(0, 1, 1)?)?)?)
}
