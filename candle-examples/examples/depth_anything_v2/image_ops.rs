use std::path::PathBuf;

use candle::{Device, Tensor};
use candle::DType::U8;
use candle_transformers::models::depth_anything_v2::PATCH_MULTIPLE;

use crate::color_map::SpectralRColormap;

// taken these from: https://huggingface.co/spaces/depth-anything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py#L207
const MAGIC_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const MAGIC_STD: [f32; 3] = [0.229, 0.224, 0.225];

const LOWER_BOUND: &'static str = "lower_bound";
const UPPER_BOUND: &'static str = "upper_bound";
const MINIMAL: &'static str = "minimal";
const DINO_IMG_SIZE: usize = 518;

pub fn load_and_prep_image(
    image_path: &PathBuf,
    device: &Device,
) -> anyhow::Result<(usize, usize, Tensor)> {
    println!("Loading image");
    let (image, original_height, original_width) = load_image_and_resize(&image_path)?;
    println!("Got image {image:?}, with original size ({original_height}, {original_width})");

    let image = image.to_device(&device)?;

    println!("Normalizing image");
    let image = normalize_image(&image, &MAGIC_MEAN, &MAGIC_STD)?;

    let image = image
        .permute((2, 0, 1))?
        .unsqueeze(0)?;

    Ok((original_height, original_width, image))
}

pub fn load_image_and_resize<P: AsRef<std::path::Path>>(
    p: P,
) -> candle::Result<(Tensor, usize, usize)> {
    let img = image::io::Reader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?;
    let (original_height, original_width) = (img.height() as usize, img.width() as usize);
    let (target_height, target_width) = get_new_size(
        original_height,
        original_width,
        DINO_IMG_SIZE,
        DINO_IMG_SIZE,
        true,
        LOWER_BOUND,
        PATCH_MULTIPLE,
    );

    let img = img.resize_to_fill(
        target_width as u32,
        target_height as u32,
        image::imageops::FilterType::CatmullRom,
    );

    let img = img.into_rgb32f();
    let data = img.into_raw();
    let rgb_data = Tensor::from_vec(data, (target_height, target_width, 3), &Device::Cpu)?;

    let index = Tensor::from_vec(vec![2u32, 1, 0], (3,), &Device::Cpu)?;
    let bgr_data = rgb_data.index_select(&index, 2)?;

    Ok((bgr_data, original_height, original_width))
}

fn normalize_image(image: &Tensor, mean: &[f32; 3], std: &[f32; 3]) -> candle::Result<Tensor> {
    let shape = (1, 1, 3);
    let mean_tensor = Tensor::from_vec(mean.to_vec(), shape, &image.device())?;
    let std_tensor = Tensor::from_vec(std.to_vec(), shape, &image.device())?;
    image
        .broadcast_sub(&mean_tensor)?
        .broadcast_div(&std_tensor)
}

pub fn post_process_image(
    image: &Tensor,
    original_height: usize,
    original_width: usize,
    color_map: bool,
) -> candle::Result<Tensor> {
    let out = image.interpolate2d(original_height, original_width)?;
    let out = scale_image(&out)?;

    let out = if color_map {
        let spectral_r = SpectralRColormap::new();
        spectral_r.gray2color(&out)?
    } else {
        let rgb_slice = [&out, &out, &out];
        Tensor::cat(&rgb_slice, 0)?.squeeze(1)?
    };

    let max_pixel_val = Tensor::try_from(255.0f32)?
        .to_device(out.device())?
        .broadcast_as(out.shape())?;
    let out = (out * max_pixel_val)?;

    out.to_dtype(U8)
}

pub fn get_min_max(t: &Tensor) -> candle::Result<(f32, f32)> {
    let flat_values: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let min_val = *flat_values.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let max_val = *flat_values.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

    Ok((min_val, max_val))
}

fn scale_image(depth: &Tensor) -> candle::Result<Tensor> {
    let (min_val, max_val) = get_min_max(&depth)?;

    let min_val_tensor = Tensor::try_from(min_val)?
        .to_device(depth.device())?
        .broadcast_as(depth.shape())?;
    let depth = (depth - min_val_tensor)?;

    let range = max_val - min_val;
    let range_tensor = Tensor::try_from(range)?
        .to_device(depth.device())?
        .broadcast_as(depth.shape())?;

    depth / range_tensor
}

fn constrain_to_multiple_of(
    value: f32,
    multiple_of: usize,
    min_val: f32,
    max_val: Option<f32>,
) -> usize {
    let mut constrained_value = (value / multiple_of as f32).round() * multiple_of as f32;

    if let Some(max_val) = max_val {
        if constrained_value > max_val {
            constrained_value = (value / multiple_of as f32).floor() * multiple_of as f32;
        }
    }

    if constrained_value < min_val {
        constrained_value = (value / multiple_of as f32).ceil() * multiple_of as f32;
    }

    constrained_value as usize
}

fn get_new_size(
    height: usize,
    width: usize,
    target_height: usize,
    target_width: usize,
    keep_aspect_ratio: bool,
    resize_method: &str,
    multiple_of: usize,
) -> (usize, usize) {
    let scale_width = target_width as f32 / width as f32;
    let scale_height = target_height as f32 / height as f32;

    let (new_width, new_height) = if keep_aspect_ratio {
        match resize_method {
            LOWER_BOUND => {
                if scale_width > scale_height {
                    (target_width, (scale_width * height as f32).round() as usize)
                } else {
                    (
                        (scale_height * width as f32).round() as usize,
                        target_height,
                    )
                }
            }
            UPPER_BOUND => {
                if scale_width < scale_height {
                    (target_width, (scale_width * height as f32).round() as usize)
                } else {
                    (
                        (scale_height * width as f32).round() as usize,
                        target_height,
                    )
                }
            }
            MINIMAL => {
                if (1.0 - scale_width).abs() < (1.0 - scale_height).abs() {
                    (target_width, (scale_width * height as f32).round() as usize)
                } else {
                    (
                        (scale_height * width as f32).round() as usize,
                        target_height,
                    )
                }
            }
            _ => panic!("resize_method {} not implemented", resize_method),
        }
    } else {
        (target_width, target_height)
    };

    let new_height =
        constrain_to_multiple_of(new_height as f32, multiple_of, target_height as f32, None);
    let new_width =
        constrain_to_multiple_of(new_width as f32, multiple_of, target_width as f32, None);

    (new_height, new_width) // switching height and width because the tensor is also height first
}
