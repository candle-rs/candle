use std::fs;
use std::path::{Path, PathBuf};

use image;
use image::DynamicImage;

use candle::{Device, DType, Error, Result, Tensor};

pub struct Dataset {
    pub dataset: crate::vision::Dataset,
    pub idx_to_classes: Vec<String>,
}

pub fn load<P: AsRef<Path>, F>(path: P, transformer: F, test_data_ratio: Option<f32>) -> Result<Dataset>
    where P: AsRef<Path>, F: Fn(DynamicImage) -> Result<Tensor>,
{
    let dir = fs::read_dir(path).map_err(|err| Error::Msg(format!("IO error: {err}")))?;
    let mut grouped: Vec<(String, Vec<PathBuf>)> = Vec::new();
    for dir_entry in dir {
        let dir_entry = dir_entry.map_err(|err| Error::Msg(format!("IO error: {err}")))?;
        let metadata = dir_entry.metadata().map_err(|err| Error::Msg(format!("IO error: {err}")))?;
        if metadata.is_dir() {
            let label = dir_entry.file_name().into_string()
                .map_err(|err| Error::Msg(format!("Parse label failed, path: {:?}, error: {:?}", dir_entry.path(), err)))?;
            let mut img_paths = Vec::new();
            for dir_entry in fs::read_dir(dir_entry.path()).map_err(|err| Error::Msg(format!("IO error: {err}")))? {
                let dir_entry = dir_entry.map_err(|err| Error::Msg(format!("IO error: {err}")))?;
                let metadata = dir_entry.metadata().map_err(|err| Error::Msg(format!("IO error: {err}")))?;
                if metadata.is_file() {
                    img_paths.push(dir_entry.path());
                }
            }
            grouped.push((label, img_paths));
        }
    }
    grouped.sort_by(|(l, _), (r, _)| l.cmp(r));
    let mut idx_to_classes = Vec::new();
    let mut train_idx_to_img = Vec::new();
    let mut test_idx_to_img = Vec::new();
    for (idx, (label, img_paths)) in grouped.into_iter().enumerate() {
        let img_len = img_paths.len();
        let split_idx = ((img_len as f32) * (1f32 - test_data_ratio.unwrap_or(0f32))) as usize;
        for (i, img_path) in img_paths.iter().enumerate() {
            let img_file = fs::read(img_path).map_err(|err| Error::Msg(format!("IO error: {err}")))?;
            let img = image::load_from_memory(&img_file).map_err(|err| Error::Msg(format!("Image error: {err}")))?;
            let img = transformer(img)?.unsqueeze(0)?;
            if i < split_idx {
                train_idx_to_img.push((idx, img));
            } else {
                test_idx_to_img.push((idx, img))
            }
        }
        idx_to_classes.push(label);
    }
    let (train_images, train_labels) = to_tensor(train_idx_to_img)?;
    let (test_images, test_labels) = to_tensor(test_idx_to_img)?;
    let dataset = crate::vision::Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: idx_to_classes.len(),
    };
    Ok(Dataset { dataset, idx_to_classes })
}

#[inline]
fn to_tensor(idx_to_img: Vec<(usize, Tensor)>) -> Result<(Tensor, Tensor)> {
    let labels = Tensor::from_vec(
        idx_to_img.iter().map(|(idx, _)| *idx as u32).collect::<Vec<_>>(),
        idx_to_img.len(),
        &Device::Cpu,
    )?;
    let imgs = Tensor::cat(
        &idx_to_img.into_iter()
            .map(|(_, tensor)| tensor)
            .collect::<Vec<_>>(), 0,
    )?
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?;
    Ok((imgs, labels))
}
