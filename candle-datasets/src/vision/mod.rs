use candle::Tensor;

pub struct Dataset {
    pub train_images: Tensor,
    pub train_labels: Tensor,
    pub test_images: Tensor,
    pub test_labels: Tensor,
    pub labels: usize,
}

pub mod cifar;
pub mod mnist;
pub mod img_folder;
