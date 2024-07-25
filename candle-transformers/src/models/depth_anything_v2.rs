use std::cell::RefCell;

use candle::{Module, Result, Tensor};
use candle::D::Minus1;
use candle_nn::{
    Activation, batch_norm, BatchNorm, BatchNormConfig, conv2d, Conv2d, conv2d_no_bias, Conv2dConfig,
    conv_transpose2d, ConvTranspose2dConfig, linear, seq, Sequential, VarBuilder,
};
use candle_nn::ops::Identity;

use crate::models::dinov2::DinoVisionTransformer;

pub const PATCH_MULTIPLE: usize = 14;

#[derive(Clone)]
pub struct DepthAnythingV2Config {
    out_channel_sizes: [usize; 4],
    in_channel_size: usize, // embed_dim in the Dino model
    num_features: usize,
    use_batch_norm: bool,
    use_class_token: bool,
    layer_ids_vits: Vec<usize>,
}

impl DepthAnythingV2Config {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        out_channel_sizes: [usize; 4],
        in_channel_size: usize,
        num_features: usize,
        use_batch_norm: bool,
        use_class_token: bool,
        layer_ids_vits: Vec<usize>,
    ) -> Self {
        Self {
            out_channel_sizes,
            in_channel_size,
            num_features,
            use_batch_norm,
            use_class_token,
            layer_ids_vits,
        }
    }

    pub fn vit_small() -> Self {
        Self {
            out_channel_sizes: [48, 96, 192, 384],
            in_channel_size: 384,
            num_features: 64,
            use_batch_norm: false,
            use_class_token: false,
            layer_ids_vits: vec![2, 5, 8, 11],
        }
    }

    pub fn vit_base() -> Self {
        Self {
            out_channel_sizes: [96, 192, 384, 768],
            in_channel_size: 768,
            num_features: 128,
            use_batch_norm: false,
            use_class_token: false,
            layer_ids_vits: vec![2, 5, 8, 11],
        }
    }

    pub fn vit_large() -> Self {
        Self {
            out_channel_sizes: [256, 512, 1024, 1024],
            in_channel_size: 1024,
            num_features: 256,
            use_batch_norm: false,
            use_class_token: false,
            layer_ids_vits: vec![4, 11, 17, 23],
        }
    }

    pub fn vit_giant() -> Self {
        Self {
            out_channel_sizes: [1536, 1536, 1536, 1536],
            in_channel_size: 1536,
            num_features: 384,
            use_batch_norm: false,
            use_class_token: false,
            layer_ids_vits: vec![9, 19, 29, 39],
        }
    }
}

pub struct ResidualConvUnit {
    activation: Activation,
    conv1: Conv2d,
    conv2: Conv2d,
    batch_norm1: Option<BatchNorm>,
    batch_norm2: Option<BatchNorm>,
}

impl ResidualConvUnit {
    pub fn new(
        conf: DepthAnythingV2Config,
        activation: Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = conv2d(
            conf.num_features,
            conf.num_features,
            3,
            conv_cfg,
            vb.pp("conv1"),
        )?;
        let conv2 = conv2d(
            conf.num_features,
            conf.num_features,
            3,
            conv_cfg,
            vb.pp("conv2"),
        )?;

        let (batch_norm1, batch_norm2) = match conf.use_batch_norm {
            true => {
                let batch_norm_cfg = BatchNormConfig {
                    eps: 1e-05,
                    remove_mean: false,
                    affine: true,
                    momentum: 0.1,
                };
                (
                    Some(batch_norm(conf.num_features, batch_norm_cfg, vb.pp("bn1"))?),
                    Some(batch_norm(conf.num_features, batch_norm_cfg, vb.pp("bn2"))?),
                )
            }
            false => (None, None),
        };

        Ok(Self {
            activation,
            conv1,
            conv2,
            batch_norm1,
            batch_norm2,
        })
    }
}

impl Module for ResidualConvUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.activation.forward(xs)?;
        let out = self.conv1.forward(&out)?;
        let out = if let Some(batch_norm1) = &self.batch_norm1 {
            batch_norm1.forward_train(&out)?
        } else {
            out
        };

        let out = self.activation.forward(&out)?;
        let out = self.conv2.forward(&out)?;
        let out = if let Some(batch_norm2) = &self.batch_norm2 {
            batch_norm2.forward_train(&out)?
        } else {
            out
        };

        let out = out.add(xs)?;

        Ok(out)
    }
}

pub struct FeatureFusionBlock {
    res_conv_unit1: ResidualConvUnit,
    res_conv_unit2: ResidualConvUnit,
    output_conv: Conv2d,
    output_size: RefCell<Option<(usize, usize)>>,
    skip_add: RefCell<Option<Tensor>>,
}

impl FeatureFusionBlock {
    pub fn new(
        conf: DepthAnythingV2Config,
        activation: Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        let output_conv = conv2d(
            conf.num_features,
            conf.num_features,
            1,
            Default::default(),
            vb.pp("out_conv"),
        )?;
        let res_conv_unit1 =
            ResidualConvUnit::new(conf.clone(), activation, vb.pp("resConfUnit1"))?;
        let res_conv_unit2 =
            ResidualConvUnit::new(conf.clone(), activation, vb.pp("resConfUnit2"))?;

        Ok(Self {
            res_conv_unit1,
            res_conv_unit2,
            output_conv,
            output_size: RefCell::new(None),
            skip_add: RefCell::new(None),
        })
    }

    fn set_output_size(&self, output_height: usize, output_width: usize) {
        *self.output_size.borrow_mut() = Some((output_height, output_width));
    }

    fn set_skip_add(&self, skip_add: &Tensor) {
        *self.skip_add.borrow_mut() = Some(skip_add.clone())
    }
}

impl Module for FeatureFusionBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = if let Some(ref skip_add) = *self.skip_add.borrow() {
            let res = self.res_conv_unit1.forward(skip_add)?;
            &xs.add(&res)?
        } else {
            xs
        };

        let out = self.res_conv_unit2.forward(out)?;
        let (target_height, target_width) = if let Some(size) = *self.output_size.borrow() {
            size
        } else {
            let (_, _, h, w) = out.dims4()?;
            (h * 2, w * 2)
        };

        let out = out.interpolate2d(target_height, target_width)?;

        let out = self.output_conv.forward(&out)?;

        Ok(out)
    }
}

pub struct Scratch {
    layer1_rn: Conv2d,
    layer2_rn: Conv2d,
    layer3_rn: Conv2d,
    layer4_rn: Conv2d,
    refine_net1: FeatureFusionBlock,
    refine_net2: FeatureFusionBlock,
    refine_net3: FeatureFusionBlock,
    refine_net4: FeatureFusionBlock,
    output_conv1: Conv2d,
    output_conv2: Sequential,
}

impl Scratch {
    pub fn new(conf: DepthAnythingV2Config, vb: VarBuilder) -> Result<Self> {
        const KERNEL_SIZE: usize = 3;
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };

        let layer1_rn = conv2d_no_bias(
            conf.out_channel_sizes[0],
            conf.num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("layer1_rn"),
        )?;
        let layer2_rn = conv2d_no_bias(
            conf.out_channel_sizes[1],
            conf.num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("layer2_rn"),
        )?;
        let layer3_rn = conv2d_no_bias(
            conf.out_channel_sizes[2],
            conf.num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("layer3_rn"),
        )?;
        let layer4_rn = conv2d_no_bias(
            conf.out_channel_sizes[3],
            conf.num_features,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("layer4_rn"),
        )?;

        let refine_net1 =
            FeatureFusionBlock::new(conf.clone(), Activation::Relu, vb.pp("refinenet1"))?;
        let refine_net2 =
            FeatureFusionBlock::new(conf.clone(), Activation::Relu, vb.pp("refinenet2"))?;
        let refine_net3 =
            FeatureFusionBlock::new(conf.clone(), Activation::Relu, vb.pp("refinenet3"))?;
        let refine_net4 =
            FeatureFusionBlock::new(conf.clone(), Activation::Relu, vb.pp("refinenet4"))?;

        let output_conv1 = conv2d(
            conf.num_features,
            conf.num_features / 2,
            KERNEL_SIZE,
            conv_cfg,
            vb.pp("output_conv1"),
        )?;

        let output_conv2 = seq();
        const HEAD_FEATURES_2: usize = 32;
        let output_conv2 = output_conv2.add(conv2d(
            conf.num_features / 2,
            HEAD_FEATURES_2,
            3,
            conv_cfg,
            vb.pp("output_conv2").pp("0"),
        )?);
        let output_conv2 = output_conv2
            .add(Activation::Relu)
            .add(conv2d(
                HEAD_FEATURES_2,
                1,
                1,
                Default::default(),
                vb.pp("output_conv2").pp("2"),
            )?)
            .add(Activation::Relu);

        Ok(Self {
            layer1_rn,
            layer2_rn,
            layer3_rn,
            layer4_rn,
            refine_net1,
            refine_net2,
            refine_net3,
            refine_net4,
            output_conv1,
            output_conv2,
        })
    }
}

pub struct DPTHead {
    conf: DepthAnythingV2Config,
    projections: Vec<Conv2d>,
    resize_layers: Vec<Box<dyn Module>>,
    readout_projections: Vec<Sequential>,
    scratch: Scratch,
    image_size: Option<(usize, usize)>,
    patch_size: Option<(usize, usize)>,
}

impl DPTHead {
    pub fn new(conf: DepthAnythingV2Config, vb: VarBuilder) -> Result<Self> {
        let mut projections: Vec<Conv2d> = Vec::with_capacity(conf.out_channel_sizes.len());
        for (conv_index, out_channel_size) in conf.out_channel_sizes.iter().enumerate() {
            projections.push(conv2d(
                conf.in_channel_size,
                *out_channel_size,
                1,
                Default::default(),
                vb.pp("projects").pp(conv_index.to_string()),
            )?);
        }

        let resize_layers: Vec<Box<dyn Module>> = vec![
            Box::new(conv_transpose2d(
                conf.out_channel_sizes[0],
                conf.out_channel_sizes[0],
                4,
                ConvTranspose2dConfig {
                    stride: 4,
                    ..Default::default()
                },
                vb.pp("resize_layers").pp("0"),
            )?),
            Box::new(conv_transpose2d(
                conf.out_channel_sizes[1],
                conf.out_channel_sizes[1],
                2,
                ConvTranspose2dConfig {
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("resize_layers").pp("1"),
            )?),
            Box::new(Identity::new()),
            Box::new(conv2d(
                conf.out_channel_sizes[3],
                conf.out_channel_sizes[3],
                3,
                Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("resize_layers").pp("3"),
            )?),
        ];

        let readout_projections = if conf.use_class_token {
            let rop = Vec::with_capacity(conf.layer_ids_vits.len());
            for rop_index in 0..conf.layer_ids_vits.len() {
                seq()
                    .add(linear(
                        2 * conf.in_channel_size,
                        conf.in_channel_size,
                        vb.pp("readout_projects").pp(rop_index.to_string()),
                    )?)
                    .add(Activation::Gelu);
            }
            rop
        } else {
            vec![]
        };

        let scratch = Scratch::new(conf.clone(), vb.pp("scratch"))?;

        Ok(Self {
            conf,
            projections,
            resize_layers,
            readout_projections,
            scratch,
            image_size: None,
            patch_size: None,
        })
    }

    fn set_image_and_patch_size(&mut self, image_height: usize, image_width: usize) {
        self.image_size = Some((image_height, image_width));
        let patch_height = image_height / PATCH_MULTIPLE;
        let patch_width = image_width / PATCH_MULTIPLE;
        self.patch_size = Some((patch_height, patch_width));
    }
}

impl Module for DPTHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut out: Vec<Tensor> = Vec::with_capacity(self.conf.layer_ids_vits.len());
        for i in 0..self.conf.layer_ids_vits.len() {
            let x = if self.conf.use_class_token {
                let x = xs.get(i)?.get(0)?;
                let class_token = xs.get(i)?.get(1)?;
                let readout = class_token.unsqueeze(1)?.expand(x.shape())?;
                let to_cat = [x, readout];
                let cat = Tensor::cat(&to_cat, Minus1)?;
                self.readout_projections[i].forward(&cat)?
            } else {
                xs.get(i)?
            };

            let x_dims = x.dims();
            let x = x.permute((0, 2, 1))?.reshape((
                x_dims[0],
                x_dims[x_dims.len() - 1],
                self.patch_size.unwrap().0,
                self.patch_size.unwrap().1,
            ))?;

            let x = self.projections[i].forward(&x)?;

            let x = self.resize_layers[i].forward(&x)?;
            out.push(x);
        }

        let layer_1_rn = self.scratch.layer1_rn.forward(&out[0])?;
        let layer_2_rn = self.scratch.layer2_rn.forward(&out[1])?;
        let layer_3_rn = self.scratch.layer3_rn.forward(&out[2])?;
        let layer_4_rn = self.scratch.layer4_rn.forward(&out[3])?;

        let (_, _, output_height, output_width) = layer_3_rn.dims4()?;
        self.scratch
            .refine_net4
            .set_output_size(output_height, output_width);
        let path4 = self.scratch.refine_net4.forward(&layer_4_rn)?;

        let (_, _, output_height, output_width) = layer_2_rn.dims4()?;
        self.scratch
            .refine_net3
            .set_output_size(output_height, output_width);
        self.scratch.refine_net3.set_skip_add(&layer_3_rn);
        let path3 = self.scratch.refine_net3.forward(&path4)?;

        let (_, _, output_height, output_width) = layer_1_rn.dims4()?;
        self.scratch
            .refine_net2
            .set_output_size(output_height, output_width);
        self.scratch.refine_net2.set_skip_add(&layer_2_rn);
        let path2 = self.scratch.refine_net2.forward(&path3)?;

        self.scratch.refine_net1.set_skip_add(&layer_1_rn);
        let path1 = self.scratch.refine_net1.forward(&path2)?;

        let out = self.scratch.output_conv1.forward(&path1)?;

        let out = out.interpolate2d(self.image_size.unwrap().0, self.image_size.unwrap().1)?;
        let out = self.scratch.output_conv2.forward(&out)?;

        Ok(out)
    }
}

pub struct DepthAnythingV2 {
    pretrained: DinoVisionTransformer,
    depth_head: DPTHead,
    conf: DepthAnythingV2Config,
}

impl<'a> DepthAnythingV2 {
    pub fn new(
        pretrained: DinoVisionTransformer,
        conf: DepthAnythingV2Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let depth_head = DPTHead::new(conf.clone(), vb.pp("depth_head"))?;

        Ok(Self {
            pretrained,
            depth_head,
            conf,
        })
    }

    pub fn set_image_and_patch_size(&mut self, image_height: usize, image_width: usize) {
        self.depth_head
            .set_image_and_patch_size(image_height, image_width);
    }
}

impl Module for DepthAnythingV2 {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let features = self.pretrained.get_intermediate_layers(
            xs,
            &self.conf.layer_ids_vits,
            false,
            self.conf.use_class_token,
            true,
        )?;

        let depth = self.depth_head.forward(&features)?;

        depth.relu()
    }
}
