# candle-dinov2

[Depth Anything V2] is a model for Monocular Depth Estimation (MDE, i.e. just using a single image) which
builds on the [DINOv2](https://github.com/facebookresearch/dinov2) vision transformer.

This example first instantiates the DINOv2 model and then proceeds to create DepthAnythingV2 and run it.

## Running an example with color map and CUDA

```bash
cargo run --features cuda,depth_anything_v2 --package candle-examples --example depth_anything_v2 -- --size s --color-map --image candle-examples/examples/yolo-v8/assets/bike.jpg 
```
