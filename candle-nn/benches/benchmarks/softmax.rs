use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Tensor};
use candle_nn::ops::softmax_last_dim;
use criterion::Throughput;
use criterion::{black_box, criterion_group, Criterion};
use std::time::Instant;

fn run(input: &Tensor) {
    let _ = softmax_last_dim(&input).unwrap();
}

const B: usize = 1;
const M: usize = 1024;
const K: usize = 1024;

fn run_softmax_benchmark(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let elements = B * M * K;

    let input = Tensor::rand(-1000.0f32, 1000.0f32, (B, M, K), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let flops = elements * dtype.size_in_bytes();
    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box(&input));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = BenchDeviceHandler::new().unwrap();
    for d in device.devices {
        run_softmax_benchmark(c, &d, DType::F32, "softmax_f32");
        run_softmax_benchmark(c, &d, DType::BF16, "softmax_bf16");
        run_softmax_benchmark(c, &d, DType::F16, "softmax_f16");
    }
}

criterion_group!(benches, criterion_benchmark);
