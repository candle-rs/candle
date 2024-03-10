use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn run(shape: (usize, usize, usize), dtype: DType, device: &Device) {
    Tensor::ones(shape, dtype, device).unwrap();
}

fn run_fill_benchmark(c: &mut Criterion, device: &Device, name: &str, dtype: DType) {
    let b = 1;
    let rows = 1024;
    let columns = 1024;

    let flops = b * rows * columns * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |bencher| {
        bencher.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(
                    black_box((b, rows, columns)),
                    black_box(dtype),
                    black_box(&device),
                );
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        run_fill_benchmark(c, &device, "fill_u8", DType::U8);
        run_fill_benchmark(c, &device, "fill_f32", DType::F32);
    }
}

criterion_group!(benches, criterion_benchmark);
