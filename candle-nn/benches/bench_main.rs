mod benchmarks;

use criterion::criterion_main;
criterion_main!(
    benchmarks::softmax::benches,
    benchmarks::layer_norm::benches,
    benchmarks::conv::benches
);
