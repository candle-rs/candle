mod benchmarks;

use criterion::criterion_main;
criterion_main!(benchmarks::matmul::benches, benchmarks::where_cond::benches);
