# DeepSeek V2

DeepSeek V2 an MoE model featuring MLA (Multi-Latent Attention). There is a lite (16B) and a full (236B) model.

- Context length of **32k tokens** (Lite model), **128k tokens** (full model)
- 64 routed experts (Lite model), 160 routed experts (full model)

## Running the example

```bash
$ cargo run --example deepseekv2 --release --features cuda -- --prompt Hello --sample-len 150

Generated text:
Write helloworld code in Rust
=============================

This is a simple example of how to write "Hello, world!" program in Rust.

## Compile and run

``bash
$ cargo build --release
   Compiling hello-world v0.1.0 (/home/user/rust/hello-world)
    Finished release [optimized] target(s) in 0.26s
$ ./target/release/hello-world
Hello, world!
``

## Source code

``rust
fn main() {
    println!("Hello, world!");
}
``

## License

This example is released under the terms
```
