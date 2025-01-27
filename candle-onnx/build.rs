fn main() -> anyhow::Result<()> {
    let fds = protox::compile(&["src/onnx.proto3"], &["src/"])?;
    prost_build::compile_fds(fds)?;
    Ok(())
}
