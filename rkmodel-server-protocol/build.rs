fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../proto/rkmodel.proto");
    tonic_build::configure().compile_protos(&["../proto/rkmodel.proto"], &["../proto"])?;
    Ok(())
}
