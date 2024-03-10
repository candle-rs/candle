use half::{bf16, f16};
use metal::{
    Buffer, CommandBufferRef, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState,
    Device, Function, FunctionConstantValues, Library, MTLDataType, MTLSize, NSUInteger,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::RwLock;

const AFFINE: &str = include_str!("affine.metal");
const INDEXING: &str = include_str!("indexing.metal");
const UNARY: &str = include_str!("unary.metal");
const BINARY: &str = include_str!("binary.metal");
const TERNARY: &str = include_str!("ternary.metal");
const CAST: &str = include_str!("cast.metal");
const FILL: &str = include_str!("fill.metal");
const CONV: &str = include_str!("conv.metal");
const REDUCE: &str = include_str!("reduce.metal");
const RANDOM: &str = include_str!("random.metal");
const MFA: &[u8] = include_bytes!("libMetalFlashAttention.metallib");
const QUANTIZED: &str = include_str!("quantized.metal");

/// Most kernels apply similarly across the tensors
/// This creates a strategy that uses the maximum amount of threads per threadgroup (capped at the
/// actual total buffer length).
/// Then kernels can just do their op on their single point in the buffer.
fn linear_split(pipeline: &ComputePipelineState, length: usize) -> (MTLSize, MTLSize) {
    let size = length as u64;
    let width = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), size);
    let count = (size + width - 1) / width;
    let thread_group_count = MTLSize {
        width: count,
        height: 1,
        depth: 1,
    };

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    (thread_group_count, thread_group_size)
}

fn set_param<P: EncoderParam>(encoder: &ComputeCommandEncoderRef, position: u64, data: P) {
    <P as EncoderParam>::set_param(encoder, position, data)
}

/// Helper functions to create the various objects on the compute command encoder
/// on a single line.
/// Prevents getting wrong some arguments number and mixing length and size in bytes.
pub trait EncoderParam: private::Sealed {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self);
}

macro_rules! primitives {
    ($($type:ty),+) => {
        $(
            impl EncoderParam for $type {
                fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
                    encoder.set_bytes(
                        position,
                        core::mem::size_of::<$type>() as u64,
                        &data as *const $type as *const c_void,
                    );
                }
            }
        )+
    };
}
primitives!(bool, usize, u8, u32, u64, i32, i64, f16, bf16, f32);

impl<T> EncoderParam for &[T] {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_bytes(
            position,
            core::mem::size_of_val(data) as u64,
            data.as_ptr() as *const c_void,
        );
    }
}

impl EncoderParam for &Buffer {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data), 0);
    }
}
impl EncoderParam for (&Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data.0), data.1 as u64);
    }
}
impl EncoderParam for &mut Buffer {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data), 0);
    }
}
impl EncoderParam for (&mut Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data.0), data.1 as u64);
    }
}

macro_rules! set_params {
    ($encoder:ident, ($($param:expr),+)) => (
        let mut _index = 0;
        $(
            set_param($encoder, _index, $param);
            _index += 1;
        )*
    );
}

// Seal for EncoderParam so that only the types we want can implement it
mod private {
    use super::*;

    pub trait Sealed {}

    macro_rules! sealed {
        ($($type:ty),+) => {
            $(impl Sealed for $type {})+
        };
    }
    sealed!(usize, u8, u32, u64, i32, i64, f16, bf16, f32, bool);
    sealed!(&Buffer, (&Buffer, usize), &mut Buffer, (&mut Buffer, usize));
    impl<T> Sealed for &[T] {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Affine,
    Indexing,
    Unary,
    Binary,
    Ternary,
    Cast,
    Reduce,
    Mfa,
    Conv,
    Fill,
    Random,
    Quantized,
}

macro_rules! ops{
    ($($name:ident),+) => {

        pub mod contiguous {
        pub struct Kernel(pub &'static str);
        $(
        pub mod $name {
            use super::Kernel;
            pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_f32"));
            pub const HALF: Kernel = Kernel(concat!(stringify!($name), "_f16"));
            pub const BFLOAT: Kernel = Kernel(concat!(stringify!($name), "_bf16"));
            pub const I64: Kernel = Kernel(concat!(stringify!($name), "_i64"));
            pub const U32: Kernel = Kernel(concat!(stringify!($name), "_u32"));
            pub const U8: Kernel = Kernel(concat!(stringify!($name), "_u8"));
        }
        )+
            pub mod copy {
                use super::Kernel;
                pub const FLOAT: Kernel = Kernel("copy_f32");
                pub const HALF: Kernel = Kernel("copy_f16");
                pub const BFLOAT: Kernel = Kernel("copy_bf16");
                pub const I64: Kernel = Kernel("copy_i64");
                pub const U32: Kernel = Kernel("copy_u32");
                pub const U8: Kernel = Kernel("copy_u8");
            }
        }

        pub mod strided {
        pub struct Kernel(pub &'static str);
        $(
        pub mod $name {
            use super::Kernel;
            pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_f32_strided"));
            pub const HALF: Kernel = Kernel(concat!(stringify!($name), "_f16_strided"));
            pub const BFLOAT: Kernel = Kernel(concat!(stringify!($name), "_bf16_strided"));
            pub const I64: Kernel = Kernel(concat!(stringify!($name), "_i64_strided"));
            pub const U32: Kernel = Kernel(concat!(stringify!($name), "_u32_strided"));
            pub const U8: Kernel = Kernel(concat!(stringify!($name), "_u8_strided"));
        }
        )+
            pub mod copy {
                use super::Kernel;
                pub const FLOAT: Kernel = Kernel("copy_f32_strided");
                pub const HALF: Kernel = Kernel("copy_f16_strided");
                pub const BFLOAT: Kernel = Kernel("copy_bf16_strided");
                pub const I64: Kernel = Kernel("copy_i64_strided");
                pub const U32: Kernel = Kernel("copy_u32_strided");
                pub const U8: Kernel = Kernel("copy_u8_strided");
            }
        }
    };
}

pub mod unary {
    ops!(
        cos, sin, exp, sqr, sqrt, neg, log, gelu, abs, ceil, floor, relu, round, erf, gelu_erf,
        tanh, recip, silu
    );
}
pub mod binary {
    ops!(add, sub, mul, div, min, max, eq, ne, le, lt, ge, gt);
}

#[derive(thiserror::Error, Debug)]
pub enum MetalKernelError {
    #[error("Invalid usage of kernel: {0}")]
    InvalidUsage(String),
    #[error("Could not lock kernel map: {0}")]
    LockError(String),
    #[error("Error while loading library: {0}")]
    LoadLibraryError(String),
    #[error("Error while loading function: {0:?}")]
    LoadFunctionError(String),
    #[error("Failed to create compute function")]
    FailedToCreateComputeFunction,
    #[error("Failed to create pipeline")]
    FailedToCreatePipeline(String),
    #[error("Invalid matmul arguments {lhs_stride:?} {rhs_stride:?} {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Vec<usize>,
        rhs_stride: Vec<usize>,
        mnk: (usize, usize, usize),
    },
}

impl<T> From<std::sync::PoisonError<T>> for MetalKernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}

type Libraries = HashMap<Source, Library>;
type Pipelines = HashMap<(&'static str, Option<ConstantValues>), ComputePipelineState>;

#[derive(Debug)]
pub struct Kernels {
    libraries: RwLock<Libraries>,
    pipelines: RwLock<Pipelines>,
}

impl Kernels {
    pub fn new() -> Self {
        let libraries = RwLock::new(Libraries::new());
        let pipelines = RwLock::new(Pipelines::new());
        Self {
            libraries,
            pipelines,
        }
    }

    fn get_library_source(&self, source: Source) -> &'static str {
        match source {
            Source::Affine => AFFINE,
            Source::Unary => UNARY,
            Source::Binary => BINARY,
            Source::Ternary => TERNARY,
            Source::Indexing => INDEXING,
            Source::Cast => CAST,
            Source::Reduce => REDUCE,
            Source::Fill => FILL,
            Source::Conv => CONV,
            Source::Random => RANDOM,
            Source::Quantized => QUANTIZED,
            Source::Mfa => panic!("Invalid lib"),
        }
    }

    /// Load the give library from its [`source`].
    /// If this has been previously loaded it will just fetch it from cache.
    pub fn load_library(
        &self,
        device: &Device,
        source: Source,
    ) -> Result<Library, MetalKernelError> {
        let mut libraries = self.libraries.write()?;
        if let Some(lib) = libraries.get(&source) {
            Ok(lib.clone())
        } else {
            let lib = match source {
                Source::Mfa => {
                    let source_data = MFA;
                    device.new_library_with_data(source_data).map_err(|e| {
                        MetalKernelError::LoadLibraryError(format!(
                            "Candle metal requires macosx > 13.0 or higher, cannot load mfa: {e}"
                        ))
                    })?
                }
                source => {
                    let source_content = self.get_library_source(source);
                    device
                        .new_library_with_source(source_content, &CompileOptions::new())
                        .map_err(|e| MetalKernelError::LoadLibraryError(e.to_string()))?
                }
            };
            libraries.insert(source, lib.clone());
            Ok(lib)
        }
    }

    fn load_function(
        &self,
        device: &Device,
        source: Source,
        name: &'static str,
        constants: Option<FunctionConstantValues>,
    ) -> Result<Function, MetalKernelError> {
        let func = self
            .load_library(device, source)?
            .get_function(name, constants)
            .map_err(|e| MetalKernelError::LoadFunctionError(e.to_string()))?;
        Ok(func)
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source
    fn load_pipeline_with_constants(
        &self,
        device: &Device,
        source: Source,
        name: &'static str,
        constants: Option<ConstantValues>,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        let mut pipelines = self.pipelines.write()?;
        let key = (name, constants);
        if let Some(pipeline) = pipelines.get(&key) {
            Ok(pipeline.clone())
        } else {
            let (name, constants) = key;
            let func = self.load_function(
                device,
                source,
                name,
                constants.as_ref().map(|c| c.function_constant_values()),
            )?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
            pipelines.insert((name, constants), pipeline.clone());

            Ok(pipeline)
        }
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source (without constants)
    pub fn load_pipeline(
        &self,
        device: &Device,
        source: Source,
        name: &'static str,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        self.load_pipeline_with_constants(device, source, name, None)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_unary_contiguous(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    kernel_name: unary::contiguous::Kernel,
    length: usize,
    input: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, kernel_name.0)?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, input, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_unary_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: unary::strided::Kernel,
    shape: &[usize],
    input: &Buffer,
    strides: &[usize],
    offset: usize,
    output: &Buffer,
    output_offset: usize,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, name.0)?;

    let num_dims: usize = shape.len();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    let length: usize = shape.iter().product();
    set_params!(
        encoder,
        (
            length,
            num_dims,
            shape,
            strides,
            (input, offset),
            (output, output_offset)
        )
    );

    let width: usize = shape.iter().product();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, width);

    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_binary_contiguous(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    kernel_name: binary::contiguous::Kernel,
    length: usize,
    left: &Buffer,
    right: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Binary, kernel_name.0)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, left, right, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.use_resource(left, metal::MTLResourceUsage::Read);
    encoder.use_resource(right, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_binary_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: binary::strided::Kernel,
    shape: &[usize],
    left_input: &Buffer,
    left_strides: &[usize],
    left_offset: usize,
    right_input: &Buffer,
    right_strides: &[usize],
    right_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Binary, name.0)?;

    let num_dims: usize = shape.len();
    let encoder = command_buffer.new_compute_command_encoder();
    let width: usize = shape.iter().product();
    encoder.set_compute_pipeline_state(&pipeline);

    let length: usize = shape.iter().product();

    set_params!(
        encoder,
        (
            length,
            num_dims,
            shape,
            left_strides,
            right_strides,
            (left_input, left_offset),
            (right_input, right_offset),
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, width);

    encoder.use_resource(left_input, metal::MTLResourceUsage::Read);
    encoder.use_resource(right_input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_cast_contiguous(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Cast, kernel_name)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, (input, input_offset), output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_cast_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    input: &Buffer,
    input_strides: &[usize],
    input_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Cast, kernel_name)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    let length: usize = shape.iter().product();

    set_params!(
        encoder,
        (
            length,
            shape.len(),
            shape,
            input_strides,
            (input, input_offset),
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

pub fn call_reduce_contiguous(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    out_length: usize,
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let elements_to_sum = length / out_length;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (length, elements_to_sum, (input, input_offset), output)
    );

    let thread_group_count = MTLSize {
        width: out_length as u64,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (elements_to_sum as u64 + 2 - 1) / 2,
    )
    .next_power_of_two();

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

pub fn call_reduce_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_length: usize,
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let length: usize = shape.iter().product();
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let elements_to_sum = length / out_length;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            shape.len(),
            shape,
            strides,
            elements_to_sum,
            (input, input_offset),
            output
        )
    );

    let thread_group_count = MTLSize {
        width: out_length as u64,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        elements_to_sum as u64,
    )
    .next_power_of_two();

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_last_softmax(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (length, elements_to_sum, (input, input_offset), output)
    );

    let out_length = length / elements_to_sum;

    let thread_group_count = MTLSize {
        width: out_length as u64,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        elements_to_sum as u64,
    )
    .next_power_of_two();

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_affine(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    size: usize,
    input: &Buffer,
    output: &Buffer,
    mul: f32,
    add: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (size, mul, add, input, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_affine_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    input: &Buffer,
    input_stride: &[usize],
    input_offset: usize,
    output: &Buffer,
    mul: f32,
    add: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;
    let size: usize = shape.iter().product();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            size,
            shape.len(),
            shape,
            input_stride,
            mul,
            add,
            (input, input_offset),
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_powf(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    size: usize,
    input: &Buffer,
    output: &Buffer,
    mul: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (size, mul, input, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_powf_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    input: &Buffer,
    input_stride: &[usize],
    input_offset: usize,
    output: &Buffer,
    mul: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;
    let size: usize = shape.iter().product();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            size,
            shape.len(),
            shape,
            input_stride,
            mul,
            (input, input_offset),
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_elu(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    size: usize,
    input: &Buffer,
    output: &Buffer,
    mul: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (size, mul, input, output));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_elu_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    input: &Buffer,
    input_stride: &[usize],
    input_offset: usize,
    output: &Buffer,
    mul: f32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Affine, name)?;
    let size: usize = shape.iter().product();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            size,
            shape.len(),
            shape,
            input_stride,
            mul,
            (input, input_offset),
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

pub fn call_where_cond_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    cond: &Buffer,
    (cond_stride, cond_offset): (&[usize], usize),
    left: &Buffer,
    (left_stride, left_offset): (&[usize], usize),
    right: &Buffer,
    (right_stride, right_offset): (&[usize], usize),
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Ternary, name)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    let size: usize = shape.iter().product();
    let rank = shape.len();

    set_params!(
        encoder,
        (
            size,
            rank,
            shape,
            cond_stride,
            left_stride,
            right_stride,
            (cond, cond_offset),
            (left, left_offset),
            (right, right_offset),
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);

    encoder.use_resource(cond, metal::MTLResourceUsage::Read);
    encoder.use_resource(left, metal::MTLResourceUsage::Read);
    encoder.use_resource(right, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_index_select(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    ids_size: usize,
    dim: usize,
    input: &Buffer,
    ids: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let src_dim_size = shape[dim];
    let dst_el = ids_size * left_size * right_size;

    let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            dst_el,
            left_size,
            src_dim_size,
            right_size,
            ids_size,
            input,
            ids,
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(ids, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_gather(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    ids_size: usize,
    dim: usize,
    input: &Buffer,
    input_offset: usize,
    ids: &Buffer,
    ids_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let src_dim_size = shape[dim];
    let dst_el = ids_size * left_size * right_size;

    let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            dst_el,
            left_size,
            src_dim_size,
            right_size,
            ids_size,
            (input, input_offset),
            (ids, ids_offset),
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(ids, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

pub fn call_scatter_add(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    src_shape: &[usize],
    dst_shape: &[usize],
    dim: usize,
    input: &Buffer,
    input_offset: usize,
    ids: &Buffer,
    ids_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let left_size: usize = src_shape[..dim].iter().product();
    let right_size: usize = src_shape[dim + 1..].iter().product();
    let src_dim_size = src_shape[dim];
    let dst_el = left_size * right_size;
    let dst_dim_size = dst_shape[dim];

    let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            dst_el,
            left_size,
            src_dim_size,
            right_size,
            dst_dim_size,
            (input, input_offset),
            (ids, ids_offset),
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(ids, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

pub fn call_index_add(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    src_shape: &[usize],
    dst_shape: &[usize],
    ids_shape: &[usize],
    dim: usize,
    input: &Buffer,
    input_offset: usize,
    ids: &Buffer,
    ids_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let left_size: usize = src_shape[..dim].iter().product();
    let right_size: usize = src_shape[dim + 1..].iter().product();
    let src_dim_size = src_shape[dim];
    let dst_el = left_size * right_size;
    let dst_dim_size = dst_shape[dim];
    let ids_dim_size = ids_shape[0];

    let pipeline = kernels.load_pipeline(device, Source::Indexing, name)?;
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            dst_el,
            left_size,
            src_dim_size,
            right_size,
            dst_dim_size,
            ids_dim_size,
            (input, input_offset),
            (ids, ids_offset),
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);

    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(ids, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[derive(Debug, PartialEq)]
pub enum Value {
    USize(usize),
    Bool(bool),
    F32(f32),
    U16(u16),
}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::F32(v) => v.to_bits().hash(state),
            Value::USize(v) => v.hash(state),
            Value::U16(v) => v.hash(state),
            Value::Bool(v) => v.hash(state),
        }
    }
}

impl Value {
    fn data_type(&self) -> MTLDataType {
        match self {
            Value::USize(_) => MTLDataType::UInt,
            Value::F32(_) => MTLDataType::Float,
            Value::U16(_) => MTLDataType::UShort,
            Value::Bool(_) => MTLDataType::Bool,
        }
    }
}

/// Not true, good enough for our purposes.
impl Eq for Value {}

#[derive(Debug, Eq, PartialEq, Hash)]
struct ConstantValues(Vec<(usize, Value)>);

impl ConstantValues {
    pub fn new(values: Vec<(usize, Value)>) -> Self {
        Self(values)
    }

    fn function_constant_values(&self) -> FunctionConstantValues {
        let f = FunctionConstantValues::new();
        for (index, value) in &self.0 {
            let ty = value.data_type();
            match value {
                Value::USize(v) => {
                    f.set_constant_value_at_index(
                        v as *const usize as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
                Value::F32(v) => {
                    f.set_constant_value_at_index(
                        v as *const f32 as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
                Value::U16(v) => {
                    f.set_constant_value_at_index(
                        v as *const u16 as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
                Value::Bool(v) => {
                    f.set_constant_value_at_index(
                        v as *const bool as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
            }
        }
        f
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_gemm(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    assert!(rhs_stride.len() >= 2);
    assert!(lhs_stride.len() >= 2);
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    let a_trans = if lhs_m1 == 1 && lhs_m2 == k {
        false
    } else if lhs_m1 == m && lhs_m2 == 1 {
        true
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        })?;
    };
    let b_trans = if rhs_m1 == 1 && rhs_m2 == n {
        false
    } else if rhs_m1 == k && rhs_m2 == 1 {
        true
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        })?;
    };
    let d_trans = false;
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let batched = b > 1;
    let fused_activation = false;
    let fused_bias = false;
    let (m_simd, n_simd, k_simd, m_splits, n_splits) = if m == 1 {
        let m_simd = 8;
        let n_simd = 8;
        let k_simd = 64;
        let m_splits = 1;
        let n_splits = 1;
        (m_simd, n_simd, k_simd, m_splits, n_splits)
    } else {
        let m_simd = 40;
        let n_simd = 40;
        let k_simd = 32;
        let m_splits = 1;
        let n_splits = 1;
        (m_simd, n_simd, k_simd, m_splits, n_splits)
    };
    let constants = Some(ConstantValues::new(vec![
        (0, Value::USize(m)),
        (1, Value::USize(n)),
        (2, Value::USize(k)),
        (10, Value::Bool(a_trans)),
        (11, Value::Bool(b_trans)),
        (13, Value::Bool(d_trans)),
        (20, Value::F32(alpha)),
        (21, Value::F32(beta)),
        (100, Value::Bool(batched)),
        (101, Value::Bool(fused_activation)),
        // Garbage
        (102, Value::Bool(false)),
        (103, Value::Bool(false)),
        (113, Value::Bool(false)),
        (50_000, Value::Bool(false)),
        // End garbage
        (200, Value::U16(m_simd)),
        (201, Value::U16(n_simd)),
        (202, Value::U16(k_simd)),
        (210, Value::U16(m_splits)),
        (211, Value::U16(n_splits)),
        (50_001, Value::Bool(fused_bias)),
    ]));
    let pipeline = kernels.load_pipeline_with_constants(device, Source::Mfa, name, constants)?;
    let m_group = m_simd * m_splits;
    let n_group = n_simd * n_splits;

    let a_block_length = m_group * k_simd;
    let b_block_length = k_simd * n_group;

    let mut block_elements = a_block_length + b_block_length;
    if (m % 8 != 0) && (n % 8 != 0) {
        let c_block_length = m_group * n_group;
        block_elements = std::cmp::max(c_block_length, block_elements)
    }
    if fused_bias {
        if d_trans {
            block_elements = std::cmp::max(block_elements, m_group);
        } else {
            block_elements = std::cmp::max(block_elements, n_group);
        }
    }
    let bytes = match name {
        "sgemm" => 4,
        "hgemm" => 2,
        other => {
            return Err(MetalKernelError::LoadLibraryError(format!(
                "{other} is not a valid kernel for gemm"
            )));
        }
    };
    let block_bytes = block_elements * bytes;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_threadgroup_memory_length(0, block_bytes.into());
    encoder.set_buffer(0, Some(lhs_buffer), lhs_offset as NSUInteger);
    encoder.set_buffer(1, Some(rhs_buffer), rhs_offset as NSUInteger);
    encoder.set_buffer(2, Some(output), 0);
    // TODO Tensor D

    let grid_z = b;
    if batched {
        let byte_stride_a: usize = lhs_stride[lhs_stride.len() - 3] * bytes as usize;
        let byte_stride_b: usize = rhs_stride[rhs_stride.len() - 3] * bytes as usize;
        let byte_stride_c = m * n * bytes as usize;
        // TODO byte_stride_d
        let byte_stride_d = 0;

        let buffer: Vec<u64> = vec![
            byte_stride_a as _,
            byte_stride_b as _,
            byte_stride_c as _,
            byte_stride_d as _,
        ];
        encoder.set_bytes(
            10,
            (buffer.len() * core::mem::size_of::<u64>()) as NSUInteger,
            buffer.as_ptr() as *const NSUInteger as *const c_void,
        );
    }

    let grid_size = MTLSize {
        width: divide(n, n_group.into()),
        height: divide(m, m_group.into()),
        depth: grid_z as NSUInteger,
    };
    let group_size = MTLSize {
        width: 32 * (m_splits as u64) * (n_splits as u64),
        height: 1,
        depth: 1,
    };
    encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_size, group_size);
    encoder.end_encoding();

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_im2col1d_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (k_size, stride, padding, dilation): (usize, usize, usize, usize),
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let l_out = (shape[2] + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;
    let dst_el = shape[0] * l_out * shape[1] * k_size;

    let encoder = command_buffer.new_compute_command_encoder();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            dst_el,
            l_out,
            k_size,
            stride,
            padding,
            dilation,
            shape,
            strides,
            (input, input_offset),
            output
        )
    );
    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_im2col_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (h_k, w_k, stride, padding, dilation): (usize, usize, usize, usize, usize),
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;

    let h = shape[2];
    let w = shape[3];
    let h_out = (h + 2 * padding - dilation * (h_k - 1) - 1) / stride + 1;
    let w_out = (w + 2 * padding - dilation * (w_k - 1) - 1) / stride + 1;

    let dst_el = shape[0] * h_out * w_out * shape[1] * h_k * w_k;

    let encoder = command_buffer.new_compute_command_encoder();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            dst_el,
            h_out,
            w_out,
            h_k,
            w_k,
            stride,
            padding,
            dilation,
            shape,
            strides,
            (input, input_offset),
            output
        )
    );
    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_upsample_nearest_2d(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, name)?;
    let dst_el = out_w * out_h * shape[0] * shape[1];
    let scale_w = shape[2] as f32 / out_w as f32;
    let scale_h = shape[3] as f32 / out_h as f32;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            out_w,
            out_h,
            scale_w,
            scale_h,
            shape,
            strides,
            (input, input_offset),
            output
        )
    );
    encoder.use_resource(input, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_random_uniform(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    min: f32,
    max: f32,
    length: usize,
    seed: &Buffer,
    buffer: &Buffer,
) -> Result<(), MetalKernelError> {
    if min >= max {
        return Err(MetalKernelError::LoadLibraryError(
            "min must be less than max".to_string(),
        ));
    }
    let pipeline = kernels.load_pipeline(device, Source::Random, name)?;
    let encoder = command_buffer.new_compute_command_encoder();

    let odd = (length % 2 != 0) as usize;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length / 2 + odd);

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, min, max, seed, buffer));

    encoder.use_resource(
        seed,
        metal::MTLResourceUsage::Read | metal::MTLResourceUsage::Write,
    );
    encoder.use_resource(buffer, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_random_normal(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: &'static str,
    mean: f32,
    stddev: f32,
    length: usize,
    seed: &Buffer,
    buffer: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Random, name)?;
    let encoder = command_buffer.new_compute_command_encoder();

    let odd = (length % 2 != 0) as usize;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length / 2 + odd);

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, mean, stddev, seed, buffer));

    encoder.use_resource(
        seed,
        metal::MTLResourceUsage::Read | metal::MTLResourceUsage::Write,
    );
    encoder.use_resource(buffer, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();

    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub enum GgmlDType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    F16,
    F32,
}

pub fn call_quantized_matmul_t(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    dtype: GgmlDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1 as i64;

    let nb00 = 0i64;
    let nb01 = 0 as i64;
    let nb02 = 0 as i64;

    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1 as i64;

    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;

    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    let (nth0, nth1, align) = match dtype {
        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_0
        | GgmlDType::Q8_1 => {
            let nth0 = 8;
            let nth1 = 8;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::Q2K => {
            // Fixing a bug in Metal for GGML
            let nth0 = 4;
            let nth1 = 8;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q4K => {
            let nth0 = 4;
            let nth1 = 8;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q3K | GgmlDType::Q5K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q6K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 2;
            (nth0, nth1, align)
        }
        GgmlDType::F16 | GgmlDType::Q8K => {
            // Original implem uses rows
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::F32 => {
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
    };
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: ne11 as u64,
        depth: (ne12 * ne13) as u64,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mv_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mv_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mv_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mv_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_q8_0_f32",
        GgmlDType::Q8_1 => "kernel_mul_mv_q8_1_f32",
        GgmlDType::Q2K => "kernel_mul_mv_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mv_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mv_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mv_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mv_q6_K_f32",
        GgmlDType::Q8K => "kernel_mul_mv_q8_K_f32",
        GgmlDType::F16 => "kernel_mul_mv_f16_f32",
        GgmlDType::F32 => "kernel_mul_mv_f32_f32",
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            rhs,
            (lhs, lhs_offset),
            output,
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.set_threadgroup_memory_length(0, 8192);
    encoder.use_resource(lhs, metal::MTLResourceUsage::Read);
    encoder.use_resource(rhs, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    encoder.end_encoding();

    Ok(())
}

#[inline(always)]
fn divide(m: usize, b: usize) -> NSUInteger {
    ((m + b - 1) / b) as NSUInteger
}

pub fn call_fill<T: FillOp>(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    elem_count: usize,
    buffer: &Buffer,
    value: T,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Fill, T::FILL_KERNEL)?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_threadgroup_memory_length(0, elem_count as NSUInteger);

    set_params!(encoder, (buffer, value, elem_count));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, elem_count);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.use_resource(buffer, metal::MTLResourceUsage::Write);
    encoder.end_encoding();

    Ok(())
}

pub fn call_fill_u8(
    command_buffer: &CommandBufferRef,
    elem_count: usize,
    buffer: &Buffer,
    value: u8,
) -> Result<(), MetalKernelError> {
    let blit = command_buffer.new_blit_command_encoder();
    blit.fill_buffer(
        buffer,
        metal::NSRange {
            location: 0,
            length: elem_count as NSUInteger,
        },
        value,
    );
    blit.end_encoding();

    Ok(())
}

pub trait FillOp: EncoderParam {
    const FILL_KERNEL: &'static str;
}

macro_rules ! impl_call_fill {
    ($($t:ty),*) => {
        $(
            impl FillOp for $t {
                const FILL_KERNEL: &'static str = concat!("fill_", stringify!($t));
            }
        )*
    };
}
impl_call_fill!(u8, u32, i64, f16, bf16, f32);

#[cfg(test)]
mod tests;
