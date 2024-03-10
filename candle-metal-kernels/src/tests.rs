use super::*;
use half::{bf16, f16};
use metal::{Buffer, Device, MTLResourceOptions};

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

fn new_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    let options = MTLResourceOptions::StorageModeManaged;
    let ptr = data.as_ptr() as *const c_void;
    let size = (data.len() * std::mem::size_of::<T>()) as u64;
    device.new_buffer_with_data(ptr, size, options)
}

fn device() -> Device {
    Device::system_default().unwrap()
}

fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t * b) / b).collect()
}

fn approx_f16(v: Vec<f16>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t.to_f32() * b) / b).collect()
}

fn approx_bf16(v: Vec<bf16>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t.to_f32() * b) / b).collect()
}

fn run<T: Clone>(v: &[T], name: unary::contiguous::Kernel) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);
    call_unary_contiguous(
        &device,
        command_buffer,
        &kernels,
        name,
        v.len(),
        &input,
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, v.len())
}

fn run_binary<T: Clone>(x: &[T], y: &[T], name: binary::contiguous::Kernel) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let options = MTLResourceOptions::StorageModeManaged;
    let left = new_buffer(&device, x);
    let right = new_buffer(&device, y);
    let output = device.new_buffer(std::mem::size_of_val(x) as u64, options);
    call_binary_contiguous(
        &device,
        command_buffer,
        &kernels,
        name,
        x.len(),
        &left,
        &right,
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, x.len())
}

fn run_strided<T: Clone>(
    v: &[T],
    kernel: unary::strided::Kernel,
    shape: &[usize],
    strides: &[usize],
    offset: usize,
) -> Vec<T> {
    let device = device();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);
    let kernels = Kernels::new();
    call_unary_strided(
        &device,
        command_buffer,
        &kernels,
        kernel,
        shape,
        &input,
        strides,
        offset,
        &output,
        0,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, v.len())
}

#[test]
fn cos_f32() {
    let v = vec![1.0f32, 2.0, 3.0];
    let results = run(&v, unary::contiguous::cos::FLOAT);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403, -0.4161, -0.99]);
    assert_eq!(approx(expected, 4), vec![0.5403, -0.4161, -0.99]);

    let v = vec![1.0f32; 10_000];
    let results = run(&v, unary::contiguous::cos::FLOAT);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403; 10_000]);
    assert_eq!(approx(expected, 4), vec![0.5403; 10_000]);
}

#[test]
fn cos_f32_strided() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![6];
    let strides = vec![1];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Contiguous
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![3, 2];
    let strides = vec![2, 1];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Transposed
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![3, 2];
    let strides = vec![1, 3];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(
        approx(results, 4),
        vec![0.5403, -0.6536, -0.4161, 0.2837, -0.99, 0.9602]
    );
    assert_eq!(
        approx(expected, 4),
        vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
    );

    // Very large
    let v = vec![1.0f32; 10_000];
    let shape = vec![2, 5_000];
    let strides = vec![2, 1];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(results, 4), vec![0.5403; 10_000]);
    assert_eq!(approx(expected, 4), vec![0.5403; 10_000]);
}

#[test]
fn cos_strided_random() {
    let v: Vec<_> = (0..10_000).map(|_| rand::random::<f32>()).collect();
    let shape = vec![5_000, 2];
    let strides = vec![1, 5_000];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(vec![results[0]], 4), approx(vec![expected[0]], 4));
    assert_eq!(
        approx(vec![results[1]], 4),
        approx(vec![expected[5_000]], 4)
    );
    assert_eq!(approx(vec![results[2]], 4), approx(vec![expected[1]], 4));
    assert_eq!(
        approx(vec![results[3]], 4),
        approx(vec![expected[5_001]], 4)
    );
    assert_eq!(
        approx(vec![results[5_000]], 4),
        approx(vec![expected[2_500]], 4)
    );
}

#[test]
fn gelu_f16() {
    let v: Vec<f16> = [-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let expected: Vec<f32> = vec![-0.0, -0.16, 0.0, 0.84, 1.96, 3.0, 10.0, 20.0];
    let results = run(&v, unary::contiguous::gelu::HALF);
    assert_eq!(approx_f16(results, 2), expected);
}

#[test]
fn gelu_f32() {
    let v: Vec<f32> = vec![-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0];
    let expected: Vec<f32> = vec![-0.0, -0.159, 0.0, 0.841, 1.955, 2.996, 10.0, 20.0];
    let results = run(&v, unary::contiguous::gelu::FLOAT);
    assert_eq!(approx(results, 3), expected);
}

#[test]
fn silu_f16() {
    let v: Vec<f16> = [-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let expected: Vec<f32> = vec![-0.0, -0.27, 0.0, 0.73, 1.76, 2.86, 10.0, 20.0];
    let results = run(&v, unary::contiguous::silu::HALF);
    assert_eq!(approx_f16(results, 2), expected);
}

#[test]
fn silu_f32() {
    let v: Vec<f32> = vec![-10f32, -1.0, 0., 1., 2., 3., 10.0, 20.0];
    let expected: Vec<f32> = vec![-0.0, -0.269, 0.0, 0.731, 1.762, 2.858, 10.0, 20.0];
    let results = run(&v, unary::contiguous::silu::FLOAT);
    assert_eq!(approx(results, 3), expected);
}

#[test]
fn binary_add_f32() {
    let left = vec![1.0f32, 2.0, 3.0];
    let right = vec![2.0f32, 3.1, 4.2];
    let results = run_binary(&left, &right, binary::contiguous::add::FLOAT);
    let expected: Vec<_> = left
        .iter()
        .zip(right.iter())
        .map(|(&x, &y)| x + y)
        .collect();
    assert_eq!(approx(results, 4), vec![3.0f32, 5.1, 7.2]);
    assert_eq!(approx(expected, 4), vec![3.0f32, 5.1, 7.2]);
}

#[test]
fn binary_ops_bf16() {
    let lhs: Vec<bf16> = [1.1f32, 2.2, 3.3].into_iter().map(bf16::from_f32).collect();
    let rhs: Vec<bf16> = [4.2f32, 5.5f32, 6.91f32]
        .into_iter()
        .map(bf16::from_f32)
        .collect();

    macro_rules! binary_op {
        ($opname:ident, $opexpr:expr) => {{
            let results = run_binary(&lhs, &rhs, binary::contiguous::$opname::BFLOAT);
            let expected: Vec<bf16> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&bf16, &bf16)| $opexpr(*x, *y))
                .collect();
            assert_eq!(results, expected);
        }};
    }

    binary_op!(add, |x, y| x + y);
    binary_op!(sub, |x, y| x - y);
    binary_op!(mul, |x, y| x * y);
    binary_op!(div, |x, y| x / y);
    binary_op!(min, |x: bf16, y| x.min(y));
    binary_op!(max, |x: bf16, y| x.max(y));
}

fn cast<T: Clone, U: Clone>(v: &[T], name: &'static str) -> Vec<U> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let input = new_buffer(&device, v);
    let options = MTLResourceOptions::StorageModeManaged;
    let size = (v.len() * std::mem::size_of::<U>()) as u64;
    let output = device.new_buffer(size, options);

    call_cast_contiguous(
        &device,
        command_buffer,
        &kernels,
        name,
        v.len(),
        &input,
        0,
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, v.len())
}

#[test]
fn cast_u32_f32() {
    let v = vec![1u32, 2, 3];
    let results = cast(&v, "cast_u32_f32");
    let expected: Vec<_> = v.iter().map(|&v| v as f32).collect();
    assert_eq!(approx(results, 4), vec![1.0f32, 2.0, 3.0]);
    assert_eq!(approx(expected, 4), vec![1.0f32, 2.0, 3.0]);

    let v = vec![1.0f32, 2.0, 3.0];
    let input: Vec<f16> = v.iter().map(|v| f16::from_f32(*v)).collect();
    let results: Vec<f32> = cast(&input, "cast_f16_f32");
    assert_eq!(results, vec![1.0f32, 2.0, 3.0]);

    let v = vec![1.0f32; 10_000];
    let input: Vec<f16> = v.iter().map(|v| f16::from_f32(*v)).collect();
    let results: Vec<f32> = cast(&input, "cast_f16_f32");
    assert_eq!(results.len(), 10_000);
    assert_eq!(&results[..10], vec![1.0f32; 10]);
    assert_eq!(results, vec![1.0f32; 10_000]);
}

#[test]
fn it_cast_bf16_u32() {
    let input: Vec<bf16> = (1..=3).map(|v| bf16::from_f32(v as f32)).collect();

    let output: Vec<u32> = cast(&input, "cast_bf16_u32");
    let expected: Vec<u32> = (1..=3).map(|v| v as u32).collect();

    assert_eq!(output, expected);
}

#[test]
fn it_cast_bf16_f32() {
    let input: Vec<bf16> = (1..=3).map(|v| bf16::from_f32(v as f32)).collect();

    let output: Vec<f32> = cast(&input, "cast_bf16_f32");
    let expected: Vec<f32> = (1..=3).map(|v| v as f32).collect();

    assert_eq!(output, expected);
}

#[test]
fn it_cast_u8_bf16() {
    let input: Vec<u8> = (1..=3).map(|v| v as u8).collect();

    let output: Vec<bf16> = cast(&input, "cast_u8_bf16");
    let expected: Vec<bf16> = input
        .iter()
        .map(|v| bf16::from_f32(*v as f32))
        .collect::<Vec<_>>();

    assert_eq!(output, expected);
}

#[test]
fn it_cast_u32_bf16() {
    let input: Vec<u32> = (1..=3).map(|v| v as u32).collect();

    let output: Vec<bf16> = cast(&input, "cast_u32_bf16");
    let expected: Vec<bf16> = input.iter().map(|v| bf16::from_f32(*v as f32)).collect();

    assert_eq!(output, expected);
}

#[test]
fn it_cast_f32_bf16() {
    let input: Vec<f32> = (1..=3).map(|v| v as f32).collect();

    let output: Vec<bf16> = cast(&input, "cast_f32_bf16");
    let expected: Vec<bf16> = input.iter().map(|v| bf16::from_f32(*v as f32)).collect();

    assert_eq!(output, expected);
}

#[test]
fn it_cast_bf16_u8() {
    let input: Vec<bf16> = (1..=3).map(|v| bf16::from_f32(v as f32)).collect();

    let output: Vec<u8> = cast(&input, "cast_bf16_u8");
    let expected: Vec<u8> = input.iter().map(|v| v.to_f32() as u8).collect();

    assert_eq!(output, expected);
}

#[test]
fn it_cast_bf16_f16() {
    let input: Vec<bf16> = (1..=3).map(|v| bf16::from_f32(v as f32)).collect();

    let output: Vec<f16> = cast(&input, "cast_bf16_f16");
    let expected: Vec<f16> = input.iter().map(|v| f16::from_f32(v.to_f32())).collect();

    assert_eq!(output, expected);
}

#[test]
fn it_cast_f16_bf16() {
    let input: Vec<f16> = (1..=3).map(|v| f16::from_f32(v as f32)).collect();

    let output: Vec<bf16> = cast(&input, "cast_f16_bf16");
    let expected: Vec<bf16> = input.iter().map(|v| bf16::from_f32(v.to_f32())).collect();

    assert_eq!(output, expected);
}

fn run_affine<T: Clone>(v: &[T], mul: f64, add: f64) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();

    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);

    let size = v.len();

    call_affine(
        &device,
        command_buffer,
        &kernels,
        "affine_f32",
        size,
        &input,
        &output,
        mul as f32,
        add as f32,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&output, v.len())
}

fn run_affine_strided<T: Clone>(
    v: &[T],
    shape: &[usize],
    strides: &[usize],
    mul: f64,
    add: f64,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();

    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);

    call_affine_strided(
        &device,
        command_buffer,
        &kernels,
        "affine_f32_strided",
        shape,
        &input,
        strides,
        0,
        &output,
        mul as f32,
        add as f32,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let len: usize = shape.iter().product();
    read_to_vec(&output, len)
}

#[test]
fn affine() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mul = 1.5;
    let add = 1.1;
    let result = run_affine(&input, mul, add);
    assert_eq!(result, vec![2.6, 4.1, 5.6, 7.1, 8.6, 10.1, 11.6, 13.1]);

    let input = [1.0f32; 40_000];
    let mul = 1.5;
    let add = 1.1;
    let result = run_affine(&input, mul, add);
    assert_eq!(result, vec![2.6; 40_000]);
}

#[test]
fn affine_strided() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mul = 1.5;
    let add = 1.1;
    let shape = [4];
    let strides = [2];
    let result = run_affine_strided(&input, &shape, &strides, mul, add);
    // 1 on 2
    assert_eq!(result, vec![2.6, 5.6, 8.6, 11.6]);
}

#[test]
fn index_select() {
    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [5, 2];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &ids, dim, "is_u32_f32");
    assert_eq!(result, vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]);

    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [2, 5];
    let ids = [0u32, 1, 0];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &ids, dim, "is_u32_f32");
    assert_eq!(
        result,
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0f32, 2.0, 3.0, 4.0, 5.0]
    );
}

#[test]
fn index_select_f16() {
    let embedding: Vec<_> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        .into_iter()
        .map(|x| f16::from_f32(x))
        .collect();
    let shape = [5, 2];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &ids, dim, "is_u32_f16");
    assert_eq!(
        approx_f16(result, 4),
        vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]
    );
}

#[test]
fn index_select_is_u32_bf16() {
    let embedding: Vec<bf16> = (1..=10).map(|x| bf16::from_f32(x as f32)).collect();
    let shape = [5, 2];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &ids, dim, "is_u32_bf16");
    assert_eq!(
        approx_bf16(result, 4),
        vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]
    );
}

#[test]
fn index_select_is_u8_bf16() {
    let embedding: Vec<bf16> = (1..=10).map(|x| bf16::from_f32(x as f32)).collect();
    let shape = [5, 2];
    let ids = [0u8, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &ids, dim, "is_u8_bf16");
    assert_eq!(
        approx_bf16(result, 4),
        vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]
    );
}

#[test]
fn index_select_dim1() {
    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [5, 2];
    let ids = [0u32, 1, 0];
    let dim = 1;
    let result = run_index_select(&embedding, &shape, &ids, dim, "is_u32_f32");
    assert_eq!(
        result,
        vec![1.0f32, 2.0, 1.0, 3.0, 4.0, 3.0, 5.0, 6.0, 5.0, 7.0, 8.0f32, 7.0, 9.0, 10.0, 9.0]
    );
}

fn run_index_select<T: Clone, I: Clone + std::fmt::Debug>(
    embeddings: &[T],
    shape: &[usize],
    ids: &[I],
    dim: usize,
    name: &'static str,
) -> Vec<T> {
    let device = Device::system_default().expect("no device found");

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let embeddings_buffer = new_buffer(&device, &embeddings);
    let ids_buffer = new_buffer(&device, &ids);

    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let dst_el = ids.len() * left_size * right_size;
    let dst_buffer = new_buffer(&device, &vec![0.0f32; dst_el]);

    let kernels = Kernels::new();
    call_index_select(
        &device,
        &command_buffer,
        &kernels,
        name,
        shape,
        ids.len(),
        dim,
        &embeddings_buffer,
        &ids_buffer,
        &dst_buffer,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&dst_buffer, dst_el)
}

#[test]
fn cos_f16() {
    let v: Vec<f16> = [1.0f32, 2.0, 3.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let results = run(&v, unary::contiguous::cos::HALF);
    let expected: Vec<f16> = v.iter().map(|v| f16::from_f32(v.to_f32().cos())).collect();
    assert_eq!(approx_f16(results, 2), vec![0.54, -0.42, -0.99]);
    assert_eq!(approx_f16(expected, 2), vec![0.54, -0.42, -0.99]);
}

fn run_reduce<T: Clone>(v: &[T], out_length: usize, name: &'static str) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let input = new_buffer(&device, v);

    let options = MTLResourceOptions::StorageModeManaged;
    let output = device.new_buffer((out_length * core::mem::size_of::<T>()) as u64, options);
    let dims = vec![v.len()];
    let strides = vec![1];
    call_reduce_strided(
        &device,
        command_buffer,
        &kernels,
        name,
        &dims,
        &strides,
        out_length,
        &input,
        0,
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&output, out_length)
}

fn run_softmax<T: Clone + std::fmt::Debug>(v: &[T], last_dim: usize, name: &'static str) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let input = new_buffer(&device, v);
    let output = new_buffer(&device, v);
    call_last_softmax(
        &device,
        command_buffer,
        &kernels,
        name,
        v.len(),
        last_dim,
        &input,
        0,
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&output, v.len())
}

#[test]
fn reduce_sum() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let out_length = 1;

    let results = run_reduce(&v, out_length, "fast_sum_f32_strided");
    assert_eq!(approx(results, 4), vec![21.0]);
}

#[test]
fn reduce_sum2() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let out_length = 2;

    let results = run_reduce(&v, out_length, "fast_sum_f32_strided");
    assert_eq!(approx(results, 4), vec![6.0, 15.0]);
}

#[test]
fn softmax() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_f32");
    assert_eq!(
        approx(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]
    );

    let last_dim = 4096;
    let n = 200;
    let mut v = vec![0.0; n * last_dim];
    for i in 0..n {
        v[i * last_dim] = 20.0;
    }
    let results = run_softmax(&v, last_dim, "softmax_f32");
    let results = approx(results, 4);
    assert_eq!(
        results.iter().map(|&s| s.round() as usize).sum::<usize>(),
        n
    );
    assert_eq!(results[0], 1.0);
    assert_eq!(results[1], 0.0);
    assert_eq!(results[last_dim], 1.0);
    assert_eq!(results[2 * last_dim], 1.0);

    let v = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_f32");
    assert_eq!(
        approx(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]
    );

    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let last_dim = 3;
    let results = run_softmax(&v, last_dim, "softmax_f32");
    assert_eq!(
        approx(results, 4),
        vec![0.0900, 0.2447, 0.6652, 0.0900, 0.2447, 0.6652]
    );

    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect::<Vec<_>>();
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_f16");
    assert_eq!(
        approx_f16(results, 4),
        vec![0.0043, 0.0116, 0.0316, 0.0858, 0.2332, 0.6338]
    );

    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect::<Vec<_>>();
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_bf16");
    assert_eq!(
        approx_bf16(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0859, 0.2324, 0.6328]
    );
}

fn run_where_cond<I: Clone, T: Clone>(
    shape: &[usize],
    cond: &[I],
    (cond_stride, cond_offset): (Vec<usize>, usize),
    left_true: &[T],
    (left_stride, left_offset): (Vec<usize>, usize),
    right_false: &[T],
    (_right_stride, _right_offset): (Vec<usize>, usize),
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let options = MTLResourceOptions::StorageModeManaged;

    let length = cond.len();
    let cond = device.new_buffer_with_data(
        cond.as_ptr() as *const core::ffi::c_void,
        std::mem::size_of_val(cond) as u64,
        options,
    );
    let left = device.new_buffer_with_data(
        left_true.as_ptr() as *const core::ffi::c_void,
        (length * core::mem::size_of::<T>()) as u64,
        options,
    );
    let right = device.new_buffer_with_data(
        right_false.as_ptr() as *const core::ffi::c_void,
        (length * core::mem::size_of::<T>()) as u64,
        options,
    );

    let output = device.new_buffer((length * core::mem::size_of::<T>()) as u64, options);
    call_where_cond_strided(
        &device,
        command_buffer,
        &kernels,
        name,
        shape,
        &cond,
        (&cond_stride, cond_offset),
        &left,
        (&left_stride, left_offset),
        &right,
        (&cond_stride, cond_offset),
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&output, length)
}

#[test]
fn where_cond() {
    let shape = vec![6];
    let cond = vec![0u8, 1, 0, 0, 1, 1];
    let cond_l = (vec![1], 0);
    let left_true = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let left_l = (vec![1], 0);
    let right_false = vec![-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0];
    let right_l = (vec![1], 0);
    let results = run_where_cond(
        &shape,
        &cond,
        cond_l,
        &left_true,
        left_l,
        &right_false,
        right_l,
        "where_u8_f32",
    );
    assert_eq!(approx(results, 4), vec![-1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0]);
}

fn run_gemm<T: Clone>(
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &[T],
    lhs_stride: Vec<usize>,
    lhs_offset: usize,
    rhs: &[T],
    rhs_stride: Vec<usize>,
    rhs_offset: usize,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let options = MTLResourceOptions::StorageModeManaged;

    let lhs = device.new_buffer_with_data(
        lhs.as_ptr() as *const core::ffi::c_void,
        std::mem::size_of_val(lhs) as u64,
        options,
    );
    let rhs = device.new_buffer_with_data(
        rhs.as_ptr() as *const core::ffi::c_void,
        std::mem::size_of_val(rhs) as u64,
        options,
    );
    let length = b * m * n;
    let output = device.new_buffer((length * core::mem::size_of::<T>()) as u64, options);
    call_gemm(
        &device,
        command_buffer,
        &kernels,
        "sgemm",
        (b, m, n, k),
        &lhs_stride,
        lhs_offset,
        &lhs,
        &rhs_stride,
        rhs_offset,
        &rhs,
        &output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&output, length)
}

#[test]
fn gemm() {
    let (b, m, n, k) = (1, 2, 4, 3);
    let lhs_stride = vec![m * k, k, 1];
    let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
    let rhs_stride = vec![n * k, n, 1];
    let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
    let results = run_gemm((b, m, n, k), &lhs, lhs_stride, 0, &rhs, rhs_stride, 0);
    assert_eq!(
        approx(results, 4),
        vec![20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0]
    );

    let (b, m, n, k) = (2, 2, 4, 3);
    let lhs_stride = vec![m * k, k, 1];
    let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
    let rhs_stride = vec![n * k, n, 1];
    let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
    let results = run_gemm((b, m, n, k), &lhs, lhs_stride, 0, &rhs, rhs_stride, 0);
    assert_eq!(
        approx(results, 4),
        vec![
            20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 344.0, 365.0, 386.0, 407.0, 488.0,
            518.0, 548.0, 578.0
        ]
    );

    // OFFSET
    let (b, m, n, k) = (2, 2, 4, 3);
    let lhs_stride = vec![m * k, k, 1];
    let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
    let rhs_stride = vec![n * k, n, 1];
    let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
    // Manually set batch_size=1 and offset 12 elements * 4 the number of bytes for f32
    let results = run_gemm((1, m, n, k), &lhs, lhs_stride, 0, &rhs, rhs_stride, 12 * 4);
    assert_eq!(
        approx(results, 4),
        vec![56.0, 59.0, 62.0, 65.0, 200.0, 212.0, 224.0, 236.0]
    );
}

fn run_fill<T: FillOp + Clone>(elem_count: usize, value: T) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let buffer = new_buffer(&device, &vec![0.0f32; elem_count]);
    call_fill(
        &device,
        command_buffer,
        &kernels,
        elem_count,
        &buffer,
        value,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&buffer, elem_count)
}

#[test]
fn fill() {
    fn assert_fill<T: FillOp + Copy + std::fmt::Debug + PartialEq>(value: T) {
        for i in 0..4 {
            assert_eq!(run_fill(8 ^ i, value), vec![value; 8 ^ i]);
        }
    }
    assert_fill(123u8);
    assert_fill(456u32);
    assert_fill(789i64);
    assert_fill(f16::from_f32(1.23));
    assert_fill(bf16::from_f32(4.56));
    assert_fill(7.89f32);
}

fn run_random<T: Clone>(name: &'static str, seed: u32, length: usize, a: f32, b: f32) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();

    let options = MTLResourceOptions::StorageModeManaged;
    let output = device.new_buffer((length * core::mem::size_of::<T>()) as NSUInteger, options);

    let seed = device.new_buffer_with_data(
        &seed as *const u32 as *const core::ffi::c_void,
        std::mem::size_of::<u32>() as NSUInteger,
        options,
    );

    if name.starts_with("rand_uniform") {
        call_random_uniform(
            &device,
            command_buffer,
            &kernels,
            name,
            a,
            b,
            length,
            &seed,
            &output,
        )
        .unwrap();
    } else {
        call_random_normal(
            &device,
            command_buffer,
            &kernels,
            name,
            a,
            b,
            length,
            &seed,
            &output,
        )
        .unwrap();
    }
    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&output, length)
}

#[test]
fn random() {
    fn calc_mean(data: &[f32]) -> f32 {
        let sum = data.iter().sum::<f32>() as f32;
        let count = data.len();
        assert!(count > 0);
        sum / count as f32
    }

    fn calc_stddev(data: &[f32]) -> f32 {
        let mean = calc_mean(data);
        let count = data.len();
        assert!(count > 0);

        let variance = data
            .iter()
            .map(|value| {
                let diff = mean - (*value as f32);
                diff * diff
            })
            .sum::<f32>()
            / count as f32;

        variance.sqrt()
    }

    let shape = vec![1024, 10];

    let length = shape.iter().product::<usize>();
    let seed = 299792458;

    let min = -30.0;
    let max = 30.0;
    let mean = 100.0;
    let stddev = 50.0;

    macro_rules! validate_random {
        ($type:ty) => {
            let results: Vec<f32> = run_random::<$type>(
                concat!("rand_uniform_", stringify!($type)),
                seed,
                length,
                min,
                max,
            )
            .into_iter()
            .map(f32::from)
            .collect();
            results.iter().for_each(|v| {
                assert!(*v >= min && *v <= max);
            });
            assert!(calc_mean(&results) > -1.0 && calc_mean(&results) < 1.0);

            let results: Vec<f32> = run_random::<$type>(
                concat!("rand_normal_", stringify!($type)),
                seed,
                length,
                mean,
                stddev,
            )
            .into_iter()
            .map(f32::from)
            .collect();
            assert!((calc_mean(&results) - mean).abs() < mean / 10.0);
            assert!((calc_stddev(&results) - stddev).abs() < stddev / 10.0);
        };
    }

    validate_random!(f32);
    validate_random!(f16);
    validate_random!(bf16);
}
