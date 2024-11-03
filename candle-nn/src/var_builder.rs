//! A `VarBuilder` is used to retrieve variables used by a model. These variables can either come
//! from a pre-trained checkpoint, e.g. using `VarBuilder::from_mmaped_safetensors`, or initialized
//! for training, e.g. using `VarBuilder::from_varmap`.
use crate::VarMap;
use candle::{safetensors::Load, DType, Device, Error, Result, Shape, Tensor};
use safetensors::{slice::IndexOp, tensor::SafeTensors};
use std::collections::HashMap;
use std::sync::Arc;

/// A structure used to retrieve variables, these variables can either come from storage or be
/// generated via some form of initialization.
///
/// The way to retrieve variables is defined in the backend embedded in the `VarBuilder`.
pub struct VarBuilderArgs<'a, B: Backend> {
    data: Arc<TensorData<B>>,
    path: Vec<String>,
    pub dtype: DType,
    _phantom: std::marker::PhantomData<&'a B>,
}

impl<'a, B: Backend> Clone for VarBuilderArgs<'a, B> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            path: self.path.clone(),
            dtype: self.dtype,
            _phantom: self._phantom,
        }
    }
}

/// A simple `VarBuilder`, this is less generic than `VarBuilderArgs` but should cover most common
/// use cases.
pub type VarBuilder<'a> = VarBuilderArgs<'a, Box<dyn SimpleBackend + 'a>>;

struct TensorData<B: Backend> {
    backend: B,
    pub device: Device,
}

/// A trait that defines how tensor data is retrieved.
///
/// Typically this would use disk storage in some specific format, or random initialization.
/// Note that there is a specialized version of this trait (`SimpleBackend`) that can be used most
/// of the time. The main restriction is that it doesn't allow for specific args (besides
/// initialization hints).
pub trait Backend: Send + Sync {
    type Hints: Default;

    /// Retrieve a tensor with some target shape.
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: Self::Hints,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor>;

    fn contains_tensor(&self, name: &str) -> bool;
}

pub trait SimpleBackend: Send + Sync {
    /// Retrieve a tensor based on a target name and shape.
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: crate::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor>;

    fn contains_tensor(&self, name: &str) -> bool;
}

impl<'a> Backend for Box<dyn SimpleBackend + 'a> {
    type Hints = crate::Init;
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: Self::Hints,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        self.as_ref().get(s, name, h, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.as_ref().contains_tensor(name)
    }
}

impl<'a, B: Backend> VarBuilderArgs<'a, B> {
    pub fn new_with_args(backend: B, dtype: DType, dev: &Device) -> Self {
        let data = TensorData {
            backend,
            device: dev.clone(),
        };
        Self {
            data: Arc::new(data),
            path: vec![],
            dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns the prefix of the `VarBuilder`.
    pub fn prefix(&self) -> String {
        self.path.join(".")
    }

    /// Returns a new `VarBuilder` using the root path.
    pub fn root(&self) -> Self {
        Self {
            data: self.data.clone(),
            path: vec![],
            dtype: self.dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns a new `VarBuilder` with the prefix set to `prefix`.
    pub fn set_prefix(&self, prefix: impl ToString) -> Self {
        Self {
            data: self.data.clone(),
            path: vec![prefix.to_string()],
            dtype: self.dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Return a new `VarBuilder` adding `s` to the current prefix. This can be think of as `cd`
    /// into a directory.
    pub fn push_prefix<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
            path,
            dtype: self.dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Short alias for `push_prefix`.
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        self.push_prefix(s)
    }

    /// The device used by default.
    pub fn device(&self) -> &Device {
        &self.data.device
    }

    /// The dtype used by default.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Clone the VarBuilder tweaking its dtype
    pub fn to_dtype(&self, dtype: DType) -> Self {
        Self {
            data: self.data.clone(),
            path: self.path.clone(),
            dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    /// This returns true only if a tensor with the passed in name is available. E.g. when passed
    /// `a`, true is returned if `prefix.a` exists but false is returned if only `prefix.a.b`
    /// exists.
    pub fn contains_tensor(&self, tensor_name: &str) -> bool {
        let path = self.path(tensor_name);
        self.data.backend.contains_tensor(&path)
    }

    /// Retrieve the tensor associated with the given name at the current path.
    pub fn get_with_hints<S: Into<Shape>>(
        &self,
        s: S,
        name: &str,
        hints: B::Hints,
    ) -> Result<Tensor> {
        self.get_with_hints_dtype(s, name, hints, self.dtype)
    }

    /// Retrieve the tensor associated with the given name at the current path.
    pub fn get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<Tensor> {
        self.get_with_hints(s, name, Default::default())
    }

    /// Retrieve the tensor associated with the given name & dtype at the current path.
    pub fn get_with_hints_dtype<S: Into<Shape>>(
        &self,
        s: S,
        name: &str,
        hints: B::Hints,
        dtype: DType,
    ) -> Result<Tensor> {
        let path = self.path(name);
        self.data
            .backend
            .get(s.into(), &path, hints, dtype, &self.data.device)
    }
}

struct Zeros;

impl SimpleBackend for Zeros {
    fn get(&self, s: Shape, _: &str, _: crate::Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        Tensor::zeros(s, dtype, dev)
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

impl SimpleBackend for HashMap<String, Tensor> {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: crate::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let tensor = self
            .get(name)
            .ok_or_else(|| {
                Error::CannotFindTensor {
                    path: name.to_string(),
                }
                .bt()
            })?
            .clone();
        if tensor.shape() != &s {
            Err(candle::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        tensor.to_device(dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.contains_key(name)
    }
}

impl SimpleBackend for VarMap {
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: crate::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        VarMap::get(self, s, name, h, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.data().lock().unwrap().contains_key(name)
    }
}

#[allow(dead_code)]
pub struct SafeTensorWithRouting<'a> {
    routing: HashMap<String, usize>,
    safetensors: Vec<SafeTensors<'a>>,
}

impl<'a> SimpleBackend for SafeTensorWithRouting<'a> {
    fn get(
        &self,
        s: Shape,
        path: &str,
        _: crate::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let index = self.routing.get(path).ok_or_else(|| {
            Error::CannotFindTensor {
                path: path.to_string(),
            }
            .bt()
        })?;
        let tensor = self.safetensors[*index]
            .tensor(path)?
            .load(dev)?
            .to_dtype(dtype)?;
        if tensor.shape() != &s {
            Err(candle::Error::UnexpectedShape {
                msg: format!("shape mismatch for {path}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.routing.contains_key(name)
    }
}

impl SimpleBackend for candle::npy::NpzTensors {
    fn get(
        &self,
        s: Shape,
        path: &str,
        _: crate::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let tensor = match self.get(path)? {
            None => Err(Error::CannotFindTensor {
                path: path.to_string(),
            }
            .bt())?,
            Some(tensor) => tensor,
        };
        let tensor = tensor.to_device(dev)?.to_dtype(dtype)?;
        if tensor.shape() != &s {
            Err(candle::Error::UnexpectedShape {
                msg: format!("shape mismatch for {path}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.get(name).map_or(false, |v| v.is_some())
    }
}

impl SimpleBackend for candle::pickle::PthTensors {
    fn get(
        &self,
        s: Shape,
        path: &str,
        _: crate::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let tensor = match self.get(path)? {
            None => Err(Error::CannotFindTensor {
                path: path.to_string(),
            }
            .bt())?,
            Some(tensor) => tensor,
        };
        let tensor = tensor.to_device(dev)?.to_dtype(dtype)?;
        if tensor.shape() != &s {
            Err(candle::Error::UnexpectedShape {
                msg: format!("shape mismatch for {path}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.get(name).map_or(false, |v| v.is_some())
    }
}

impl SimpleBackend for candle::safetensors::MmapedSafetensors {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: crate::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let tensor = self.load(name, dev)?.to_dtype(dtype)?;
        if tensor.shape() != &s {
            Err(candle::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.get(name).is_ok()
    }
}

impl SimpleBackend for candle::safetensors::BufferedSafetensors {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: crate::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let tensor = self.load(name, dev)?.to_dtype(dtype)?;
        if tensor.shape() != &s {
            Err(candle::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.get(name).is_ok()
    }
}

impl<'a> SimpleBackend for candle::safetensors::SliceSafetensors<'a> {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: crate::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let tensor = self.load(name, dev)?.to_dtype(dtype)?;
        if tensor.shape() != &s {
            Err(candle::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.get(name).is_ok()
    }
}

impl<'a> VarBuilder<'a> {
    /// Initializes a `VarBuilder` using a custom backend.
    ///
    /// It is preferred to use one of the more specific constructors. This
    /// constructor is provided to allow downstream users to define their own
    /// backends.
    pub fn from_backend(
        backend: Box<dyn SimpleBackend + 'a>,
        dtype: DType,
        device: Device,
    ) -> Self {
        let data = TensorData { backend, device };
        Self {
            data: Arc::new(data),
            path: vec![],
            dtype,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Initializes a `VarBuilder` that uses zeros for any tensor.
    pub fn zeros(dtype: DType, dev: &Device) -> Self {
        Self::from_backend(Box::new(Zeros), dtype, dev.clone())
    }

    /// Initializes a `VarBuilder` that retrieves tensors stored in a hashtable. An error is
    /// returned if no tensor is available under the requested path or on shape mismatches.
    pub fn from_tensors(ts: HashMap<String, Tensor>, dtype: DType, dev: &Device) -> Self {
        Self::from_backend(Box::new(ts), dtype, dev.clone())
    }

    /// Initializes a `VarBuilder` using a `VarMap`. The requested tensors are created and
    /// initialized on new paths, the same tensor is used if the same path is requested multiple
    /// times. This is commonly used when initializing a model before training.
    ///
    /// Note that it is possible to load the tensor values after model creation using the `load`
    /// method on `varmap`, this can be used to start model training from an existing checkpoint.
    pub fn from_varmap(varmap: &VarMap, dtype: DType, dev: &Device) -> Self {
        Self::from_backend(Box::new(varmap.clone()), dtype, dev.clone())
    }

    /// Initializes a `VarBuilder` that retrieves tensors stored in a collection of safetensors
    /// files.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`candle::safetensors::MmapedSafetensors::multi`].
    pub unsafe fn from_mmaped_safetensors<P: AsRef<std::path::Path>>(
        paths: &[P],
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let tensors = candle::safetensors::MmapedSafetensors::multi(paths)?;
        Ok(Self::from_backend(Box::new(tensors), dtype, dev.clone()))
    }

    /// Initializes a `VarBuilder` from a binary buffer in the safetensor format.
    pub fn from_buffered_safetensors(data: Vec<u8>, dtype: DType, dev: &Device) -> Result<Self> {
        let tensors = candle::safetensors::BufferedSafetensors::new(data)?;
        Ok(Self::from_backend(Box::new(tensors), dtype, dev.clone()))
    }

    /// Initializes a `VarBuilder` from a binary slice in the safetensor format.
    pub fn from_slice_safetensors(data: &'a [u8], dtype: DType, dev: &Device) -> Result<Self> {
        let tensors = candle::safetensors::SliceSafetensors::new(data)?;
        Ok(Self::from_backend(Box::new(tensors), dtype, dev.clone()))
    }

    /// Initializes a `VarBuilder` that retrieves tensors stored in a numpy npz file.
    pub fn from_npz<P: AsRef<std::path::Path>>(p: P, dtype: DType, dev: &Device) -> Result<Self> {
        let npz = candle::npy::NpzTensors::new(p)?;
        Ok(Self::from_backend(Box::new(npz), dtype, dev.clone()))
    }

    /// Initializes a `VarBuilder` that retrieves tensors stored in a pytorch pth file.
    pub fn from_pth<P: AsRef<std::path::Path>>(p: P, dtype: DType, dev: &Device) -> Result<Self> {
        let pth = candle::pickle::PthTensors::new(p, None)?;
        Ok(Self::from_backend(Box::new(pth), dtype, dev.clone()))
    }

    /// Gets a VarBuilder that applies some renaming function on tensor it gets queried for before
    /// passing the new names to the inner VarBuilder.
    ///
    /// ```rust
    /// use candle::{Tensor, DType, Device};
    ///
    /// let a = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    /// let tensors: std::collections::HashMap<_, _> = [
    ///     ("foo".to_string(), a),
    /// ]
    /// .into_iter()
    /// .collect();
    /// let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    /// assert!(vb.contains_tensor("foo"));
    /// assert!(vb.get((2, 3), "foo").is_ok());
    /// assert!(!vb.contains_tensor("bar"));
    /// let vb = vb.rename_f(|f: &str| if f == "bar" { "foo".to_string() } else { f.to_string() });
    /// assert!(vb.contains_tensor("bar"));
    /// assert!(vb.contains_tensor("foo"));
    /// assert!(vb.get((2, 3), "bar").is_ok());
    /// assert!(vb.get((2, 3), "foo").is_ok());
    /// assert!(!vb.contains_tensor("baz"));
    /// # Ok::<(), candle::Error>(())
    /// ```
    pub fn rename_f<F: Fn(&str) -> String + Sync + Send + 'static>(self, f: F) -> Self {
        let f: Box<dyn Fn(&str) -> String + Sync + Send + 'static> = Box::new(f);
        self.rename(f)
    }

    pub fn rename<R: Renamer + Send + Sync + 'a>(self, renamer: R) -> Self {
        let dtype = self.dtype();
        let device = self.device().clone();
        let path = self.path.clone();
        let backend = Rename::new(self, renamer);
        let backend: Box<dyn SimpleBackend + 'a> = Box::new(backend);
        let data = TensorData { backend, device };
        Self {
            data: Arc::new(data),
            dtype,
            path,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct ShardedSafeTensors(candle::safetensors::MmapedSafetensors);

pub type ShardedVarBuilder<'a> = VarBuilderArgs<'a, ShardedSafeTensors>;

impl ShardedSafeTensors {
    /// Initializes a `VarBuilder` that retrieves tensors stored in a collection of safetensors
    /// files and make them usable in a sharded way.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`candle::safetensors::MmapedSafetensors::multi`].
    pub unsafe fn var_builder<P: AsRef<std::path::Path>>(
        paths: &[P],
        dtype: DType,
        dev: &Device,
    ) -> Result<ShardedVarBuilder<'static>> {
        let tensors = candle::safetensors::MmapedSafetensors::multi(paths)?;
        let backend = ShardedSafeTensors(tensors);
        Ok(VarBuilderArgs::new_with_args(backend, dtype, dev))
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Shard {
    pub dim: usize,
    pub rank: usize,
    pub world_size: usize,
}

impl Default for Shard {
    fn default() -> Self {
        Self {
            dim: 0,
            rank: 0,
            world_size: 1,
        }
    }
}

/// Get part of a tensor, typically used to do Tensor Parallelism sharding.
///
/// If the tensor is of size (1024, 1024).
///
/// `dim` corresponds to the dimension to slice into
/// `rank` is the rank of the current process
/// `world_size` is the total number of ranks in the process group
///
/// `get_sharded("tensor", 0, 0, 2)` means `tensor.i((..512))`
/// `get_sharded("tensor", 0, 1, 2)` means `tensor.i((512..))`
/// `get_sharded("tensor", 1, 0, 2)` means `tensor.i((.., ..512))`
impl Backend for ShardedSafeTensors {
    type Hints = Shard;

    fn get(
        &self,
        target_shape: Shape, // The size is only checked when the world size is 1.
        path: &str,
        h: Self::Hints,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        if h.world_size == 1 {
            // There is no sharding to be applied here so we use the default backend to speed
            // things up.
            return SimpleBackend::get(&self.0, target_shape, path, Default::default(), dtype, dev);
        }

        let Shard {
            dim,
            rank,
            world_size,
        } = h;
        let view = self.0.get(path)?;
        let view_dtype = view.dtype();
        let mut shape = view.shape().to_vec();
        let size = shape[dim];

        if size % world_size != 0 {
            return Err(Error::ShapeMismatchSplit {
                shape: shape.into(),
                dim,
                n_parts: world_size,
            });
        }
        let block_size = size / world_size;
        let start = rank * block_size;
        let stop = (rank + 1) * block_size;

        // Everything is expressed in tensor dimension
        // bytes offsets is handled automatically for safetensors.

        let iterator = if dim == 0 {
            view.slice(start..stop).map_err(|_| {
                Error::Msg(format!(
                    "Cannot slice tensor {path} ({shape:?} along dim {dim} with {start}..{stop}"
                ))
            })?
        } else if dim == 1 {
            view.slice((.., start..stop)).map_err(|_| {
                Error::Msg(format!(
                    "Cannot slice tensor {path} ({shape:?} along dim {dim} with {start}..{stop}"
                ))
            })?
        } else {
            candle::bail!("Get sharded on dimensions != 0 or 1")
        };

        shape[dim] = block_size;

        let view_dtype: DType = view_dtype.try_into()?;
        let raw: Vec<u8> = iterator.into_iter().flatten().cloned().collect();
        Tensor::from_raw_buffer(&raw, view_dtype, &shape, dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.0.get(name).is_ok()
    }
}

/// This traits specifies a way to rename the queried names into names that are stored in an inner
/// VarBuilder.
pub trait Renamer {
    /// This is applied to the name obtained by a name call and the resulting name is passed to the
    /// inner VarBuilder.
    fn rename(&self, v: &str) -> std::borrow::Cow<'_, str>;
}

pub struct Rename<'a, R: Renamer> {
    inner: VarBuilder<'a>,
    renamer: R,
}

impl<'a, R: Renamer + Sync + Send> SimpleBackend for Rename<'a, R> {
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: crate::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let name = self.renamer.rename(name);
        self.inner
            .get_with_hints_dtype(s, &name, h, dtype)?
            .to_device(dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        let name = self.renamer.rename(name);
        self.inner.contains_tensor(&name)
    }
}

impl<'a, R: Renamer> Rename<'a, R> {
    pub fn new(inner: VarBuilder<'a>, renamer: R) -> Self {
        Self { inner, renamer }
    }
}

impl Renamer for Box<dyn Fn(&str) -> String + Sync + Send> {
    fn rename(&self, v: &str) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Owned(self(v))
    }
}
