# Training


Training starts with data. We're going to use the huggingface hub and
start with the Hello world dataset of machine learning, MNIST.

Let's start with downloading `MNIST` from [huggingface](https://huggingface.co/datasets/mnist).

This requires [`hf-hub`](https://github.com/huggingface/hf-hub).
```bash
cargo add hf-hub
```

This is going to be very hands-on for now.

```rust,ignore
{{#include ../lib.rs:book_training_1}}
```

This uses the standardized `parquet` files from the `refs/convert/parquet` branch on every dataset.
Our handles are now [`parquet::file::serialized_reader::SerializedFileReader`].

We can inspect the content of the files with:

```rust,ignore
{{#include ../lib.rs:book_training_2}}
```

You should see something like:

```bash
Column id 1, name label, value 6
Column id 0, name image, value {bytes: [137, ....]
Column id 1, name label, value 8
Column id 0, name image, value {bytes: [137, ....]
```

So each row contains 2 columns (image, label) with image being saved as bytes.
Let's put them into a useful struct.
