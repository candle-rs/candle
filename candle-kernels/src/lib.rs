use phf::phf_map;

pub static KERNELS: phf::Map<&'static str, &'static str> = phf_map! {"AFFINE_86" => include_str!(concat!(env!("OUT_DIR"), "/affine_86.ptx")),
"BINARY_86" => include_str!(concat!(env!("OUT_DIR"), "/binary_86.ptx")),
"CAST_86" => include_str!(concat!(env!("OUT_DIR"), "/cast_86.ptx")),
"CONV_86" => include_str!(concat!(env!("OUT_DIR"), "/conv_86.ptx")),
"FILL_86" => include_str!(concat!(env!("OUT_DIR"), "/fill_86.ptx")),
"INDEXING_86" => include_str!(concat!(env!("OUT_DIR"), "/indexing_86.ptx")),
"REDUCE_86" => include_str!(concat!(env!("OUT_DIR"), "/reduce_86.ptx")),
"TERNARY_86" => include_str!(concat!(env!("OUT_DIR"), "/ternary_86.ptx")),
"UNARY_86" => include_str!(concat!(env!("OUT_DIR"), "/unary_86.ptx")),
"AFFINE_90" => include_str!(concat!(env!("OUT_DIR"), "/affine_90.ptx")),
"BINARY_90" => include_str!(concat!(env!("OUT_DIR"), "/binary_90.ptx")),
"CAST_90" => include_str!(concat!(env!("OUT_DIR"), "/cast_90.ptx")),
"CONV_90" => include_str!(concat!(env!("OUT_DIR"), "/conv_90.ptx")),
"FILL_90" => include_str!(concat!(env!("OUT_DIR"), "/fill_90.ptx")),
"INDEXING_90" => include_str!(concat!(env!("OUT_DIR"), "/indexing_90.ptx")),
"REDUCE_90" => include_str!(concat!(env!("OUT_DIR"), "/reduce_90.ptx")),
"TERNARY_90" => include_str!(concat!(env!("OUT_DIR"), "/ternary_90.ptx")),
"UNARY_90" => include_str!(concat!(env!("OUT_DIR"), "/unary_90.ptx")),
};
