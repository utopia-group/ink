pub const INPUT_PARAM: &str = "xs";
pub const INPUT_PARAM_1: &str = "ys";
pub const INPUT_PARAM_2: &str = "zs";

pub const NORMALIZER_ID: &[&str] = &[
    "h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10", "h11", "h12", "h13", "h14",
    "h15", "h16", "h17", "h18", "h19",
];
pub const DESTRUCTOR_ID: &[&str] = &[
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
];

pub const NORMALIZER_PARAM_1: &str = "s1";
pub const NORMALIZER_PARAM_2: &str = "s2";

pub const TESTING_INT_RANGE_MIN: i32 = -5;
pub const TESTING_INT_RANGE_MAX: i32 = 5;
pub const TESTING_COLLECTION_SIZE: usize = 5;

pub const CVC5_TIMEOUT_SECS: u64 = 60 * 10;

/// replace all non-alphanumeric characters with underscores
pub fn as_id_str(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

pub fn is_builtin_ids(s: &str) -> bool {
    matches!(
        s,
        "set_add"
            | "intersection"
            | "union"
            | "empty_set"
            | "filter_set"
            | "map_set"
            | "is_set_empty"
            | "concat_map"
            | "filter_values"
            | "map_values"
            | "contains_key"
    )
}

pub fn sygus_builtin_requires_type(f: &str) -> bool {
    matches!(
        f,
        "is_set_empty" | "concat_map" | "filter_values" | "map_values" | "contains_key"
    )
}
