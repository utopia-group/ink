#![allow(non_snake_case)]
use nameof::name_of;
use std::collections::HashMap;

use super::{benchmarks, Benchmark};
use rink::lang::{macros::*, Expr, Type};

pub fn benchmarks() -> HashMap<&'static str, Benchmark> {
    benchmarks![add]
}

pub fn add() -> Benchmark {
    let accumulator =
        (var!(s) + var!(x)).bind_params(vec![param!(s, Type::Num), param!(x, Type::Num)]);
    let init = 0.into();
    Benchmark {
        init,
        accumulator,
        is_homomorphic: Some(true),
    }
}
