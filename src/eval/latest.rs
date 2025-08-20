#![allow(non_snake_case)]
use nameof::name_of;
use std::collections::HashMap;

use super::{benchmarks, Benchmark};
use rink::lang::{macros::*, BinOpKinds, Expr, IsCurriedFunction, Type};

pub fn benchmarks() -> HashMap<&'static str, Benchmark> {
    benchmarks![
        AggRowFirst,
        CollectList,
        CombineMaps,
        EnsembleByKey,
        GroupConcatDistinctUDAF,
        HyperLogLogAggregationFunction,
        LocalStatsAggregate,
        MinPooling,
        OHLCAggregator,
        PaymentsAggregateFunction,
        RasterFunction,
        TermFrequencyAccumulator,
        TopUdaf,
        UnionSketchUDAF,
        VectorSumUDAF
    ]
}

/// https://github.com/baosince/daily-learning/blob/e7618f56aff526c181c7abcf08f863145f15a8fd/spark16/src/main/scala/sql/udaf/demo1/CombineMaps.scala
fn CombineMaps() -> Benchmark {
    /*
    f s m = concat_map(s, m)
    h a b = concat_map(a, b)
     */
    let accumulator = var!(concat_map)
        .call(vec![var!(s), var!(m)])
        .bind_params(vec![
            param!(s, tmap!(Type::Num, Type::Num)),
            param!(m, tmap!(Type::Num, Type::Num)),
        ]);
    let init = var!(empty_map);
    Benchmark {
        init,
        accumulator,
        is_homomorphic: Some(true),
    }
}

/// https://github.com/SonarSource/slang-test-sources/blob/062fc80f0528d5379be76a4c12d2a16d518728bd/scala/mmlspark/src/ensemble/src/main/scala/EnsembleByKey.scala
fn EnsembleByKey() -> Benchmark {
    /*
       f (sv, sn) vec = (map(+, zip(sv, vec)), sn + 1)
    */

    let accumulator = tuple![
        var!(map).call(vec![
            lambda!(p: ttuple!(Type::Num, Type::Num) =>
                    tuple_access!(var!(p), 0) + tuple_access!(var!(p), 1)
            ),
            var!(zip).call(vec![tuple_access!(var!(s), 0), var!(x)]),
        ]),
        tuple_access!(var!(s), 1) + 1.into(),
    ]
    .bind_params(vec![
        (sym!(s), ttuple![tlist!(Type::Num), Type::Num]),
        (sym!(x), tlist!(Type::Num)),
    ]);

    let init = tuple![vec![0; 10].into(), 0.into()];
    Benchmark {
        init,
        accumulator,
        is_homomorphic: Some(true),
    }
}

/// https://github.com/okmich/fx-usd-non-farm-payroll/blob/f54a3bad1c597b0b00d0d14096753de47ba2e833/OHLCAggregator.scala
fn OHLCAggregator() -> Benchmark {
    /*
       init = ("", -1)
       x_old: tup(str, num, num, num, num, num, num, num, num)
       x_new: tup(str, num, num, num, num, num, num, num, num, num, num, num)
       s: tup(str, num, num, num)
       x_new[0..9] = x_old
       x_new[9] = getNfpTs(x_old)
       x_new[10] = getTickTs(x_old)
       x_new[11] = tsPlusMins(x_old)
       c x_new = x_new[10]>=0               (constraint: all date values >= 0)
       f s x = if (c x_new) and x_new[10]>x_new[9] and x_new[10]<x_new[11] then (x_new[7], x_new[10]) else s
       h a b = if (b[1]>=0) then b else a
    */
    let state_type = ttuple!(Type::Num, ttuple!(Type::Num, Type::Num));
    let input_type = ttuple!(
        Type::Str,
        Type::Num,
        Type::Num,
        Type::Num,
        Type::Num,
        Type::Num,
        Type::Num,
        Type::Num,
        Type::Num
    );

    /*
       f s x = ite(x.5 < x.8 && x.6 < x.8, (x.7, (x.1, x.2)), s)
    */
    let accumulator = ite!(
        expr_and!(
            expr_lt!(tuple_access!(var!(x), 5), tuple_access!(var!(x), 8)),
            expr_lt!(tuple_access!(var!(x), 6), tuple_access!(var!(x), 8))
        ),
        tuple![
            tuple_access!(var!(x), 7),
            tuple![tuple_access!(var!(x), 1), tuple_access!(var!(x), 2)]
        ],
        var!(s)
    )
    .bind_params(vec![
        param!(s, state_type.clone()),
        param!(x, input_type.clone()),
    ]);
    let init = state_type.default_value().into();
    Benchmark {
        init,
        accumulator,
        is_homomorphic: Some(false),
    }
}

/// https://github.com/mfoglino/java-experiments/blob/64f661d91fc653dcb438917e3a97f071f94f665f/flink-aggregation-examples-job/src/main/java/PaymentsAggregateFunction.java
fn PaymentsAggregateFunction() -> Benchmark {
    // input: (status, amount)
    // status: 0 - approved, 1 - declined, 2 - cancelled
    // output: (timestamp, num_approved, num_declined, num_cancelled, approved amount, declined amount, cancelled amount)
    let accumulator = tuple!(
        tuple_access!(var!(s), 0),
        tuple_access!(var!(s), 1)
            + ite!(
                expr_eq!(tuple_access!(var!(x), 0), 0.into()),
                1.into(),
                0.into()
            ),
        tuple_access!(var!(s), 2)
            + ite!(
                expr_eq!(tuple_access!(var!(x), 0), 1.into()),
                1.into(),
                0.into()
            ),
        tuple_access!(var!(s), 3)
            + ite!(
                expr_eq!(tuple_access!(var!(x), 0), 2.into()),
                1.into(),
                0.into()
            ),
        tuple_access!(var!(s), 4)
            + ite!(
                expr_eq!(tuple_access!(var!(x), 0), 0.into()),
                tuple_access!(var!(x), 1),
                0.into()
            ),
        tuple_access!(var!(s), 5)
            + ite!(
                expr_eq!(tuple_access!(var!(x), 0), 1.into()),
                tuple_access!(var!(x), 1),
                0.into()
            ),
        tuple_access!(var!(s), 6)
            + ite!(
                expr_eq!(tuple_access!(var!(x), 0), 2.into()),
                tuple_access!(var!(x), 1),
                0.into()
            ),
    )
    .bind_params(vec![
        param!(
            s,
            ttuple!(
                Type::Num,
                Type::Num,
                Type::Num,
                Type::Num,
                Type::Num,
                Type::Num,
                Type::Num
            )
        ),
        param!(x, ttuple!(Type::Num, Type::Num)),
    ]);

    let init = tuple!(
        0.into(),
        0.into(),
        0.into(),
        0.into(),
        0.into(),
        0.into(),
        0.into()
    );
    Benchmark {
        init,
        accumulator,
        is_homomorphic: Some(true),
    }
}

/// https://github.com/zdkzdk/aaocp/blob/fef227114592ac9a261f3477a93e8407127a643d/top3product_by_area/src/main/scala/GroupConcatDistinctUDAF.scala
fn GroupConcatDistinctUDAF() -> Benchmark {
    /*
       f s x = set_add(s, x)
       h a b = set_union(a, b)
    */
    let accumulator = var!(set_add)
        .call(vec![var!(x), var!(s)])
        .bind_params(vec![param!(s, tset!(Type::Num)), param!(x, Type::Num)]);

    let init = var!(empty_set);
    Benchmark {
        init,
        accumulator,
        is_homomorphic: Some(true),
    }
}

/// https://github.com/Elkoumy/StreamCardinalityEstimation/blob/6bfb7e1bc716aa1825f8b4acc8813c5e5b89e1ac/CardinalityBenchmarking/src/main/java/ee/ut/cs/dsg/StreamCardinality/ApproximateCardinalityAggregateFunction/HyperLogLogAggregationFunction.java

fn HyperLogLogAggregationFunction() -> Benchmark {
    // input: (x: num, y: num, z: num)
    // output: (latest x, latest y, sets of z)

    let input_type = ttuple!(Type::Num, Type::Num, Type::Num);
    let state_type = ttuple!(Type::Num, Type::Num, tset!(Type::Num));
    let accumulator = tuple!(
        tuple_access!(var!(x), 0),
        tuple_access!(var!(x), 1),
        var!(set_add).call(vec![tuple_access!(var!(x), 2), tuple_access!(var!(s), 2)]),
    )
    .bind_params(vec![param!(s, state_type), param!(x, input_type)]);
    let init = tuple!(0.into(), 0.into(), var!(empty_set));
    Benchmark {
        init,
        accumulator,
        is_homomorphic: Some(true),
    }
}

/// https://github.com/feathr-ai/feathr/blob/45e44afc1ebd3abc0fa8313aac21db7b1f05580a/feathr-impl/src/main/scala/com/linkedin/feathr/offline/generation/aggregations/MinPooling.scala
fn MinPooling() -> Benchmark {
    let state_type = tlist!(Type::Num);
    let input_type = tlist!(Type::Num);
    let accumulator = var!(map)
        .call(vec![
            var!(min)
                .call(vec![tuple_access!(var!(p), 0), tuple_access!(var!(p), 1)])
                .bind_params(vec![param!(p, ttuple!(Type::Num, Type::Num))]),
            var!(zip).call(vec![var!(s), var!(x)]),
        ])
        .bind_params(vec![param!(s, state_type), param!(x, input_type)]);
    let init = vec![var!(_mx); 10].into();

    Benchmark {
        init,
        accumulator,
        is_homomorphic: Some(true),
    }
}

/// https://github.com/locationtech/rasterframes/blob/45c6a1a05638c682bec9fb521dc1b5ac13a5194e/core/src/main/scala/org/locationtech/rasterframes/expressions/aggregates/LocalStatsAggregate.scala
fn LocalStatsAggregate() -> Benchmark {
    // count, min, max, sum, sumSqr
    let state_type = ttuple!(Type::Num, Type::Num, Type::Num, Type::Num, Type::Num);
    let input_type = Type::Num;

    let accumulator = tuple!(
        tuple_access!(var!(s), 0) + 1.into(),
        var!(min).call(vec![tuple_access!(var!(s), 1), var!(x)]),
        var!(max).call(vec![tuple_access!(var!(s), 2), var!(x)]),
        tuple_access!(var!(s), 3) + var!(x),
        tuple_access!(var!(s), 4) + var!(x) * var!(x),
    )
    .bind_params(vec![param!(s, state_type), param!(x, input_type)]);
    let init = tuple!(0.into(), var!(_mx), var!(_mn), 0.into(), 0.into());
    Benchmark {
        init,
        accumulator,
        is_homomorphic: Some(true),
    }
}

// https://github.com/dbis-ilm/stark/blob/8efa2e04617a81b1de65b91267ec8a0ce77af899/src/main/scala/dbis/stark/sql/raster/RasterFunction.scala
fn RasterFunction() -> Benchmark {
    /*
        bucket: (values: [num], lowerbound: num, upperbound: num)
        x: [bucket]
        s: [bucket]
        init = []
        f s x =
            if len(x)==10
            then
               map(|row0, row1| (row0.0 ++ row1.0, row1.1, row1.2), zip(s, x))
            else s

        Note: f is not a homomorphism because of the inner degenerate program
        h a b = map(|row0, row1| (row0.0 ++ row1.0, row1.1, row1.2), zip(a, b))
    */
    let bucket_t = ttuple!(tlist!(Type::Num), Type::Num, Type::Num);
    let map_body = tuple!(
        binop!(
            BinOpKinds::Concat,
            tuple_access!(tuple_access!(var!(zipped_row), 0), 0),
            tuple_access!(tuple_access!(var!(zipped_row), 1), 0)
        ),
        tuple_access!(tuple_access!(var!(zipped_row), 1), 1),
        tuple_access!(tuple_access!(var!(zipped_row), 1), 2)
    )
    .bind_params(vec![param!(
        zipped_row,
        ttuple!(bucket_t.clone(), bucket_t.clone())
    )]);
    let map_call: Expr = var!(map).call(vec![map_body, var!(zip).call(vec![var!(s), var!(x)])]);

    // we need to implement an ITE rule but for now just ignore the condition
    let _constraint = binop!(BinOpKinds::Eq, var!(length).call(vec![var!(x)]), 10.into());
    let accumulator = map_call.bind_params(vec![
        param!(s, tlist!(bucket_t.clone())),
        param!(x, tlist!(bucket_t.clone())),
    ]);
    let init = vec![tuple!(vec![Expr::Num(0); 1].into(), 0.into(), 0.into()); 1].into();

    Benchmark {
        init,
        accumulator,
        is_homomorphic: Some(false),
    }
}
/// https://github.com/mozilla/telemetry-batch-view/blob/d78f3bba1dafb37caae56d4a5aa7c4fadcaa56f0/src/main/scala/com/mozilla/telemetry/utils/udfs/AggRowFirst.scala
fn AggRowFirst() -> Benchmark {
    /*
        idIndex: num
        x: tup(num, num)
        xs: list(tup(num, num))
        s: map(num, tup(num, num))

        free-vars = ()
        set-vars = (id_index := 0)
        init = empty-map
        f s xs = foldl-list(|m, x| assign(m, x[idIndex]->row), s, xs)
        h a b = concat-map(a, b)
    */
    let x_type = ttuple![Type::Num, Type::Num];
    let xs_type = tlist![x_type.clone()];
    let s_type = tmap!(Type::Num, ttuple![Type::Num, Type::Num]);
    let id_index = 0;

    let acc = ite!(
        !var!(contains_key).call(vec![var!(m), tuple_access!(var!(x), id_index)]),
        map_assign!(var!(m), tuple_access!(var!(x), id_index), var!(x)),
        var!(m)
    )
    .bind_params(vec![param!(m, s_type.clone()), param!(x, x_type.clone())]);

    let f = var!(foldl)
        .call(vec![acc, var!(map_output), var!(xs)])
        .bind_params(vec![
            param!(map_output, s_type.clone()),
            param!(xs, xs_type),
        ]);

    let init = var!(empty_map);
    Benchmark {
        init,
        accumulator: f,
        is_homomorphic: Some(true),
    }
}

/// https://github.com/mozilla/telemetry-batch-view/blob/d78f3bba1dafb37caae56d4a5aa7c4fadcaa56f0/src/main/scala/com/mozilla/telemetry/utils/udfs/CollectList.scala
fn CollectList() -> Benchmark {
    // ASSUME: type(x) = tuple(num, num, num)
    /*
       init = ([], [], [], ...)
       f s x = (s.0 + x.0, s.1 + x.1, s.2 + x.2, ...)
       h a b = (a.0 ++ b.0, a.1 ++ b.1, a.2 ++ b.2, ...)
    */
    let res1 = binop!(
        BinOpKinds::Concat,
        tuple_access!(var!(map_output), 0),
        cons!(tuple_access!(var!(x), 0), vec![0; 0].into())
    );
    let res2 = binop!(
        BinOpKinds::Concat,
        tuple_access!(var!(map_output), 1),
        cons!(tuple_access!(var!(x), 1), vec![0; 0].into())
    );
    let res3 = binop!(
        BinOpKinds::Concat,
        tuple_access!(var!(map_output), 2),
        cons!(tuple_access!(var!(x), 2), vec![0; 0].into())
    );
    let accumulator = tuple![res1, res2, res3].bind_params(vec![
        param!(
            map_output,
            ttuple![tlist![Type::Num], tlist![Type::Num], tlist![Type::Num]]
        ),
        param!(x, ttuple![Type::Num, Type::Num, Type::Num]),
    ]);

    let init = tuple!(Expr::Nil, Expr::Nil, Expr::Nil);
    Benchmark {
        init,
        accumulator,
        is_homomorphic: Some(true),
    }
}

/// https://github.com/SamSJackson/bigdata-ae/blob/3e2f3138f107440a5eb0eb87810feae363ac12ce/src/uk/ac/gla/dcs/bigdata/studentstructures/TermFrequencyAccumulator.java
fn TermFrequencyAccumulator() -> Benchmark {
    /*
    f s xs = foldl(λ(acc, x) => {
        acc[x._0 <- acc[x._0] + x._1]
    }) s xs
     */

    let x_type = ttuple![Type::Num, Type::Num];
    let xs_type = tlist![x_type.clone()];
    let s_type = tmap!(Type::Num, Type::Num);
    let acc = map_assign!(
        var!(m),
        tuple_access!(var!(x), 0),
        map_access!(var!(m), tuple_access!(var!(x), 0)) + tuple_access!(var!(x), 1)
    )
    .bind_params(vec![param!(m, s_type.clone()), param!(x, x_type.clone())]);
    let f = var!(foldl)
        .call(vec![acc, var!(map_output), var!(xs)])
        .bind_params(vec![
            param!(map_output, s_type.clone()),
            param!(xs, xs_type),
        ]);
    let init = var!(empty_map);
    Benchmark {
        init,
        accumulator: f,
        is_homomorphic: Some(true),
    }
}

/// https://github.com/ytsaurus/ytsaurus-spyt/blob/72b9d2fd6952ddc39948f588f049612de40d32fe/data-source/src/main/scala/tech/ytsaurus/spyt/common/utils/TopUdaf.scala
fn TopUdaf() -> Benchmark {
    let accumulator = ite!(
        expr_lt!(tuple_access!(var!(x), 0), tuple_access!(var!(s), 0)),
        var!(x),
        var!(s)
    )
    .bind_params(vec![
        param!(
            s,
            ttuple![
                Type::Num,
                Type::Num,
                Type::Num,
                Type::Num,
                Type::Num,
                Type::Num,
                Type::Num
            ]
        ),
        param!(
            x,
            ttuple![
                Type::Num,
                Type::Num,
                Type::Num,
                Type::Num,
                Type::Num,
                Type::Num,
                Type::Num
            ]
        ),
    ]);
    let init = tuple!(
        var!(_mx),
        0.into(),
        0.into(),
        0.into(),
        0.into(),
        0.into(),
        0.into()
    );
    Benchmark {
        init,
        accumulator,
        is_homomorphic: Some(true),
    }
}

/// https://github.com/sanxore/spark-theta-sketch-udfs/blob/c1b648eb2145196575a2d8b9d93f50b3b30fae85/src/main/scala/com/sketches/spark/theta/udaf/UnionSketchUDAF.scala
fn UnionSketchUDAF() -> Benchmark {
    let accumulator = var!(union).call(vec![var!(s), var!(x)]).bind_params(vec![
        param!(s, tset!(Type::Num)),
        param!(x, tset!(Type::Num)),
    ]);
    let init = var!(empty_set);
    Benchmark {
        init,
        accumulator,
        is_homomorphic: Some(true),
    }
}

/// https://github.com/uhh-lt/wsd/blob/50311290ad559b207b010ff808cb9d9f69166899/spark/src/main/scala/de/tudarmstadt/lt/wsd/pipeline/sql/VectorSumUDAF.scala
fn VectorSumUDAF() -> Benchmark {
    /*
       f (sv, sn) vec = (map(+, zip(sv, vec)), sn + 1)
    */

    let x = var!(x);
    let s_list = tuple_access!(var!(s), 0);
    let s_n = tuple_access!(var!(s), 1);

    let accumulator = tuple![
        var!(map).call(vec![
            lambda!(p: ttuple!(Type::Num, Type::Num) =>
                binop!(
                    BinOpKinds::Add,
                    tuple_access!(var!(p), 0),
                    tuple_access!(var!(p), 1)
                )
            ),
            var!(zip).call(vec![s_list.clone(), x.clone()]),
        ]),
        binop!(BinOpKinds::Add, s_n, 1.into()),
    ]
    .bind_params(vec![
        (sym!(s), ttuple![tlist!(Type::Num), Type::Num]),
        (sym!(x), tlist!(Type::Num)),
    ]);

    let init = tuple![vec![0; 5].into(), 0.into()];
    Benchmark {
        init,
        accumulator,
        is_homomorphic: Some(true),
    }
}
