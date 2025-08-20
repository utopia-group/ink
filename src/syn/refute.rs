use crate::consts::{TESTING_COLLECTION_SIZE, TESTING_INT_RANGE_MAX, TESTING_INT_RANGE_MIN};
use crate::lang::{
    create_env, eval, infer, macros::*, BinOpKinds, Env, Expr, IsCurriedFunction, Type, TypeEnv,
    UnaryOpKinds, Value,
};
use proptest::prelude::{prop, BoxedStrategy, TestCaseError};
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRng, TestRunner};
use tracing::{event, Level};

use super::FunctionHasFreeVar;

fn create_config() -> Config {
    Config {
        cases: 1000,
        failure_persistence: None,
        max_shrink_iters: 0,
        ..Config::default()
    }
}

fn create_runner() -> TestRunner {
    let config = create_config();
    let algorithm = config.rng_algorithm;
    TestRunner::new_with_rng(config, TestRng::deterministic_rng(algorithm))
}

#[derive(Debug, Clone)]
struct StateValue(Value);
#[derive(Debug, Clone)]
struct InputValue(Value);

fn create_testing_env() -> Env {
    let mut env = create_env();
    env.push((sym!(_mn), TESTING_INT_RANGE_MIN.into()));
    env.push((sym!(_mx), TESTING_INT_RANGE_MAX.into()));
    env
}

fn prop(prop: Expr) -> impl Fn((StateValue, InputValue)) -> Result<(), TestCaseError> {
    move |(s, x)| {
        let mut env = create_testing_env();
        env.push((sym!(_s), s.0));
        env.push((sym!(_x), x.0));

        let result = eval(&prop, &mut env).unwrap();
        if matches!(result, Value::Bool(true)) {
            Ok(())
        } else if matches!(result, Value::Bool(false)) {
            Err(TestCaseError::fail("refutation succeeded"))
        } else {
            Err(TestCaseError::Fail(
                format!("unexpected result: {:?}", result).into(),
            ))
        }
    }
}

fn prop_two_input(
    prop: Expr,
) -> impl Fn((StateValue, InputValue, InputValue)) -> Result<(), TestCaseError> {
    move |(s, x1, x2)| {
        let mut env: Vec<(egg::Symbol, Value)> = create_testing_env();
        env.push((sym!(_s), s.0));
        env.push((sym!(_x1), x1.0));
        env.push((sym!(_x2), x2.0));

        let result = eval(&prop, &mut env).unwrap();
        if matches!(result, Value::Bool(true)) {
            Ok(())
        } else if matches!(result, Value::Bool(false)) {
            Err(TestCaseError::fail("refutation succeeded"))
        } else {
            Err(TestCaseError::Fail(
                format!("unexpected result: {:?}", result).into(),
            ))
        }
    }
}

fn get_type_strategy(ty: &Type) -> BoxedStrategy<Value> {
    match ty {
        Type::Bool => prop::bool::ANY.prop_map(Value::Bool).boxed(),
        Type::Str => prop::sample::select(vec!["", "order_checkout", "order_add", "N/A", "other"])
            .prop_map(|s| Value::Str(s.to_string()))
            .boxed(),
        Type::Num => (TESTING_INT_RANGE_MIN..TESTING_INT_RANGE_MAX)
            .prop_map(Value::Num)
            .boxed(),
        Type::List(t) => prop::collection::vec(get_type_strategy(t), 0..TESTING_COLLECTION_SIZE)
            .prop_map(|x| x.into())
            .boxed(),
        Type::Set(t) => prop::collection::vec(get_type_strategy(t), 0..TESTING_COLLECTION_SIZE)
            .prop_map(Value::Set)
            .boxed(),
        Type::Tuple(ts) => {
            let strategies = ts.iter().map(get_type_strategy).collect::<Vec<_>>();
            // create a tuple of values for each type
            strategies
                .prop_map(|xs: Vec<Value>| Value::Tuple(xs))
                .boxed()
        }
        Type::Map(k, v) => {
            let k_strategy = get_type_strategy(k);
            let v_strategy = get_type_strategy(v);
            prop::collection::vec((k_strategy, v_strategy), 0..TESTING_COLLECTION_SIZE)
                .prop_map(Value::Map)
                .boxed()
        }
        _ => unimplemented!("unsupported type {}", ty),
    }
}

pub fn refute_normalizer(
    func_acc: &Expr,
    init: &Expr,
    free_var: FunctionHasFreeVar,
) -> Result<bool, String> {
    let acc_type = infer(&TypeEnv::default(), func_acc)?;
    let (param_types, ret_type) = acc_type.uncurry_fn();
    let [state_type, input_type] = param_types.as_slice() else {
        panic!("expected two parameters: {:?}", param_types)
    };
    assert!(
        ret_type == *state_type,
        "expected the return type {} to match the accumulator type {}",
        ret_type,
        state_type,
    );

    let append_free_vars = |expr: Expr| {
        expr.bind_lets(match free_var {
            FunctionHasFreeVar::Yes(n, t) => vec![(n, t.default_value().into())],
            FunctionHasFreeVar::No => vec![],
        })
    };

    let state_strategy = get_type_strategy(state_type).prop_map(StateValue);
    let input_strategy = get_type_strategy(input_type).prop_map(InputValue);

    let prop_1 = prop({
        let premise = binop!(
            BinOpKinds::Eq,
            func_acc.clone().call(vec![init.clone(), var!(_x)]),
            init.clone()
        );
        let conclusion = binop!(
            BinOpKinds::Eq,
            func_acc.clone().call(vec![var!(_s), var!(_x)]),
            var!(_s)
        );
        append_free_vars(binop!(
            BinOpKinds::Or,
            unaryop!(UnaryOpKinds::Not, premise),
            conclusion
        ))
    });

    let result_1 = {
        let strategy = (state_strategy.clone(), input_strategy.clone());
        let result = create_runner().run(&strategy, prop_1);

        match result {
            Ok(_) => false,
            Err(err) => {
                event!(Level::DEBUG, "refutation succeeded for prop_1: {:?}", err);
                true
            }
        }
    };

    if result_1 {
        return Ok(true);
    }

    let prop_2 = prop_two_input({
        let premise = binop!(
            BinOpKinds::Eq,
            func_acc.clone().call(vec![init.clone(), var!(_x1)]),
            func_acc.clone().call(vec![init.clone(), var!(_x2)])
        );
        let conclusion = binop!(
            BinOpKinds::Eq,
            func_acc.clone().call(vec![var!(_s), var!(_x1)]),
            func_acc.clone().call(vec![var!(_s), var!(_x2)])
        );
        append_free_vars(binop!(
            BinOpKinds::Or,
            unaryop!(UnaryOpKinds::Not, premise),
            conclusion
        ))
    });

    let result_2 = {
        let strategy = (state_strategy, input_strategy.clone(), input_strategy);
        let result = create_runner().run(&strategy, prop_2);

        match result {
            Ok(_) => false,
            Err(err) => {
                event!(Level::DEBUG, "refutation succeeded for prop_2: {:?}", err);
                true
            }
        }
    };

    Ok(result_2)
}

pub fn refute_homomorphism(prog: &Expr) -> Result<bool, String> {
    let type_env = TypeEnv::default();
    let Type::Fn(input, _) = infer(&type_env, prog)? else {
        return Err("program must be a function".to_string());
    };
    let input_strategy = get_type_strategy(&input);
    let strategy = (
        input_strategy.clone(),
        input_strategy.clone(),
        input_strategy.clone(),
        input_strategy.clone(),
    );
    match create_runner().run(&strategy, {
        |(xs1, xs2, ys1, ys2)| {
            let p1 = binop!(
                BinOpKinds::Eq,
                prog.clone().app(var!(_xs1)),
                prog.clone().app(var!(_xs2))
            );
            let p2 = binop!(
                BinOpKinds::Eq,
                prog.clone().app(var!(_ys1)),
                prog.clone().app(var!(_ys2))
            );
            let p3 = unaryop!(
                UnaryOpKinds::Not,
                binop!(
                    BinOpKinds::Eq,
                    prog.clone()
                        .app(binop!(BinOpKinds::Concat, var!(_xs1), var!(_ys1))),
                    prog.clone()
                        .app(binop!(BinOpKinds::Concat, var!(_xs2), var!(_ys2)))
                )
            );

            let prop = binop!(BinOpKinds::And, binop!(BinOpKinds::And, p1, p2), p3);
            let mut env = create_testing_env();
            env.push((sym!(_xs1), xs1));
            env.push((sym!(_xs2), xs2));
            env.push((sym!(_ys1), ys1));
            env.push((sym!(_ys2), ys2));

            match eval(&prop, &mut env) {
                Ok(Value::Bool(true)) => {
                    Err(TestCaseError::fail("refutation succeeded for top-level"))
                }
                Ok(Value::Bool(false)) => Ok(()),
                err => {
                    event!(
                        Level::DEBUG,
                        "refutation error: {:?} for program {} \n prop: {}",
                        err,
                        prog,
                        prop
                    );
                    Err(TestCaseError::Reject("invalid input".into()))
                }
            }
        }
    }) {
        Ok(_) => Ok(false),
        Err(err) => {
            event!(Level::DEBUG, "refutation: {}", err);
            Ok(true)
        }
    }
}

pub fn refute_homomorphism2(func_acc: &Expr, init: &Expr) -> Result<bool, String> {
    let acc_type = infer(&TypeEnv::default(), func_acc)?;
    let (param_types, ret_type) = acc_type.uncurry_fn();
    let &[state_type, input_type] = param_types.as_slice() else {
        panic!("expected two parameters: {:?}", param_types)
    };
    assert!(
        ret_type == state_type,
        "expected the return type {} to match the accumulator type {}",
        ret_type,
        state_type,
    );

    let top_level = Expr::create_top_level(func_acc.clone(), init.clone());

    let input_strategy = get_type_strategy(&tlist!(input_type.clone()));
    let strategy = (
        input_strategy.clone(),
        input_strategy.clone(),
        input_strategy.clone(),
    );
    match create_runner().run(&strategy, {
        |(xs1, xs2, ys)| {
            let p1 = expr_eq!(
                top_level.clone().call(vec![var!(_xs1)]),
                top_level.clone().call(vec![var!(_xs2)])
            );
            let alt_top_level =
                Expr::create_top_level(func_acc.clone(), top_level.clone().app(var!(_ys)));
            let p2 = expr_eq!(
                alt_top_level.clone().app(var!(_xs1)),
                alt_top_level.app(var!(_xs2))
            );

            let prop = p1 & (!p2);
            let mut env = create_testing_env();
            env.push((sym!(_xs1), xs1));
            env.push((sym!(_xs2), xs2));
            env.push((sym!(_ys), ys));

            match eval(&prop, &mut env) {
                Ok(Value::Bool(true)) => Err(TestCaseError::fail(
                    "refutation succeeded for alternative top-level",
                )),
                Ok(Value::Bool(false)) => Ok(()),
                err => {
                    event!(
                        Level::DEBUG,
                        "refutation error: {:?} for program {} \n prop: {}",
                        err,
                        func_acc,
                        prop
                    );
                    Err(TestCaseError::Reject("invalid input".into()))
                }
            }
        }
    }) {
        Ok(_) => Ok(false),
        Err(err) => {
            event!(Level::DEBUG, "refutation: {}", err);
            Ok(true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_log::test]
    fn test_degenerate_program() {
        let f = var!(x).bind_params(vec![param!(s, Type::Num), param!(x, Type::Num)]);

        assert!(refute_normalizer(&f, &0.into(), Default::default()).unwrap());

        let p = Expr::create_top_level(f, 0.into());
        let result = refute_homomorphism(&p).unwrap();
        assert!(result);
    }

    #[test_log::test]
    fn test_mts() {
        let f = var!(max)
            .call(vec![var!(mts) + var!(x), 0.into()])
            .bind_params(vec![param!(mts, Type::Num), param!(x, Type::Num)]);

        let init = 0.into();
        let refute = refute_normalizer(&f, &init, Default::default());
        assert!(refute.unwrap());
    }

    #[test_log::test]
    fn test_mps() {
        let f = tuple!(
            tuple_access!(var!(s), 0) + var!(x),
            var!(max).call(vec![
                tuple_access!(var!(s), 0) + var!(x) + var!(x),
                tuple_access!(var!(s), 1)
            ])
        )
        .bind_params(vec![
            param!(s, ttuple!(Type::Num, Type::Num)),
            param!(x, Type::Num),
        ]);

        let init = tuple!(0.into(), 0.into());
        let refute = refute_normalizer(&f, &init, Default::default());
        assert!(refute.unwrap());
    }

    #[test_log::test]
    fn test_mss() {
        let mts = tuple_access!(var!(s), 0);
        let mss = tuple_access!(var!(s), 1);

        let f = tuple!(
            var!(max).call(vec![mts.clone() + var!(x), 0.into()]),
            var!(max).call(vec![mts.clone() + var!(x), mss])
        )
        .bind_params(vec![
            param!(s, ttuple!(Type::Num, Type::Num)),
            param!(x, Type::Num),
        ]);

        let init = tuple!(0.into(), 0.into());
        let refute = refute_normalizer(&f, &init, Default::default());
        assert!(refute.unwrap());
    }

    #[test_log::test]
    fn test_atoi() {
        let f = (var!(s) * 10.into() + var!(x))
            .bind_params(vec![param!(s, Type::Num), param!(x, Type::Num)]);
        let init = 0.into();
        let refute = refute_normalizer(&f, &init, Default::default());
        assert!(refute.unwrap());
    }

    #[test_log::test]
    fn test_balanced_parenthesis() {
        let cnt = tuple_access!(var!(s), 0);
        let is_bal = tuple_access!(var!(s), 1);

        let cnt_expr = cnt + ite!(expr_gt!(var!(x), 0.into()), 1.into(), (-1).into());

        let f = tuple!(cnt_expr.clone(), is_bal & !expr_lt!(cnt_expr, 0.into())).bind_params(vec![
            param!(s, ttuple!(Type::Num, Type::Bool)),
            param!(x, Type::Num),
        ]);
        let init = tuple!(0.into(), true.into());

        let refute = refute_homomorphism2(&f, &init);
        assert!(refute.unwrap());
    }

    #[test_log::test]
    fn test_0star1star() {
        /*
        _Bool _0star1star (_Bool *ar, int n) {
          _Bool an = 1;
          _Bool bn = 1;

          for(int i = 0; i < n; i++) {
            an = (ar [i]) && an;
            bn = ((! ar [i]) || an) && bn;
          }

          return bn;
        }
        */
        let an_expr = tuple_access!(var!(s), 0) & var!(x);
        let bn_expr = ((!var!(x)) | an_expr.clone()) & tuple_access!(var!(s), 1);

        let f = tuple!(an_expr, bn_expr).bind_params(vec![
            param!(s, ttuple!(Type::Bool, Type::Bool)),
            param!(x, Type::Bool),
        ]);

        let init = tuple!(true.into(), true.into());
        let refute = refute_normalizer(&f, &init, Default::default());
        assert!(refute.unwrap());
    }

    #[test_log::test]
    fn test_count_1s() {
        /*
        int _cnt1s (_Bool * a, int n) {
          int i = 0;
          _Bool f = 0;
          int cnt = 0;
          for(i =0; i < n; i++) {
            cnt += (a[i] && !f) ? 1 : 0;
            f = a[i];
          }
          return cnt;
        }
        */
        let cnt_expr = ite!(var!(x) & (!tuple_access!(var!(s), 1)), 1.into(), 0.into())
            + tuple_access!(var!(s), 0);

        let f = tuple!(cnt_expr, var!(x)).bind_params(vec![
            param!(s, ttuple!(Type::Num, Type::Bool)),
            param!(x, Type::Bool),
        ]);

        let init = tuple!(0.into(), false.into());
        let refute = refute_normalizer(&f, &init, Default::default());
        assert!(refute.unwrap());
    }

    #[test_log::test]
    fn test_line_sight() {
        /*
        _Bool _line_sight (int *a, int n) {
          int amax = 0;
          _Bool visible = 1;

          for(int i = 0; i < n; i++) {
            visible = (amax <= a[i]);
            amax = max (amax, a[i]);
          }
          return visible;
        }
        */

        let f = tuple!(
            !(expr_gt!(tuple_access!(var!(s), 1), var!(x))),
            var!(max).call(vec![tuple_access!(var!(s), 1), var!(x)])
        )
        .bind_params(vec![
            param!(s, ttuple!(Type::Bool, Type::Num)),
            param!(x, Type::Num),
        ]);

        let init = tuple!(true.into(), 0.into());
        let refute = refute_normalizer(&f, &init, Default::default());
        assert!(refute.unwrap());
    }

    #[test_log::test]
    fn test_0_after_1() {
        /*
        _Bool _0after1 (_Bool *a, int n) {
          _Bool seen1 = 0;
          _Bool res = 0;

          for (int i = 0; i < n; i++) {
            if (seen1 && !(a[i]))
              res = 1;
            seen1 = seen1 || a[i];
          }

          return res;
        }
        */

        let f = tuple!(
            tuple_access!(var!(s), 0) | var!(x),
            ite!(
                tuple_access!(var!(s), 0) & (!var!(x)),
                true.into(),
                tuple_access!(var!(s), 1)
            )
        )
        .bind_params(vec![
            param!(s, ttuple!(Type::Bool, Type::Bool)),
            param!(x, Type::Bool),
        ]);

        let init = tuple!(false.into(), false.into());
        let refute = refute_normalizer(&f, &init, Default::default());
        assert!(refute.unwrap());
    }

    #[test_log::test]
    fn test_max_length_block() {
        /*
        int _max_length_of_1 (_Bool *a, int n) {
          int cl = 0;
          int ml = 0;

          for (int i = 0; i < n; i++) {
            cl = a[i] ? cl + 1 : 0;
            ml = max (ml, cl);
          }
          return ml + c;
        }
        */

        let cl_expr = ite!(var!(x), tuple_access!(var!(s), 0) + 1.into(), 0.into());
        let ml_expr = var!(max).call(vec![tuple_access!(var!(s), 1), cl_expr.clone()]);

        let f = tuple!(cl_expr, ml_expr).bind_params(vec![
            param!(s, ttuple!(Type::Num, Type::Num)),
            param!(x, Type::Bool),
        ]);

        let init = tuple!(0.into(), 0.into());
        let refute = refute_normalizer(&f, &init, Default::default());
        assert!(refute.unwrap());
    }
}
