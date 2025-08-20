use crate::lang::macros::*;
use crate::lang::{BinOpKinds, Env, Expr, UnaryOpKinds, Value};

fn eval_binop(binop: &BinOpKinds, lhs: &Value, rhs: &Value) -> Result<Value, String> {
    use BinOpKinds::*;
    use Value::*;

    match (binop, lhs, rhs) {
        (Add, &Num(lhs), &Num(rhs)) => Ok(Num(lhs + rhs)),
        (Sub, &Num(lhs), &Num(rhs)) => Ok(Num(lhs - rhs)),
        (Mul, &Num(lhs), &Num(rhs)) => Ok(Num(lhs * rhs)),
        (Div, &Num(lhs), &Num(rhs)) => {
            if rhs == 0 {
                Ok(Num(0))
            } else {
                Ok(Value::Num(lhs / rhs))
            }
        }

        (Gt, &Num(lhs), &Num(rhs)) => Ok(Bool(lhs > rhs)),
        (Lt, &Num(lhs), &Num(rhs)) => Ok(Bool(lhs < rhs)),
        (Eq, lhs, rhs) => Ok(Bool(lhs == rhs)),

        (And, &Bool(lhs), &Bool(rhs)) => Ok(Bool(lhs && rhs)),
        (Or, &Bool(lhs), &Bool(rhs)) => Ok(Bool(lhs || rhs)),

        (Concat, any, Nil) => Ok(any.clone()),
        (Concat, Nil, any) => Ok(any.clone()),
        (Concat, Cons(head, tail), ys) => {
            Ok(Cons(head.clone(), Box::new(eval_binop(binop, tail, ys)?)))
        }

        _ => Err(format!("Invalid operation: {lhs:?} {binop:?} {rhs:?}")),
    }
}

fn eval_unaryop(unaryop: &UnaryOpKinds, expr: &Value) -> Result<Value, String> {
    use UnaryOpKinds::*;
    use Value::*;

    match (unaryop, expr) {
        (Neg, &Num(n)) => Ok(Num(-n)),
        (Not, &Bool(b)) => Ok(Bool(!b)),
        _ => Err(format!("Invalid operation: {unaryop:?} {expr:?}")),
    }
}

fn eval_app_multi(func: &Value, args: Vec<Value>) -> Result<Value, String> {
    args.iter().fold(Ok(func.clone()), |acc, arg| {
        acc.and_then(|f| eval_app(&f, arg.clone()))
    })
}

fn eval_app(func: &Value, arg: Value) -> Result<Value, String> {
    use Value::*;

    match func {
        Closure { env, param, body } => {
            let mut env = env.clone();
            env.push((*param, arg));
            eval(body, &mut env)
        }
        PrimitiveFn { func, num_params } => {
            if *num_params == 1 {
                func(vec![arg])
            } else {
                Ok(PrimitiveFnCall {
                    func: func.clone(),
                    num_params: *num_params,
                    args: vec![arg],
                })
            }
        }
        PrimitiveFnCall {
            func,
            num_params,
            args,
        } => {
            let mut args = args.clone();
            args.push(arg);
            if args.len() == *num_params {
                func(args)
            } else {
                Ok(PrimitiveFnCall {
                    func: func.clone(),
                    num_params: *num_params,
                    args,
                })
            }
        }
        _ => Err(format!("Invalid function: {func:?} {arg:?}")),
    }
}

fn eval_if_cond(cond: &Value) -> Result<bool, String> {
    match cond {
        Value::Bool(b) => Ok(*b),
        _ => Err("Invalid condition".to_string()),
    }
}

pub fn eval(expr: &Expr, env: &mut Env) -> Result<Value, String> {
    use Expr::*;
    use Value::*;

    match expr {
        Expr::Num(n) => Ok(Value::Num(*n)),
        Expr::Bool(b) => Ok(Value::Bool(*b)),
        Expr::Str(s) => Ok(Value::Str(s.clone())),
        Var(name) => env
            .iter()
            .rev()
            .find_map(|(n, v)| if n == name { Some(v.clone()) } else { None })
            .ok_or_else(|| format!("Variable {name} not found")),

        UnaryOp(op, expr) => eval_unaryop(op, &(eval(expr, env)?)),
        BinOp(op, lhs, rhs) => eval_binop(op, &eval(lhs, env)?, &eval(rhs, env)?)
            .map_err(|e| format!("when evaluating {lhs} {op} {rhs}: {e}")),

        Expr::Nil => Ok(Value::Nil),
        Expr::Cons { head, tail } => Ok(Value::Cons(
            Box::new(eval(head, env)?),
            Box::new(eval(tail, env)?),
        )),
        Expr::Tuple(exprs) => Ok(Value::Tuple(
            exprs
                .iter()
                .map(|e| eval(e, env))
                .collect::<Result<_, _>>()?,
        )),
        Expr::TupleAccess(tup, idx) => {
            let tup = eval(tup, env)?;
            let idx = *idx;
            match tup {
                Value::Tuple(tup) => Ok(tup[idx].clone()),
                _ => Err(format!("expected tuple but got {tup}")),
            }
        }
        MapAssign { map, key, value } => {
            let map = eval(map, env)?;
            let key = eval(key, env)?;
            let value = eval(value, env)?;
            match map {
                Map(mut vec) => {
                    vec.retain(|(k, _)| k != &key);
                    vec.push((key, value));
                    Ok(Map(vec))
                }
                _ => Err(format!("Invalid map in MapAssign: {:?}", map)),
            }
        }
        MapAccess { map, key } => {
            let map = eval(map, env)?;
            let key = eval(key, env)?;
            match map {
                Map(vec) => vec
                    .iter()
                    .find_map(|(k, v)| if k == &key { Some(v.clone()) } else { None })
                    // todo: need to get the type of the map
                    .or(Some(0.into()))
                    .ok_or_else(|| "Key not found in map".to_string()),
                _ => Err(format!("Invalid map in MapAccess: {:?}", map)),
            }
        }

        App { func, arg } => {
            let func = eval(func, env)?;
            let arg = eval(arg, env)?;
            eval_app(&func, arg)
        }

        Lambda { param, body, .. } => Ok(Closure {
            env: env.clone(),
            param: *param,
            body: *body.clone(),
        }),
        Let { name, expr, body } => {
            let value = eval(expr, env)?;
            env.push((*name, value));
            let result = eval(body, env);
            env.pop();
            result
        }

        Ite {
            cond,
            then_expr,
            else_expr,
        } => {
            let cond = eval(cond, env)?;
            match cond {
                Value::Bool(true) => eval(then_expr, env),
                Value::Bool(false) => eval(else_expr, env),
                _ => Err("Invalid condition".to_string()),
            }
        }
    }
}

pub fn create_env() -> Env {
    use Value::*;

    vec![
        (sym!(true), Bool(true)),
        (sym!(false), Bool(false)),
        (sym!(nil), Nil),
        (sym!(empty_map), Map(vec![])),
        (sym!(empty_set), Set(vec![])),
        (
            sym!(head),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [Cons(head, _)] => Ok(*head.clone()),
                    [Nil] => Err("Empty list".to_string()),
                    _ => panic!("impossible: invalid arguments"),
                },
                num_params: 1,
            },
        ),
        (
            sym!(tail),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [Cons(_, tail)] => Ok(*tail.clone()),
                    [Nil] => Err("Empty list".to_string()),
                    _ => panic!("impossible: invalid arguments"),
                },
                num_params: 1,
            },
        ),
        (
            sym!(foldl),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [f, acc, it] => {
                        let mut iterable = it;
                        let mut acc = (*acc).clone();

                        while *iterable != Nil {
                            match iterable {
                                Cons(head, tail) => {
                                    iterable = tail;
                                    acc = eval_app_multi(&f, vec![acc, (**head).clone()])?;
                                }
                                Nil => panic!("impossible: invalid arguments"),
                                _ => Err(format!("invalid iterable for foldl: {}", iterable))?,
                            }
                        }

                        Ok(acc)
                    }
                    _ => panic!("impossible: invalid arguments"),
                },
                num_params: 3,
            },
        ),
        (
            sym!(filter),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [f, it] => {
                        let it: Vec<Value> = it.clone().into();
                        let filtered = it
                            .into_iter()
                            .filter(|v| {
                                let result = eval_app_multi(&f, vec![v.clone()]).unwrap();
                                eval_if_cond(&result).unwrap()
                            })
                            .collect::<Vec<_>>();
                        Ok(filtered.into())
                    }
                    _ => Err("Invalid arguments".to_string()),
                },
                num_params: 2,
            },
        ),
        (
            sym!(map),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [f, it] => {
                        let it: Vec<Value> = it.clone().into();
                        let mapped = it
                            .into_iter()
                            .map(|v| eval_app_multi(&f, vec![v.clone()]).unwrap())
                            .collect::<Vec<_>>();
                        Ok(mapped.into())
                    }
                    _ => Err("Invalid arguments".to_string()),
                },
                num_params: 2,
            },
        ),
        (
            sym!(update_map),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [m, k, v] => {
                        let mut m = m.clone();
                        let k = k.clone();
                        let v = v.clone();
                        match &mut m {
                            Map(vec) => {
                                vec.retain(|(key, _)| key != &k);
                                vec.push((k, v));
                                Ok(m)
                            }
                            _ => Err("Invalid map".to_string()),
                        }
                    }
                    _ => Err("Invalid arguments".to_string()),
                },
                num_params: 3,
            },
        ),
        (
            sym!(zip),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [xs, ys] => {
                        let xs: Vec<Value> = xs.clone().into();
                        let ys: Vec<Value> = ys.clone().into();
                        let zipped = xs
                            .into_iter()
                            .zip(ys.into_iter())
                            .map(|(x, y)| Tuple(vec![x, y]))
                            .collect::<Vec<_>>();
                        Ok(zipped.into())
                    }
                    _ => Err("Invalid arguments".to_string()),
                },
                num_params: 2,
            },
        ),
        (
            sym!(id),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [x] => Ok(x.clone()),
                    _ => Err("more than one arguments to id".to_string()),
                },
                num_params: 1,
            },
        ),
        (
            sym!(max),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [Value::Num(lhs), Value::Num(rhs)] => Ok(Value::Num((*lhs).max(*rhs))),
                    _ => Err("Invalid arguments".to_string()),
                },
                num_params: 2,
            },
        ),
        (
            sym!(min),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [Value::Num(lhs), Value::Num(rhs)] => Ok(Value::Num((*lhs).min(*rhs))),
                    _ => Err("Invalid arguments".to_string()),
                },
                num_params: 2,
            },
        ),
        (
            sym!(abs),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [Value::Num(n)] => Ok(Value::Num(n.abs())),
                    _ => Err("Invalid arguments".to_string()),
                },
                num_params: 1,
            },
        ),
        (
            sym!(union),
            PrimitiveFn {
                func: |args| {
                    let sets = args
                        .into_iter()
                        .map(|arg| match arg {
                            Value::Set(s) => Ok(s),
                            _ => return Err(format!("expected set but get {:?}", arg)),
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    Ok(Value::Set(sets.into_iter().flatten().collect()))
                },
                num_params: 2,
            },
        ),
        (
            sym!(intersection),
            PrimitiveFn {
                func: |args| {
                    let sets = args
                        .into_iter()
                        .map(|arg| match arg {
                            Value::Set(s) => Ok(s),
                            _ => return Err(format!("expected set but get {:?}", arg)),
                        })
                        .collect::<Result<Vec<_>, String>>()?;

                    Ok(Value::Set(
                        sets[0]
                            .iter()
                            .filter(|k| sets.iter().all(|s| s.contains(k)))
                            .cloned()
                            .collect(),
                    ))
                },
                num_params: 2,
            },
        ),
        (
            sym!(set_add),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [v, Value::Set(set)] => Ok(Value::Set(
                        set.iter()
                            .filter(|x| *x != v)
                            .cloned()
                            .chain(std::iter::once(v.clone()))
                            .collect(),
                    )),
                    _ => Err("Invalid arguments".to_string()),
                },
                num_params: 2,
            },
        ),
        (
            sym!(map_set),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [f, Value::Set(s)] => {
                        let mapped = s
                            .iter()
                            .map(|v| eval_app_multi(&f, vec![v.clone()]).unwrap())
                            .collect::<Vec<_>>();
                        Ok(Value::Set(mapped))
                    }
                    _ => Err("Invalid arguments".to_string()),
                },
                num_params: 2,
            },
        ),
        (
            sym!(filter_set),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [p, Value::Set(s)] => {
                        let filtered = s
                            .into_iter()
                            .filter(|v| {
                                let result = eval_app_multi(&p, vec![(*v).clone()]).unwrap();
                                eval_if_cond(&result).unwrap()
                            })
                            .cloned()
                            .collect::<Vec<_>>();
                        Ok(Value::Set(filtered))
                    }
                    _ => Err("Invalid arguments".to_string()),
                },
                num_params: 2,
            },
        ),
        (
            sym!(contains_key),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [Value::Map(map), key] => Ok(Value::Bool(map.iter().any(|(k, _)| k == key))),
                    _ => Err("Invalid arguments".to_string()),
                },
                num_params: 2,
            },
        ),
        (
            sym!(map_values),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [f, Value::Map(map)] => Ok(Value::Map(
                        map.iter()
                            .filter_map(|(k, v)| {
                                let result = eval_app_multi(
                                    &f,
                                    vec![k.clone(), Value::Tuple(vec![true.into(), v.clone()])],
                                )
                                .unwrap();
                                match result {
                                    Tuple(vec) => {
                                        if let [Value::Bool(true), v] = vec.as_slice() {
                                            Some((k.clone(), v.clone()))
                                        } else {
                                            None
                                        }
                                    }
                                    _ => None,
                                }
                            })
                            .collect::<Vec<_>>(),
                    )),

                    _ => Err("Invalid arguments".to_string()),
                },
                num_params: 2,
            },
        ),
        (
            sym!(concat_map),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [Value::Map(map), Value::Map(map2)] => Ok(Value::Map(
                        map2.iter()
                            .chain(
                                map.iter()
                                    .filter(|(k, _)| !map2.iter().any(|(k2, _)| k == k2)),
                            )
                            .cloned()
                            .collect(),
                    )),
                    _ => Err("concat_map requires two maps".to_string()),
                },
                num_params: 2,
            },
        ),
        (
            sym!(filter_values),
            PrimitiveFn {
                func: |args| match args.as_slice() {
                    [f, Value::Map(map)] => Ok(Value::Map(
                        map.iter()
                            .filter(|(_k, v)| {
                                let result = eval_app_multi(&f, vec![v.clone()]).unwrap();
                                match result {
                                    Value::Bool(true) => true,
                                    Value::Bool(false) => false,
                                    _ => panic!("Invalid return type from filter function"),
                                }
                            })
                            .cloned()
                            .collect::<Vec<_>>(),
                    )),
                    _ => Err("Invalid arguments".to_string()),
                },
                num_params: 2,
            },
        ),
        (
            sym!(length),
            PrimitiveFn {
                func: |mut args| match args.as_slice() {
                    [_] => match args.remove(0) {
                        Value::Nil => Ok(Value::Num(0)),
                        Value::Cons(_, tail) => {
                            let mut len = 1;
                            let mut cur = *tail;
                            while let Value::Cons(_, t) = cur {
                                len += 1;
                                cur = *t;
                            }
                            Ok(Value::Num(len))
                        }
                        Value::Str(s) => Ok(Value::Num(s.len() as i32)),
                        _ => Err("Invalid arguments".to_string()),
                    },
                    _ => Err("Invalid arguments".to_string()),
                },
                num_params: 1,
            },
        ),
    ]
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::consts::INPUT_PARAM;
    use crate::lang::{IsCurriedFunction, Type};
    use quickcheck::quickcheck;
    use rstest::rstest;

    #[derive(Debug, Copy, Clone)]
    struct SmallNumber(i32);

    impl quickcheck::Arbitrary for SmallNumber {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            SmallNumber(i32::arbitrary(g) % 10000)
        }
    }

    #[rstest]
    #[case(Expr::Num(123), Value::Num(123))]
    fn test_eval(#[case] input_expr: Expr, #[case] expected_value: Value) {
        assert_eq!(eval(&input_expr, &mut vec![]), Ok(expected_value));
    }

    macro_rules! eval_binop_num_tests {
        ($($test_name:ident: ($x:path, $op:tt, $res_type:path),)*) => {
            $(
                quickcheck! {
                    #[allow(arithmetic_overflow)]
                    fn $test_name(lhs: SmallNumber, rhs: SmallNumber) -> bool {
                        let SmallNumber(lhs) = lhs;
                        let SmallNumber(rhs) = rhs;

                        let lv = lhs.into();
                        let rv = rhs.into();
                        let expected =
                            if $x == BinOpKinds::Div && rhs == 0 { Value::Num(0) }
                            else { $res_type(lhs $op rhs) };
                        eval_binop(&$x, &lv, &rv) == Ok(expected)
                    }
                }
            )*
        };
    }

    eval_binop_num_tests! {
        test_binop_add: (BinOpKinds::Add, +, Value::Num),
        test_binop_sub: (BinOpKinds::Sub, -, Value::Num),
        test_binop_mul: (BinOpKinds::Mul, *, Value::Num),
        test_binop_div: (BinOpKinds::Div, /, Value::Num),
        test_binop_gt: (BinOpKinds::Gt, >, Value::Bool),
        test_binop_lt: (BinOpKinds::Lt, <, Value::Bool),
        test_binop_eq: (BinOpKinds::Eq, ==, Value::Bool),
    }

    macro_rules! eval_binop_bool_tests {
        ($($test_name:ident: ($x:path, $op:tt),)*) => {
            $(
                quickcheck! {
                    fn $test_name(lhs: bool, rhs: bool) -> bool {
                        let lv = Value::Bool(lhs);
                        let rv = Value::Bool(rhs);
                        let expected = Value::Bool(lhs $op rhs);
                        eval_binop(&$x, &lv, &rv) == Ok(expected)
                    }
                }
            )*
        };
    }

    eval_binop_bool_tests! {
        test_binop_and: (BinOpKinds::And, &&),
        test_binop_or: (BinOpKinds::Or, ||),
    }

    #[rstest]
    #[case(BinOpKinds::Add, Value::Num(1), Value::Num(2), Value::Num(3))]
    fn test_eval_binop(
        #[case] binop: BinOpKinds,
        #[case] lhs: Value,
        #[case] rhs: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(eval_binop(&binop, &lhs, &rhs), Ok(expected));
    }

    #[rstest]
    #[case(BinOpKinds::Add, Value::Num(1), Value::Bool(true))]
    #[case(BinOpKinds::And, Value::Num(1), Value::Bool(true))]
    fn test_eval_binop_err(#[case] binop: BinOpKinds, #[case] lhs: Value, #[case] rhs: Value) {
        assert!(eval_binop(&binop, &lhs, &rhs).is_err());
    }

    #[test]
    fn test_eval_foldl() {
        let mut env = create_env();
        let init = 1;
        let xs: Value = vec![1, 2, 3, 4].into();

        let body = Expr::from(sym!(foldl)).call(vec![
            Expr::BinOp(
                BinOpKinds::Mul,
                Box::new(Expr::from(sym!(acc))),
                Box::new(Expr::from(sym!(x))),
            )
            .bind_params(vec![param!(acc, Type::Num), param!(x, Type::Num)]),
            Expr::Num(init),
            Expr::Var(INPUT_PARAM.into()),
        ]);

        env.push((INPUT_PARAM.into(), xs));
        assert_eq!(eval(&body, &mut env), Ok(Value::Num(24)));
        env.pop();
        env.push((INPUT_PARAM.into(), Value::Nil));
        assert_eq!(eval(&body, &mut env), Ok(Value::Num(init)));
    }

    #[test]
    fn test_eval_map() {
        let mut env = create_env();
        let xs: Value = vec![1, 2, 3, 4].into();

        let body = Expr::from(sym!(map)).call(vec![
            Expr::BinOp(
                BinOpKinds::Add,
                Box::new(Expr::from(1)),
                Box::new(Expr::from(sym!(x))),
            )
            .bind_params(vec![param!(x, Type::Num)]),
            Expr::Var(INPUT_PARAM.into()),
        ]);

        env.push((INPUT_PARAM.into(), xs));
        assert_eq!(eval(&body, &mut env), Ok(vec![2, 3, 4, 5].into()));

        env.pop();
        env.push((INPUT_PARAM.into(), Value::Nil));
        assert_eq!(eval(&body, &mut env), Ok(Value::Nil));
    }
}
