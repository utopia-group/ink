use super::preprocess::preprocess;
use super::simp::simplify_function;
use super::{Binding, DecomposedExpression, Destructor, InputSimplifiedFunction};
use crate::lang::macros::*;
use crate::lang::{
    simplify, BasicSubexprTypeMap, BinOpKinds, Expr, IsCurriedFunction, Type, TypeEnv, TypedEnv,
};
use crate::{is_func_filter, is_func_zip};

use crate::Symbol;
use tracing::{event, Level};

/// Abstraction Decomposition state; keeps track of the fresh variables and type information.
#[derive(Debug, Clone)]
pub struct VdState {
    num_fresh_vars: usize,
    type_map: BasicSubexprTypeMap,
}

impl VdState {
    /// add a fresh variable to the environment of the given type.
    fn add_fresh_var(&mut self, ty: Type) -> Symbol {
        let var = format!("v{}", self.num_fresh_vars).into();
        self.num_fresh_vars += 1;
        self.type_map.add_var(var, crate::lang::PolyType::Mono(ty));
        var
    }
}

fn decompose_lambda(expr: &Expr, state: &mut VdState) -> Result<DecomposedExpression, String> {
    let Expr::Lambda {
        param,
        param_type,
        body,
    } = expr
    else {
        unimplemented!("unsupported lambda expression: {}", expr);
    };

    Ok(simplify_function(
        *param,
        param_type,
        &Destructor::Id,
        decompose_vertical(body, state)?,
    ))
}

fn decompose_tuple(expr: &Expr, state: &mut VdState) -> Result<DecomposedExpression, String> {
    match expr {
        Expr::Tuple(els) => {
            let decomps = els
                .into_iter()
                .map(|expr| decompose_vertical(expr, state))
                .collect::<Result<Vec<_>, _>>()?;

            event!(
                Level::DEBUG,
                "decompose_tuple: decomposed {} into {} elements",
                expr,
                decomps.len()
            );
            for decomp in &decomps {
                event!(Level::DEBUG, "\t\tdecompose_tuple: decomp: {}", decomp);
            }

            Ok(DecomposedExpression::Product {
                typ: state.type_map.get_type(expr),
                decomps,
            })
        }

        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => {
            // propagate ite to the elements of the tuple
            let Type::Tuple(typs) = state.type_map.get_type(expr) else {
                panic!("decompose_tuple: expected Tuple type");
            };

            let elms = typs
                .into_iter()
                .enumerate()
                .map(|(i, _elm_type)| {
                    let then_expr = tuple_access!(*then_expr.clone(), i);
                    let else_expr = tuple_access!(*else_expr.clone(), i);

                    simplify(&ite!(*cond.clone(), then_expr, else_expr))
                })
                .collect::<Vec<_>>();

            decompose_tuple(&Expr::Tuple(elms), state)
        }

        _ => Err(format!("decompose_tuple: expected Tuple, got {}", expr)),
    }
}

fn decompose_map(
    func: &Expr,
    iter: DecomposedExpression,
    state: &mut VdState,
) -> Result<DecomposedExpression, String> {
    let DecomposedExpression::Collection {
        decomp,
        iter_type,
        predicate,
        destructors,
        bindings,
    } = iter
    else {
        panic!("decompose_map: expected Collection; instead got {:?}", iter);
    };

    let DecomposedExpression::Func(InputSimplifiedFunction::Id(value)) = *decomp else {
        return Err(format!(
            "decompose_map does not support {:?} when applying map rule currently",
            decomp
        ));
    };

    let expr = match iter_type {
        Type::List(..) | Type::Set(..) => simplify(&func.clone().app(value)),
        Type::Map(..) => {
            let &[(Binding::KeyValuePair(k, _), _)] = &bindings.as_slice() else {
                panic!("decompose_map: expected KeyValuePair binding");
            };

            simplify(&func.clone().call(vec![(*k).into(), value]))
        }
        _ => return Err(format!("unsupported map type: {:?}", iter_type)),
    };

    let typ = match iter_type {
        Type::Map(..) => {
            // map_values takes an expression of (key_exists, value)
            // we only need value as the collection type
            let Type::Tuple(tys) = state.type_map.get_type(&expr) else {
                panic!("invalid map type");
            };
            assert_eq!(tys.len(), 2, "invalid map type");
            assert_eq!(tys[0], Type::Bool, "invalid map type");
            tys.into_iter().last().unwrap()
        }
        _ => state.type_map.get_type(&expr),
    };

    Ok(DecomposedExpression::Collection {
        decomp: Box::new(decompose_vertical(&expr, state)?),
        iter_type: iter_type.with_new_element_type(typ),
        predicate,
        destructors,
        bindings,
    })
}

fn decompose_filter(
    pred: &Expr,
    iter: DecomposedExpression,
    _: &mut VdState,
) -> Result<DecomposedExpression, String> {
    let DecomposedExpression::Collection {
        decomp,
        iter_type,
        predicate,
        destructors,
        bindings,
    } = iter
    else {
        panic!("decompose_map: expected Collection");
    };

    let DecomposedExpression::Func(InputSimplifiedFunction::Id(value)) = *decomp.clone() else {
        return Err(format!(
            "decompose_filter does not support {:?} when applying filter rule currently",
            decomp
        ));
    };

    let expr = simplify(&pred.clone().app(value));
    Ok(DecomposedExpression::Collection {
        decomp,
        iter_type,
        predicate: binop!(BinOpKinds::And, predicate, expr),
        destructors,
        bindings,
    })
}

macro_rules! is_decomposable_var {
    ($expr:expr) => {
        matches!(
            $expr,
            crate::lang::Expr::Var(..) | crate::lang::Expr::TupleAccess(..)
        )
    };
}

fn decompose_collection(expr: &Expr, state: &mut VdState) -> Result<DecomposedExpression, String> {
    let expr_type = state.type_map.get_type(expr);
    match expr {
        Expr::App { .. } => {
            let uncurried_expr = expr.clone().uncurry();
            let Expr::App { func, arg } = uncurried_expr else {
                panic!("uncurry failed");
            };
            let Expr::Tuple(args) = *arg else {
                panic!("uncurry failed");
            };

            match (func.as_ref(), args.as_slice()) {
                (Expr::Var(s), [arg_func, arg_iter]) if is_func_map!(s) => {
                    decompose_map(arg_func, decompose_vertical(arg_iter, state)?, state)
                }

                (Expr::Var(s), [arg_pred, arg_iter]) if is_func_filter!(s) => {
                    decompose_filter(arg_pred, decompose_vertical(arg_iter, state)?, state)
                }

                (Expr::Var(s), [iter_1, iter_2]) if is_func_zip!(s) => {
                    let decomp_1 = decompose_vertical(iter_1, state)?;
                    let decomp_2 = decompose_vertical(iter_2, state)?;

                    let (
                        DecomposedExpression::Collection {
                            decomp: d1,
                            iter_type: ty_1,
                            predicate: pred_1,
                            destructors: destructors_1,
                            bindings: bindings_1,
                        },
                        DecomposedExpression::Collection {
                            decomp: d2,
                            iter_type: ty_2,
                            predicate: pred_2,
                            destructors: destructors_2,
                            bindings: bindings_2,
                        },
                    ) = (decomp_1, decomp_2)
                    else {
                        panic!("decompose_collection: expected Collection");
                    };

                    let (
                        DecomposedExpression::Func(InputSimplifiedFunction::Id(d1)),
                        DecomposedExpression::Func(InputSimplifiedFunction::Id(d2)),
                    ) = (*d1, *d2)
                    else {
                        panic!("decompose_collection: expected Func");
                    };

                    let typ = Type::Tuple(vec![ty_1, ty_2]);
                    let iter_type = tlist!(typ.clone());

                    Ok(DecomposedExpression::Collection {
                        decomp: Box::new(DecomposedExpression::Func(
                            tuple!(d1.clone(), d2.clone()).into(),
                        )),
                        iter_type,
                        predicate: binop!(BinOpKinds::And, pred_1, pred_2),
                        destructors: destructors_1.into_iter().chain(destructors_2).collect(),
                        bindings: bindings_1.into_iter().chain(bindings_2).collect(),
                    })
                }

                // do not attempt to decompose further for other built-in functions
                _ => Ok(DecomposedExpression::Func(InputSimplifiedFunction::Id(
                    expr.clone(),
                ))),
            }
        }

        expr if matches!(expr_type, Type::List(..) | Type::Set(..))
            && is_decomposable_var!(expr) =>
        {
            let v = state.add_fresh_var(expr_type.element_type().clone());

            Ok(DecomposedExpression::Collection {
                decomp: Box::new(DecomposedExpression::Func(Expr::from(v).into())),
                iter_type: expr_type.clone(),
                predicate: true.into(),
                destructors: vec![],
                bindings: vec![(Binding::Single(v), expr.clone())],
            })
        }

        // handles custom iterators for map type
        // add fresh variables for key and value, and discard the key
        expr if matches!(expr_type, Type::Map(..)) && is_decomposable_var!(expr) => {
            let Type::Map(ty_k, ty_v) = &expr_type else {
                panic!("decompose_map: expected Map type");
            };
            let k = state.add_fresh_var(*ty_k.clone());
            let v = state.add_fresh_var(ttuple!(Type::Bool, *ty_v.clone()));

            Ok(DecomposedExpression::Collection {
                decomp: Box::new(DecomposedExpression::Func(Expr::from(v).into())),
                iter_type: expr_type.clone(),
                predicate: true.into(),
                destructors: vec![],
                bindings: vec![(Binding::KeyValuePair(k, v), expr.clone())],
            })
        }

        // does not fit any of the above cases, so we cannot decompose further
        _ => Ok(DecomposedExpression::Func(InputSimplifiedFunction::Id(
            expr.clone(),
        ))),
    }
}

fn decompose_vertical(expr: &Expr, state: &mut VdState) -> Result<DecomposedExpression, String> {
    match state.type_map.get_type(expr) {
        // Tuple
        c if c.is_compound() => decompose_tuple(expr, state),

        // Collection, Map, Filter
        c if c.is_collection() => decompose_collection(expr, state),

        // BaseType
        c if c.is_base() => Ok(DecomposedExpression::Func(InputSimplifiedFunction::Id(
            expr.clone(),
        ))),

        // Lam-Base
        Type::Fn(..) if matches!(expr, Expr::Var(..)) => Ok(DecomposedExpression::Func(
            InputSimplifiedFunction::Id(expr.clone()),
        )),

        // Lam-Ind
        Type::Fn(..) => decompose_lambda(expr, state),

        x => Err(format!("unsupported type: {}", x)),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UsePreprocess(pub bool);
impl Default for UsePreprocess {
    fn default() -> Self {
        Self(true)
    }
}
impl From<bool> for UsePreprocess {
    fn from(b: bool) -> Self {
        Self(b)
    }
}

pub fn vd_default(
    expr: &Expr,
    env: &TypeEnv,
    use_preprocess: UsePreprocess,
) -> Result<DecomposedExpression, String> {
    let expr = &simplify(expr);
    let mut state = VdState {
        num_fresh_vars: 0,
        type_map: BasicSubexprTypeMap::new(expr, env.clone()),
    };
    let expr = if !use_preprocess.0 {
        expr
    } else {
        &preprocess(expr.clone(), &state.type_map)
    };
    event!(Level::INFO, "Preprocessed expression for VD: {}", expr);
    decompose_vertical(expr, &mut state)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::{BinOpKinds, Expr::*, PolyType};
    use rstest::rstest;

    #[test_log::test]
    fn test_product_nested() {
        // \s. \x. ((s.1.1 + x, s.1.2 + 2*x), s.2 + 1)

        let elem_1 = tuple_access!(tuple_access!(Var(sym!(s)), 0), 0) + var!(x);
        let elem_2 = tuple_access!(tuple_access!(Var(sym!(s)), 0), 1) + (var!(x) * Expr::Num(2));

        let elem_3 = tuple_access!(Var(sym!(s)), 1) + 1.into();

        let func = Tuple(vec![Tuple(vec![elem_1, elem_2]), elem_3]).bind_params(vec![
            param!(
                s,
                Type::Tuple(vec![Type::Tuple(vec![Type::Num, Type::Num]), Type::Num])
            ),
            param!(x, Type::Num),
        ]);

        let decomp = vd_default(&func, &TypeEnv::default(), Default::default());
        assert_eq!(decomp.expect("decomp failed").to_string(), "((num, num), num)((num, num)((λ(s_0_0: num, x: num). (s_0_0 + x))*(λ(arg: ((num, num), num)). arg._0._0), (λ(s_0_1: num, x: num). (s_0_1 + (x * 2)))*(λ(arg: ((num, num), num)). arg._0._1)), (λ(s_1: num, x: num). (1 + s_1))*(λ(arg: ((num, num), num)). arg._1))");
    }

    #[test_log::test]
    fn test_product_multi() {
        let cond = binop!(
            BinOpKinds::Lt,
            tuple_access!(var!(s), 4),
            tuple_access!(var!(x), 4)
        );
        let temp = ite!(cond, var!(x), var!(s));
        let f = temp.bind_params(vec![
            param!(
                s,
                ttuple![Type::Num, Type::Num, Type::Num, Type::Num, Type::Num]
            ),
            param!(
                x,
                ttuple![Type::Num, Type::Num, Type::Num, Type::Num, Type::Num]
            ),
        ]);
        let decomp = vd_default(&f, &TypeEnv::default(), Default::default());
        println!("{}", decomp.expect("decomp failed"));
    }

    #[test_log::test]
    fn test_product_base() {
        // \s. \x. (s.1 + x, s.2 + 2*x)

        let elem_1 = binop!(
            BinOpKinds::Add,
            tuple_access!(Var(sym!(s)), 0),
            Var(sym!(x))
        );

        let elem_2 = binop!(
            BinOpKinds::Add,
            tuple_access!(Var(sym!(s)), 1),
            binop!(BinOpKinds::Mul, Num(2), Var(sym!(x)))
        );

        let body = Tuple(vec![elem_1, elem_2]);
        let func = body.bind_params(vec![
            param!(s, Type::Tuple(vec![Type::Num, Type::Num])),
            param!(x, Type::Num),
        ]);

        let decomp = vd_default(&func, &TypeEnv::default(), Default::default());
        assert_eq!(decomp.unwrap().to_string(), "(num, num)((λ(s_0: num, x: num). (s_0 + x))*(λ(arg: (num, num)). arg._0), (λ(s_1: num, x: num). (s_1 + (x * 2)))*(λ(arg: (num, num)). arg._1))");
    }

    #[rstest]
    #[case::trivial_var_1(var!(s), TypeEnv::new(vec![(sym!(s), PolyType::Mono(tlist!(Type::Num)))]), "[num][v0, true, (v0 in s)]")]
    #[case::trivial_var_2(var!(s), TypeEnv::new(vec![(sym!(s), PolyType::Mono(tset!(Type::Num)))]), "{num}[v0, true, (v0 in s)]")]
    #[case::trivial_var_3(var!(s), TypeEnv::new(vec![(sym!(s), PolyType::Mono(tmap!(Type::Bool, Type::Num)))]), "{bool: num}[v1, true, ((v0, v1) in s)]")]
    #[case::trivial_abs_1(
        var!(s).bind_params(vec![param!(s, tlist!(Type::Num)), param!(x, Type::Num)]),
        TypeEnv::default(),
        "[num][λ(v0: num, x: num). v0, λ(v0: num). true, (Id)]"
    )]
    #[case::trivial_abs_2(
        var!(s).bind_params(vec![param!(s, tset!(Type::Num)), param!(x, Type::Num)]),
        TypeEnv::default(),
        "{num}[λ(v0: num, x: num). v0, λ(v0: num). true, (Id)]"
    )]
    #[case::trivial_abs_3(
        var!(s).bind_params(vec![param!(s, tmap!(Type::Bool, Type::Num)), param!(x, Type::Num)]),
        TypeEnv::default(),
        "{bool: num}[λ(v0: bool, v1: (bool, num), x: num). v1, λ(v1: (bool, num)). true, (Id)]"
    )]
    #[case::simple_abs_1(
        var!(map).call(vec![
            binop!(BinOpKinds::Add, sym!(v).into(), sym!(x).into()).bind_params(vec![param!(v, Type::Num)]),
            sym!(xs).into(),
        ]).bind_params(vec![param!(xs, tlist!(Type::Num)), param!(x, Type::Num)]),
        TypeEnv::default(),
        "[num][λ(v0: num, x: num). (x + v0), λ(v0: num). true, (Id)]")]
    #[case::simple_abs_2(
        var!(map_set).call(vec![
            binop!(BinOpKinds::Add, sym!(v).into(), sym!(x).into()).bind_params(vec![param!(v, Type::Num)]),
            sym!(xs).into(),
        ]).bind_params(vec![param!(xs, tset!(Type::Num)), param!(x, Type::Num)]),
        TypeEnv::default(),
        "{num}[λ(v0: num, x: num). (x + v0), λ(v0: num). true, (Id)]")]
    #[case::counter(
        var!(map_values)
            .call(vec![
                ite!(
                    binop!(BinOpKinds::Eq, var!(x), var!(k)),
                    tuple!(true.into(), binop!(BinOpKinds::Add, tuple_access!(var!(v), 1), 1.into())),
                    var!(v)
                )
                .bind_params(vec![param!(k, Type::Num), param!(v, ttuple!(Type::Bool, Type::Num))]),
                var!(counter),
            ])
            .bind_params(vec![
                param!(counter, tmap!(Type::Num, Type::Num)),
                param!(x, Type::Num),
            ]),
        TypeEnv::default(),
        "{num: num}[(bool, num)((λ(v0: num, v1_0: bool, x: num). if (x = v0) then true else v1_0)*(λ(arg: (bool, num)). arg._0), (λ(v0: num, v1_1: num, x: num). if (x = v0) then (1 + v1_1) else v1_1)*(λ(arg: (bool, num)). arg._1)), λ(v1: (bool, num)). true, (Id)]"
    )]
    #[case::zip_base(
        var!(zip).call(vec![var!(xs).into(), var!(ys).into()]).bind_params(vec![
            param!(xs, tlist!(Type::Num)),
            param!(ys, tlist!(Type::Num)),
        ]),
        TypeEnv::default(),
        "[([num], [num])][λ(v0: num, v1: num). (v0, v1), λ(v0: num, v1: num). (true && true), (Id, Id)]"
    )]
    #[case::zip_vector_sum(
        var!(map).call(vec![
            binop!(BinOpKinds::Add, sym!(x).into(), sym!(y).into()).bind_lets(vec![
                (sym!(x), tuple_access!(sym!(v).into(), 0)),
                (sym!(y), tuple_access!(sym!(v).into(), 1)),
            ]).bind_params(vec![param!(v, ttuple![Type::Num, Type::Num])]),
            var!(zip).call(vec![var!(xs).into(), var!(ys).into()])
        ]).bind_params(vec![
            param!(xs, tlist!(Type::Num)),
            param!(ys, tlist!(Type::Num)),
        ]),
        TypeEnv::default(),
        "[num][λ(v0: num, v1: num). (v0 + v1), λ(v0: num, v1: num). (true && true), (Id, Id)]"
    )]
    fn test_collection(#[case] input_expr: Expr, #[case] env: TypeEnv, #[case] expected: &str) {
        assert_eq!(
            vd_default(&input_expr, &env, Default::default())
                .unwrap()
                .to_string(),
            expected
        );
    }

    #[test_log::test]
    fn test_collection_rewrite() {
        let _acc_n = tuple_access!(var!(s), 0);
        let acc_map = tuple_access!(var!(s), 1);

        let x_name = tuple_access!(var!(x), 0);
        let x_n = tuple_access!(var!(x), 1);
        let x_count = tuple_access!(var!(x), 2);

        let f = tuple![
            x_n,
            map_assign!(
                acc_map.clone(),
                x_name.clone(),
                binop!(
                    BinOpKinds::Add,
                    map_access!(acc_map.clone(), x_name.clone()),
                    x_count
                )
            ),
        ]
        .bind_params(vec![
            (sym!(s), ttuple![Type::Num, tmap!(Type::Num, Type::Num)]),
            (sym!(x), ttuple![Type::Num, Type::Num, Type::Num]),
        ]);

        assert_eq!(f.to_string(), "λ(s: (num, {num: num}), x: (num, num, num)). (x._1, s._1[x._0 <- (s._1[x._0] + x._2)])");

        let decomp = vd_default(&f, &TypeEnv::default(), Default::default()).unwrap();
        assert_eq!(decomp.to_string(), "(num, {num: num})(((λ(s_0: num, x_1: num). x_1)*(λ(arg: (num, num, num)). arg._1))*(λ(arg: (num, {num: num})). arg._0), {num: num}[(bool, num)(((λ(v0: num, v1_0: bool, x_02_0: num). if (x_02_0 = v0) then true else v1_0)*(λ(arg: (num, num, num)). arg._0))*(λ(arg: (bool, num)). arg._0), ((λ(v0: num, v1_1: num, x_02: (num, num)). if (x_02._0 = v0) then (x_02._1 + v1_1) else v1_1)*(λ(arg: (num, num, num)). (arg._0, arg._2)))*(λ(arg: (bool, num)). arg._1)), λ(v1: (bool, num)). true, (λ(arg: (num, {num: num})). arg._1)])");
    }

    #[test_log::test]
    fn test_nondecomposable_collection() {
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
        let f: Expr = tuple![res1, res2, res3].bind_params(vec![
            param!(
                map_output,
                ttuple![tlist![Type::Num], tlist![Type::Num], tlist![Type::Num]]
            ),
            param!(x, ttuple![Type::Num, Type::Num, Type::Num]),
        ]);

        let decomp = vd_default(&f, &TypeEnv::default(), Default::default()).unwrap();
        println!("{}", decomp);
    }

    #[test_log::test]
    fn test_counter() {
        let f = map_assign!(
            var!(counter),
            var!(x),
            binop!(
                BinOpKinds::Add,
                map_access!(var!(counter), sym!(x).into()),
                Expr::Num(1)
            )
        )
        .bind_params(vec![
            (sym!(counter), tmap!(Type::Num, Type::Num)),
            (sym!(x), Type::Num),
        ]);

        let decomp = vd_default(&f, &TypeEnv::default(), Default::default()).unwrap();
        println!("{}", decomp)
        // {num: num}[(bool, num)(λ(v0: num, v1_0: bool, x: num). if (x = v0) then true else v1_0*λ(arg: (bool, num)). arg._0, λ(v0: num, v1_1: num, x: num). if (x = v0) then (1 + v1_1) else v1_1*λ(arg: (bool, num)). arg._1), λ(v1: (bool, num)). true, (Id)]
    }

    #[test_log::test]
    fn test_bug1() {
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
        let f: Expr = map_call.bind_params(vec![
            param!(s, tlist!(bucket_t.clone())),
            param!(x, tlist!(bucket_t.clone())),
        ]);

        let decomp = vd_default(&f, &TypeEnv::default(), Default::default()).unwrap();
        println!("{}", decomp);
    }

    #[test_log::test]
    fn test_bug2() {
        let accumulator = tuple!(
            tuple_access!(var!(x), 0),
            tuple_access!(var!(s), 1) + tuple_access!(var!(x), 1),
            var!(min).call(vec![tuple_access!(var!(x), 1), tuple_access!(var!(s), 2),]),
            var!(max).call(vec![tuple_access!(var!(x), 1), tuple_access!(var!(s), 3),]),
            tuple_access!(var!(s), 4) + 1.into(),
        )
        .bind_params(vec![
            param!(
                s,
                ttuple!(
                    tlist!(Type::Num),
                    Type::Num,
                    Type::Num,
                    Type::Num,
                    Type::Num
                )
            ),
            param!(x, ttuple!(tlist!(Type::Num), Type::Num)),
        ]);

        let decomposed = vd_default(&accumulator, &TypeEnv::default(), Default::default()).unwrap();
        assert_eq!(
            decomposed.to_string(),
            "([num], num, num, num, num)(((λ(s_0: [num], x_0: [num]). x_0)*(λ(arg: ([num], num)). arg._0))*(λ(arg: ([num], num, num, num, num)). arg._0), ((λ(s_1: num, x_1: num). (s_1 + x_1))*(λ(arg: ([num], num)). arg._1))*(λ(arg: ([num], num, num, num, num)). arg._1), ((λ(s_2: num, x_1: num). min(x_1, s_2))*(λ(arg: ([num], num)). arg._1))*(λ(arg: ([num], num, num, num, num)). arg._2), ((λ(s_3: num, x_1: num). max(x_1, s_3))*(λ(arg: ([num], num)). arg._1))*(λ(arg: ([num], num, num, num, num)). arg._3), ((λ(s_4: num, x_1: num). (1 + s_4))*(λ(arg: ([num], num)). arg._1))*(λ(arg: ([num], num, num, num, num)). arg._4))"
        );
    }

    #[test_log::test]
    fn test_bug3() {
        let accumulator = tuple![
            tuple_access!(var!(buffer), 0) + 1.into(),
            tuple_access!(var!(buffer), 0),
            tuple_access!(tuple_access!(var!(input), 0), 1),
        ]
        .bind_params(vec![
            param!(buffer, ttuple!(Type::Num, Type::Num, Type::Num)),
            param!(
                input,
                ttuple!(
                    ttuple![Type::Num, Type::Num, Type::Num, Type::Num, Type::Num],
                    Type::Num
                )
            ),
        ]);

        let decomp = vd_default(&accumulator, &TypeEnv::default(), Default::default()).unwrap();
        println!("{}", decomp);
    }
}
