mod assumption;
mod refute;
mod sygus;

use std::collections::HashMap;

use crate::incr::{vd_default, DecomposedExpression, Destructor, InputSimplifiedFunction};
use crate::lang::{macros::*, simplify, Expr, IsCurriedFunction as _, IsFunction as _, Type};
use crate::Symbol;
use bitflags::bitflags;
pub use refute::{refute_homomorphism, refute_homomorphism2, refute_normalizer};
pub use sygus::synthesize_normalizer;
pub use sygus::SynthesisMode;
use tracing::{event, Level};

#[derive(Debug, Clone)]
pub enum NormalizerSynthesisFailure {
    Timeout,
    Refuted,
    CannotDecompose,
    Other(String),
}

#[derive(Debug, Clone, Copy, Default)]
pub enum FunctionHasFreeVar<'a> {
    Yes(Symbol, &'a Type),

    #[default]
    No,
}

#[derive(Default, Debug, Clone)]
pub struct Assumptions(pub Vec<String>);

impl Assumptions {
    pub fn instantiate(&self, vars: Vec<&str>) -> Assumptions {
        let mut assumptions = Assumptions::default();
        for assumption in self.0.iter() {
            for var in vars.iter() {
                let assumption = assumption.replace("$state_var", var);
                assumptions.0.push(assumption);
            }
        }
        assumptions
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct NormalizerConstraint: u8 {
        const None = 0b0000;
        const Commutative = 0b0001;
        const Inductive = 0b0010;
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct SynthesisFeatures: u8 {
        const None = 0b0000;

        /// Reduction of merge operator synthesis to normalizer synthesis for accumulator
        const Reduction = 0b0001;

        /// Type-directed decomposition (expression decomposition)
        /// no decomposition but apply normalizer rules if they syntactically match
        const Decomposition = 0b0010;

        /// Refutation rules
        const Refutation = 0b0100;

        /// Deductive synthesis rules for normalizer construction
        const Deduction = 0b1000;
    }
}

impl Default for SynthesisFeatures {
    fn default() -> Self {
        SynthesisFeatures::all()
    }
}

impl Default for NormalizerConstraint {
    fn default() -> Self {
        NormalizerConstraint::Commutative | NormalizerConstraint::Inductive
    }
}

impl From<&str> for NormalizerSynthesisFailure {
    fn from(s: &str) -> Self {
        s.to_string().into()
    }
}

impl From<String> for NormalizerSynthesisFailure {
    fn from(s: String) -> Self {
        NormalizerSynthesisFailure::Other(s)
    }
}

pub fn check_homomorphism(
    func_acc: &Expr,
    init: &Expr,
    features: SynthesisFeatures,
) -> Result<Expr, NormalizerSynthesisFailure> {
    let _top_level = Expr::create_top_level(func_acc.clone(), init.clone());

    if features.contains(SynthesisFeatures::Refutation) {
        event!(Level::INFO, "Refuting homomorphism...");
        if refute_normalizer(func_acc, init, Default::default())?
            || refute_homomorphism2(func_acc, init)?
        {
            return Err(NormalizerSynthesisFailure::Refuted);
        }
    }

    if features.contains(SynthesisFeatures::Reduction) {
        if features.contains(SynthesisFeatures::Deduction) {
            event!(Level::INFO, "Decomposing accumulator...");
            let decomp = vd_default(
                func_acc,
                &Default::default(),
                features.contains(SynthesisFeatures::Decomposition).into(),
            )?;

            event!(Level::INFO, "Synthesizing normalizer...");
            match synthesize_normalizer_from_decomp(&decomp, init, Default::default(), features) {
                Ok(normalizer) => Ok(simplify(&normalizer)),
                Err(NormalizerSynthesisFailure::CannotDecompose) => synthesize_normalizer(
                    func_acc,
                    init,
                    Default::default(),
                    Default::default(),
                    Default::default(),
                ),
                err => err,
            }
        } else {
            // no deduction
            // don't need to refute here because we've already done it
            synthesize_normalizer(
                func_acc,
                init,
                Default::default(),
                Default::default(),
                Default::default(),
            )
        }
    } else {
        // no reduction; use the merge operator spec (foldl)
        synthesize_normalizer(
            func_acc,
            init,
            Default::default(),
            Default::default(),
            SynthesisMode::MergeOperator {
                num_left_args: 10,
                num_right_args: 10,
            },
        )
    }
}

fn check_free_variable(func: &Expr) -> (&Expr, FunctionHasFreeVar) {
    let (_, mut params) = func.uncurry_lambda();
    match params.len() {
        3 => {
            let Expr::Lambda { body, .. } = func else {
                panic!("unexpected expression: {:?}", func);
            };
            let (n, t) = params.remove(0);
            (body.as_ref(), FunctionHasFreeVar::Yes(n, t))
        }
        2 => (func, FunctionHasFreeVar::No),
        _ => panic!("unexpected number of parameters: {:?}", params),
    }
}

pub fn synthesize_or_refute_normalizer(
    func: &Expr,
    init: &Expr,
    constraint: NormalizerConstraint,
    features: SynthesisFeatures,
) -> Result<Expr, NormalizerSynthesisFailure> {
    let (func, map_key_free_var) = check_free_variable(func);
    event!(
        Level::DEBUG,
        rule = "Syn-Leaf",
        init = init.to_string(),
        func = func.to_string(),
        free_var = format!("{:?}", map_key_free_var),
        constraint = format!("{:?}", constraint),
        "Begin synthesizing normalizer for leaf function"
    );
    if features.contains(SynthesisFeatures::Refutation)
        && constraint.contains(Default::default())
        && refute_normalizer(&func, &init, map_key_free_var)?
    {
        return Err(NormalizerSynthesisFailure::Refuted);
    } else {
        synthesize_normalizer(
            &func,
            &init,
            map_key_free_var,
            constraint,
            SynthesisMode::Normalizer,
        )
    }
}

fn synthesize_normalizer_isf(
    func: &InputSimplifiedFunction,
    init: &Expr,
    constraint: NormalizerConstraint,
    features: SynthesisFeatures,
) -> Result<Expr, NormalizerSynthesisFailure> {
    let mut destructors = func.get_destructors();
    if destructors.len() == 3 {
        destructors.remove(0);
    }
    let func = func.get_expr();

    let destructor = destructors.into_iter().next().cloned().unwrap();
    let init = destructor.clone().call(vec![init.clone()]);

    let state_type = match &destructor {
        Destructor::Id => {
            let (func, _) = check_free_variable(func);
            let (_, params) = func.uncurry_lambda();
            params.into_iter().next().unwrap().1.clone()
        }
        Destructor::Lambda(expr) => expr.uncurry_lambda().1[0].1.clone(),
    };

    let normalizer = synthesize_or_refute_normalizer(func, &init, constraint, features)?;
    event!(
        Level::DEBUG,
        rule = "Syn-Leaf",
        norm = normalizer.to_string(),
        "Synthesized normalizer for leaf function"
    );

    Ok(normalizer
        .call(vec![
            destructor.clone().call(vec![var!(s1)]),
            destructor.clone().call(vec![var!(s2)]),
        ])
        .bind_params(vec![param!(s1, state_type.clone()), param!(s2, state_type)]))
}

fn merge_subproblems(fs: Vec<InputSimplifiedFunction>) -> InputSimplifiedFunction {
    let fs = fs
        .into_iter()
        .map(|f| simplify(&f.to_function()))
        .collect::<Vec<_>>();

    let types = {
        let f = fs.iter().next().unwrap();
        let (_, params) = f.uncurry_lambda();

        params
            .into_iter()
            .enumerate()
            .map(|(i, (_, t))| (Symbol::from(format!("merged_{}", i)), t.clone()))
            .collect::<Vec<_>>()
    };

    let merged = InputSimplifiedFunction::Id(Expr::Tuple(
        fs.into_iter()
            .map(|f| f.call(types.iter().map(|(ss, _)| Expr::from(*ss)).collect()))
            .collect(),
    ));

    types
        .into_iter()
        .rev()
        .fold(merged, |acc, (s, t)| acc.abs(s, t, Destructor::Id))
}

fn is_valid_accumulator(func: &Expr) -> bool {
    use crate::lang::{infer, TypeEnv};

    let (func, _) = check_free_variable(func);
    let Ok(acc_type) = infer(&TypeEnv::default(), func) else {
        return false;
    };
    let (param_types, ret_type) = acc_type.uncurry_fn();
    let state_type = param_types.into_iter().next().unwrap();

    *ret_type == *state_type
}

fn get_destructor_index(destructor: &Destructor) -> Vec<usize> {
    let Destructor::Lambda(func) = destructor else {
        panic!("unexpected destructor: {:?}", destructor);
    };

    let mut indices = vec![];
    let (body, params) = func.uncurry_lambda();
    assert!(params.len() == 1);
    let (param, _) = params.into_iter().next().unwrap();

    body.into_iter().for_each(|expr| match expr {
        Expr::TupleAccess(v, i) if matches!(v.as_ref(), Expr::Var(v) if *v == param) => {
            indices.push(*i)
        }

        _ => (),
    });

    indices.sort();
    indices
}

pub fn synthesize_normalizer_from_decomp(
    decomp: &DecomposedExpression,
    init: &Expr,
    constraint: NormalizerConstraint,
    features: SynthesisFeatures,
) -> Result<Expr, NormalizerSynthesisFailure> {
    event!(
        Level::DEBUG,
        decomp = decomp.to_string(),
        init = init.to_string(),
        "Synthesizing normalizer"
    );

    match decomp {
        DecomposedExpression::Func(func) => {
            synthesize_normalizer_isf(func, init, constraint, features)
        }

        DecomposedExpression::Product { typ, decomps } => {
            assert!(matches!(typ, Type::Tuple(..)));

            let decomps = decomps.iter().enumerate().collect::<HashMap<_, _>>();
            let valid_decomps = decomps
                .iter()
                .filter(|&(_, v)| {
                    let DecomposedExpression::Func(func) = v else {
                        return true;
                    };

                    is_valid_accumulator(func.get_expr())
                })
                .map(|(&k, &v)| (k, v))
                .collect::<HashMap<_, _>>();

            if valid_decomps.len() != decomps.len()
                && !features.contains(SynthesisFeatures::Decomposition)
            {
                return Err(NormalizerSynthesisFailure::CannotDecompose);
            }

            let reconstructed_decomps = decomps
                .iter()
                .filter(|&(k, _)| !valid_decomps.contains_key(&k))
                .map(|(&k, &v)| {
                    let DecomposedExpression::Func(func) = v else {
                        unreachable!()
                    };

                    let (_, params) = func.get_expr().uncurry_lambda();
                    let destructor = func.get_destructors();
                    let destructor_state = destructor[0];
                    let destructor_input = destructor[1];
                    let state_indices = match destructor_state {
                        Destructor::Id => (0..decomps.len()).into_iter().collect(),
                        Destructor::Lambda(_) => get_destructor_index(destructor_state),
                    };

                    let dummy_state = {
                        let param_name = params[0].0;
                        let mut v = vec![];
                        let mut tup_idx = 0;
                        for i in 0..decomps.len() {
                            if state_indices.contains(&i) {
                                v.push(tuple_access!(param_name.into(), tup_idx));
                                tup_idx += 1;
                            } else {
                                v.push(Expr::Num(0));
                            }
                        }
                        Expr::Tuple(v)
                    };
                    let dummy_input = {
                        let param_name = params[1].0;
                        match destructor_input {
                            Destructor::Id => Expr::Var(param_name),
                            Destructor::Lambda(func) => {
                                let (_, params) = func.uncurry_lambda();
                                assert!(params.len() == 1, "unexpected number of parameters");
                                let (_, Type::Tuple(ts)) = params.into_iter().next().unwrap()
                                else {
                                    panic!("unexpected parameter type");
                                };

                                let input_indices = get_destructor_index(destructor_input);
                                let mut v = vec![];
                                let mut tup_idx = 0;
                                for i in 0..ts.len() {
                                    if input_indices.contains(&i) {
                                        v.push(if input_indices.len() == 1 {
                                            param_name.into()
                                        } else {
                                            tuple_access!(param_name.into(), tup_idx)
                                        });
                                        tup_idx += 1;
                                    } else {
                                        v.push(Expr::Num(0));
                                    }
                                }

                                Expr::Tuple(v)
                            }
                        }
                    };

                    let func = Expr::Tuple({
                        state_indices
                            .into_iter()
                            .map(|i| {
                                let func = decomps[&i].get_isf().clone().to_function();
                                let func =
                                    func.call(vec![dummy_state.clone(), dummy_input.clone()]);
                                simplify(&func)
                            })
                            .collect()
                    })
                    .bind_params(params.into_iter().map(|(s, t)| (s, t.clone())).collect());

                    let func = InputSimplifiedFunction::Id(simplify(&func));
                    let func = InputSimplifiedFunction::Abs(
                        Box::new(InputSimplifiedFunction::Abs(
                            Box::new(func),
                            destructor_input.clone(),
                        )),
                        destructor_state.clone(),
                    );

                    (k, DecomposedExpression::Func(func))
                })
                .collect::<HashMap<_, _>>();

            let valid_decomps = valid_decomps
                .into_iter()
                .chain(reconstructed_decomps.iter().map(|(k, v)| (*k, v)))
                .collect::<HashMap<_, _>>();

            let mut normalizers = vec![];
            let mut errors = vec![];
            for (i, decomp) in valid_decomps.iter() {
                match synthesize_normalizer_from_decomp(decomp, init, constraint, features) {
                    Ok(normalizer) => normalizers.push((i, normalizer)),
                    Err(err) => errors.push((i, err)),
                }
            }
            let normalizers = normalizers.into_iter().collect::<HashMap<_, _>>();
            let errors = errors.into_iter().collect::<HashMap<_, _>>();

            if normalizers.is_empty() && !features.contains(SynthesisFeatures::Refutation) {
                return Err(NormalizerSynthesisFailure::CannotDecompose);
            }

            if normalizers.is_empty()
                || errors.iter().any(|(i, e)| {
                    reconstructed_decomps.contains_key(i)
                        && matches!(e, NormalizerSynthesisFailure::Refuted)
                })
            {
                return Err(errors
                    .into_iter()
                    .map(|(_, e)| e)
                    .next()
                    .unwrap_or(NormalizerSynthesisFailure::Refuted));
            }

            let params = normalizers
                .iter()
                .next()
                .unwrap()
                .1
                .uncurry_lambda()
                .1
                .into_iter()
                .map(|(s, t)| (s, t.clone()))
                .collect();

            if normalizers.len() == decomps.len() {
                let mut normalizers = normalizers.into_iter().collect::<Vec<_>>();
                normalizers.sort_by(|(i1, _), (i2, _)| i1.cmp(i2));
                return Ok(Expr::Tuple(
                    normalizers
                        .into_iter()
                        .map(|(_, n)| {
                            let (body, _) = n.uncurry_lambda();
                            body.clone()
                        })
                        .collect(),
                )
                .bind_params(params));
            }

            if !features.contains(SynthesisFeatures::Decomposition) {
                return Err(NormalizerSynthesisFailure::CannotDecompose);
            }

            // original function
            let func_acc = {
                let subproblems = (0..decomps.len())
                    .into_iter()
                    .map(|i| match decomps[&i] {
                        DecomposedExpression::Func(func) => Ok(func.clone()),
                        _ => Err(NormalizerSynthesisFailure::Other(format!(
                            "Partial Norm Optimization: Unsupported decomposition\n{}",
                            decomps[&i]
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                merge_subproblems(subproblems)
            };

            let &[(s1, _), (s2, _)] = params.as_slice() else {
                panic!("unexpected number of parameters: {:?}", params);
            };

            let partial_norm = Expr::Tuple({
                (0..decomps.len())
                    .into_iter()
                    .map(|i| {
                        if normalizers.contains_key(&i) {
                            return Ok(normalizers[&i].clone().call(vec![s1.into(), s2.into()]));
                        }

                        if !matches!(errors[&i], NormalizerSynthesisFailure::Refuted) {
                            return Err(errors[&i].clone());
                        }

                        let h1 = synthesize_normalizer_isf(
                            decomps[&i].get_isf(),
                            init,
                            NormalizerConstraint::Commutative,
                            features,
                        )?;

                        let h2 = synthesize_normalizer_isf(
                            decomps[&i].get_isf(),
                            init,
                            NormalizerConstraint::Inductive,
                            features,
                        )?;

                        Ok(ite!(
                            var!(merge).call(vec![s1.into(), s2.into()]),
                            h1.call(vec![s1.into(), s2.into()]),
                            h2.call(vec![s1.into(), s2.into()])
                        ))
                    })
                    .collect::<Result<Vec<Expr>, NormalizerSynthesisFailure>>()?
            })
            .bind_params(params);
            let partial_norm = simplify(&partial_norm);

            event!(
                Level::INFO,
                partial_norm = partial_norm.to_string(),
                func_acc = func_acc.to_string(),
                "synthesize normalizer from partial result"
            );

            let (func_acc, free_var) = check_free_variable(func_acc.get_expr());
            let cond = synthesize_normalizer(
                func_acc,
                init,
                free_var,
                Default::default(),
                SynthesisMode::NormalizerFromPartialResult {
                    ret_type: Type::Bool,
                    partial_norm: partial_norm.clone(),
                },
            )?;
            let norm = partial_norm.bind_lets(vec![(sym!(merge), cond)]);
            Ok(simplify(&norm))
        }

        DecomposedExpression::Collection {
            decomp,
            iter_type,
            destructors,
            ..
        } => {
            // always use the destructor for the first parameter, i.e., the accumulator
            let destructor = destructors.iter().next().unwrap();

            let init = match init {
                Expr::Cons { head, .. } => *head.clone(),
                _ if matches!(iter_type, Type::Map(..)) => {
                    tuple!(
                        false.into(),
                        iter_type.element_type().default_value().into()
                    )
                }
                _ => iter_type.element_type().default_value().into(),
            };

            let normalizer =
                synthesize_normalizer_from_decomp(decomp, &init, constraint, features)?;
            Ok(var!(map)
                .call(vec![
                    normalizer,
                    var!(outer_join).call(vec![
                        destructor.clone().call(vec![var!(xs)]),
                        destructor.clone().call(vec![var!(ys)]),
                    ]),
                ])
                .bind_params(vec![
                    param!(xs, iter_type.clone()),
                    param!(ys, iter_type.clone()),
                ]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        check_homomorphism, refute_normalizer, synthesize_normalizer, FunctionHasFreeVar,
        NormalizerConstraint,
    };
    use crate::{
        lang::{macros::*, BinOpKinds, Expr, IsCurriedFunction, Type},
        syn::synthesize_or_refute_normalizer,
    };

    #[test_log::test]
    fn test_scalar() {
        /*
           f s x = s + 1
           h a b = a + b
        */
        let expr = binop!(BinOpKinds::Add, Expr::Num(1), sym!(s).into());
        let func = expr.bind_params(vec![("s".into(), Type::Num), ("x".into(), Type::Num)]);

        let init = Expr::Num(0);
        let output = synthesize_normalizer(
            &func,
            &init,
            Default::default(),
            Default::default(),
            Default::default(),
        )
        .unwrap()
        .to_string();

        assert_eq!(output, "λ(val_l: num, val_r: num). (val_l + val_r)")
    }

    #[test_log::test]
    fn test_trivial() {
        let expr = Expr::from(0).bind_params(vec![param!(s, Type::Num), param!(x, Type::Num)]);
        let init = Expr::from(0);
        let output = synthesize_normalizer(
            &expr,
            &init,
            Default::default(),
            Default::default(),
            Default::default(),
        )
        .unwrap()
        .to_string();
        assert_eq!(output, "λ(val_l: num, val_r: num). 0")
    }

    #[test_log::test]
    fn test_tuple() {
        let f = tuple!(
            tuple_access!(var!(s), 0) + var!(x),
            tuple!(
                tuple_access!(tuple_access!(var!(s), 1), 0) + 1.into(),
                tuple_access!(tuple_access!(var!(s), 1), 1) + var!(x),
            )
        )
        .bind_params(vec![
            param!(s, ttuple![Type::Num, ttuple!(Type::Num, Type::Num)]),
            param!(x, Type::Num),
        ]);

        let init = tuple!(0.into(), tuple!(0.into(), 0.into()));

        let output = check_homomorphism(&f, &init, Default::default())
            .unwrap()
            .to_string();
        assert_eq!(output, "λ(s1: (num, (num, num)), s2: (num, (num, num))). ((s1._0 + s2._0), ((s1._1._0 + s2._1._0), (s1._1._1 + s2._1._1)))")
    }

    #[test_log::test]
    fn test_list() {
        /*
           f xs x = map (+x) xs

           h as bs = map2 (+) as bs
        */

        let f = var!(map)
            .call(vec![
                (var!(v) + var!(x)).bind_params(vec![param!(v, Type::Num)]),
                sym!(xs).into(),
            ])
            .bind_params(vec![param!(xs, tlist!(Type::Num)), param!(x, Type::Num)]);

        let init = vec![0; 5].into();

        let output = check_homomorphism(&f, &init, Default::default())
            .unwrap()
            .to_string();
        assert_eq!(
            output,
            "λ(xs: [num], ys: [num]). map(λ(s1: num, s2: num). (s1 + s2), outer_join(xs, ys))"
        )
    }

    #[test_log::test]
    fn test_map_concat() {
        let f = map_assign!(var!(m), var!(x), var!(x)).bind_params(vec![
            param!(m, tmap!(Type::Num, Type::Num)),
            param!(x, Type::Num),
        ]);
        let init = var!(empty_map);

        let output = check_homomorphism(&f, &init, Default::default()).unwrap();
        println!("Normalizer: \n{}", output);
    }

    #[test_log::test]
    fn test_map() {
        /*
           f counter x = counter[x <- counter[x] + 1]

           h c1 c2 = {k: c1[k] + c2[k] | k in keys(c1) /\ keys(c2)}
        */

        let f = map_assign!(
            var!(counter),
            var!(x),
            map_access!(var!(counter), var!(x)) + 1.into()
        )
        .bind_params(vec![
            param!(counter, tmap!(Type::Num, Type::Num)),
            param!(x, Type::Num),
        ]);

        let init = var!(empty_map);

        let output = check_homomorphism(&f, &init, Default::default())
            .unwrap()
            .to_string();
        assert_eq!(output, "λ(xs: {num: num}, ys: {num: num}). map(λ(s1: (bool, num), s2: (bool, num)). (or(s1._0, s2._0), (s1._1 + s2._1)), outer_join(xs, ys))")
    }

    #[test_log::test]
    fn test_set() {
        let state_type = tset!(ttuple!(Type::Num, Type::Num, Type::Num));
        let input_type = ttuple!(Type::Num, Type::Num, Type::Num);

        let f = var!(set_add)
            .call(vec![var!(x), var!(s)])
            .bind_params(vec![param!(s, state_type), param!(x, input_type)]);
        let init = var!(empty_set);

        let output = check_homomorphism(&f, &init, Default::default())
            .unwrap()
            .to_string();
        assert_eq!(
            output,
            "λ(s1: {(num, num, num)}, s2: {(num, num, num)}). union(s1, s2)"
        )
    }

    #[test_log::test]
    fn test_ite() {
        let f = ite!(expr_eq!(var!(s), 0.into()), var!(x), var!(s))
            .bind_params(vec![param!(s, Type::Num), param!(x, Type::Num)]);

        let init = Expr::Num(0);
        let output = check_homomorphism(&f, &init, Default::default())
            .unwrap()
            .to_string();
        assert_eq!(output, "λ(s1: num, s2: num). ite((s1 = 0), s2, s1)")
    }

    #[test_log::test]
    fn test_dependent_product() {
        let cond = binop!(
            BinOpKinds::Lt,
            tuple_access!(var!(s), 2),
            tuple_access!(var!(x), 2)
        );
        let temp = ite!(cond, var!(x), var!(s));
        let f = temp.bind_params(vec![
            param!(s, ttuple![Type::Num, Type::Num, Type::Num]),
            param!(x, ttuple![Type::Num, Type::Num, Type::Num]),
        ]);

        let init = Expr::Tuple(vec![Expr::Num(0), Expr::Num(0), var!(_mn)]);
        let output =
            check_homomorphism(&f, &init, Default::default()).expect("normalizer synthesis failed");
        println!("Normalizer: \n{}", output);
    }

    #[test_log::test]
    fn test_dependent_product2() {
        let cond = !expr_eq!(var!(x), 2.into());
        let f = tuple![
            tuple_access!(var!(s), 0) + 1.into(),
            ite!(cond, tuple_access!(var!(s), 1), tuple_access!(var!(s), 0))
        ]
        .bind_params(vec![
            param!(s, ttuple![Type::Num, Type::Num]),
            param!(x, Type::Num),
        ]);

        let init = Expr::Tuple(vec![Expr::Num(0), Expr::Num(0)]);
        assert!(synthesize_or_refute_normalizer(
            &f,
            &init,
            NormalizerConstraint::all(),
            Default::default()
        )
        .is_err_and(|e| matches!(e, crate::syn::NormalizerSynthesisFailure::Refuted)))
    }

    #[test_log::test]
    fn test_map_values_alternative() {
        // f m x = m[x <- x]
        // illustrates that null flag is necessary
        let f = ite!(expr_eq!(var!(k), var!(x)), var!(x), var!(s))
            .bind_params(vec![param!(s, Type::Num), param!(x, Type::Num)]);
        let init = 0.into();
        let refute =
            refute_normalizer(&f, &init, FunctionHasFreeVar::Yes(sym!(k), &Type::Num)).unwrap();
        assert!(refute);

        // complete example
        let f = tuple!(true.into(), var!(x)).bind_params(vec![
            param!(s, ttuple![Type::Bool, Type::Num]),
            param!(x, Type::Num),
        ]);

        let init = tuple!(false.into(), 0.into());
        let output = synthesize_normalizer(
            &f,
            &init,
            FunctionHasFreeVar::Yes(sym!(k), &Type::Num),
            Default::default(),
            Default::default(),
        )
        .expect("normalizer synthesis failed");
        println!("Normalizer: \n{}", output);
    }

    #[test_log::test]
    fn test_map_values_alternative2() {
        // f m x = if (x in m) then m else m[x <- x]
        let f = ite!(
            expr_eq!(var!(k), var!(x)),
            ite!(
                tuple_access!(var!(s), 0),
                var!(s),
                tuple!(true.into(), var!(x))
            ),
            var!(s)
        )
        .bind_params(vec![
            param!(s, ttuple![Type::Bool, Type::Num]),
            param!(x, Type::Num),
        ]);

        let init = tuple!(false.into(), 0.into());

        let refute =
            refute_normalizer(&f, &init, FunctionHasFreeVar::Yes(sym!(k), &Type::Num)).unwrap();
        assert!(!refute);

        let output = synthesize_normalizer(
            &f,
            &init,
            FunctionHasFreeVar::Yes(sym!(k), &Type::Num),
            Default::default(),
            Default::default(),
        )
        .expect("normalizer synthesis failed");
        println!("Normalizer: \n{}", output);
    }

    #[test_log::test]
    fn test_set_nondecomposable() {
        let f = lambda!(s : ttuple!(Type::Num, tset!(Type::Num)) =>
            lambda!(x : Type::Num => tuple!(var!(x), var!(set_add).call(vec![var!(x), tuple_access!(var!(s), 1)])))
        );
        let init = tuple!(0.into(), var!(empty_set));
        let output = check_homomorphism(&f, &init, Default::default()).unwrap();
        println!("Normalizer: \n{}", output);
    }

    #[test_log::test]
    fn test_trivial_nondecomposable() {
        let f = tuple!(true.into(), var!(x)).bind_params(vec![
            param!(s, ttuple![Type::Bool, Type::Num]),
            param!(x, Type::Num),
        ]);

        let init = tuple!(false.into(), 0.into());

        let result = check_homomorphism(&f, &init, Default::default()).unwrap();
        println!("Normalizer: \n{}", result);
    }

    #[test_log::test]
    fn test_degenerate_program() {
        let f = var!(x).bind_params(vec![param!(s, Type::Num), param!(x, Type::Num)]);
        let init = 0.into();

        let result = synthesize_normalizer(
            &f,
            &init,
            Default::default(),
            NormalizerConstraint::Inductive,
            Default::default(),
        )
        .unwrap();
        println!("Normalizer: \n{}", result);
    }
}
