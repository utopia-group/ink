use super::{Binding, DecomposedExpression, Destructor, Transformable};
use crate::lang::{macros::*, Expr, IsFunction, LangAnalyzer, Type};

use crate::Symbol;
use std::collections::{HashMap, HashSet};
use tracing::{event, Level};

/// return the set of tuple indices used in the given decomposed expression
fn find_tuple_index_usage(decomp: &DecomposedExpression, tuple_var: Symbol) -> HashSet<usize> {
    let mut d_exprs = decomp.clone().extract_exprs();
    if let DecomposedExpression::Collection { bindings, .. } = decomp {
        for (_, expr) in bindings {
            d_exprs.push(expr.clone());
        }
    }

    d_exprs
        .into_iter()
        .flat_map(|e| find_tuple_index_usage_in_expr(&e, tuple_var))
        .collect::<HashSet<_>>()
}

/// build a map from the index of the product to the indices of the dependent products
fn _build_product_dependent_closure_map(
    tuple_var: Symbol,
    subdecomps: Vec<&DecomposedExpression>,
) -> HashMap<usize, Vec<usize>> {
    let mut index_map = HashMap::new();

    for (i, decomp) in subdecomps.iter().enumerate() {
        let indices = find_tuple_index_usage(decomp, tuple_var)
            .into_iter()
            .filter(|x| *x != i)
            .collect::<Vec<_>>();

        index_map.insert(i, indices);
    }

    let mut changed = false;
    loop {
        let index_map_copy = index_map.clone();

        for (i, indices) in index_map.iter_mut() {
            let new_indices = indices
                .iter()
                .flat_map(|idx| index_map_copy.get(idx).unwrap().clone())
                .filter(|x| !indices.contains(x))
                .collect::<HashSet<_>>()
                .into_iter()
                .filter(|x| *x != *i)
                .collect::<Vec<_>>();

            if !new_indices.is_empty() {
                changed = true;
                indices.extend(new_indices);
            }
        }

        if !changed {
            break;
        }
    }

    for (i, indices) in index_map.iter_mut() {
        indices.sort();
        indices.insert(0, *i);
    }

    index_map
}

/// Find all tuple indices used in the given expression.
fn find_tuple_index_usage_in_expr(expr: &Expr, tuple_var: Symbol) -> HashSet<usize> {
    let mut analysis = LangAnalyzer::new(expr);

    let indices = expr
        .clone()
        .into_iter()
        .fold(HashSet::new(), |mut acc, expr| match expr {
            Expr::TupleAccess(_, idx) => {
                acc.insert(*idx);
                acc
            }

            _ => acc,
        });

    // Note: this is very inefficient but i don't know how to do it better
    expr.into_iter().fold(HashSet::new(), |mut acc, expr| {
        for idx in indices.iter() {
            let tuple_access = tuple_access!(tuple_var.into(), *idx);
            if analysis.check_eq(&expr, &tuple_access) {
                acc.insert(*idx);
            }
        }
        acc
    })
}

fn compose_destructors(d1: Destructor, d2: Destructor) -> Destructor {
    match d2 {
        Destructor::Id => d1,
        Destructor::Lambda(body) => {
            let Expr::Lambda {
                param,
                param_type,
                body,
            } = body
            else {
                panic!("unexpected lambda body: {:?}", body);
            };

            Destructor::Lambda(Expr::Lambda {
                param,
                param_type,
                body: Box::new(d1.call(vec![*body])),
            })
        }
    }
}

pub(super) fn simplify_function(
    param_name: Symbol,
    param_type: &Type,
    destructor: &Destructor,
    decomp: DecomposedExpression,
) -> DecomposedExpression {
    match decomp {
        // Expr and Function
        DecomposedExpression::Func(f) => {
            DecomposedExpression::Func(f.abs(param_name, param_type.clone(), destructor.clone()))
        }

        // Tuple-Inductive
        DecomposedExpression::Product { typ, decomps } if param_type.is_compound() => {
            let Type::Tuple(els) = param_type else {
                panic!("only tuple is supported for product. got: {:?}", param_type);
            };

            let decomps = decomps
                .into_iter()
                .enumerate()
                .map(move |(i, decomp)| {
                    // find all tuple indices used in the decomposed expression
                    let mut tuple_indices = find_tuple_index_usage(&decomp, param_name);

                    // edge case: if the tuple is not used in the decomposed expression
                    // we still need to create a new parameter
                    if tuple_indices.is_empty() {
                        tuple_indices.insert(i.min(els.len() - 1));
                    }

                    // edge case: if tuple_indices is everything
                    // use Id as the destructor
                    if tuple_indices.len() == els.len() {
                        // FIXME: potential infinite recursion??
                        return simplify_function(param_name, param_type, destructor, decomp);
                    }

                    let mut tuple_indices = tuple_indices.into_iter().collect::<Vec<_>>();
                    tuple_indices.sort();
                    let tuple_indices = tuple_indices;

                    let new_param_name = Symbol::from(format!(
                        "{}_{}",
                        param_name,
                        tuple_indices
                            .iter()
                            .map(|i| i.to_string())
                            .collect::<String>()
                    ));

                    // new expression to replace param_name
                    // either a tuple access (for single element) or a tuple
                    let (expr_map, new_destructor, new_type) = if tuple_indices.len() == 1 {
                        let idx = tuple_indices.into_iter().next().unwrap();
                        let orig_expr = tuple_access!(param_name.into(), idx);
                        let new_expr = new_param_name.into();

                        (
                            vec![(orig_expr, new_expr)],
                            Destructor::Lambda(
                                tuple_access!(sym!(arg).into(), idx)
                                    .bind_params(vec![param!(arg, Type::Tuple(els.clone()))]),
                            ),
                            els[idx].clone(),
                        )
                    } else {
                        let m = tuple_indices
                            .iter()
                            .enumerate()
                            .map(|(i, idx)| {
                                let orig_expr = tuple_access!(param_name.into(), *idx);
                                let new_expr = tuple_access!(new_param_name.into(), i);
                                (orig_expr, new_expr)
                            })
                            .collect();

                        let destructor = Destructor::Lambda(
                            Expr::Tuple(
                                tuple_indices
                                    .iter()
                                    .map(|i| tuple_access!(sym!(arg).into(), *i))
                                    .collect(),
                            )
                            .bind_params(vec![param!(arg, Type::Tuple(els.clone()))]),
                        );
                        let new_type = Type::Tuple(
                            tuple_indices
                                .iter()
                                .map(|i| els[*i].clone())
                                .collect::<Vec<_>>(),
                        );

                        (m, destructor, new_type)
                    };

                    let mut analyzer = LangAnalyzer::default();

                    // rewrite expression to use new_param_name
                    let decomp = decomp.transform(&mut |expr| {
                        for (orig_expr, new_expr) in expr_map.iter() {
                            if analyzer.check_eq(orig_expr, &expr) {
                                event!(
                                    Level::INFO,
                                    expr = expr.to_string(),
                                    new_expr = new_expr.to_string(),
                                    "replacing product expression"
                                );
                                return new_expr.clone();
                            }
                        }
                        expr
                    });

                    simplify_function(
                        new_param_name,
                        &new_type,
                        &compose_destructors(new_destructor, destructor.clone()),
                        decomp,
                    )
                })
                .collect::<Vec<_>>();

            DecomposedExpression::Product { typ, decomps }
        }

        // Tuple-Base
        DecomposedExpression::Product { typ, decomps } => {
            let decomps = decomps
                .into_iter()
                .map(|decomp| simplify_function(param_name, param_type, destructor, decomp))
                .collect();

            DecomposedExpression::Product { typ, decomps }
        }

        // Collection-Inductive
        DecomposedExpression::Collection {
            decomp,
            iter_type,
            predicate,
            destructors,
            bindings,
        } if bindings
            .iter()
            .any(|(_, c)| matches!(c, Expr::Var(v) if *v == param_name))
            && param_type.is_collection() =>
        {
            let (binding, _) = bindings
                .iter()
                .find(|(_, c)| matches!(c, Expr::Var(v) if *v == param_name))
                .unwrap()
                .clone();

            let iter_id_and_types = match binding {
                Binding::Single(v) => vec![(v, param_type.element_type().clone())],
                Binding::KeyValuePair(k, v) => {
                    let Type::Map(ty_k, ty_v) = param_type else {
                        panic!("expected map type, got: {:?}", param_type);
                    };

                    // (key_exists, value)
                    vec![(v, ttuple!(Type::Bool, *ty_v.clone())), (k, *ty_k.clone())]
                }
            };

            let decomp = iter_id_and_types.iter().fold(*decomp, |acc, (id, typ)| {
                simplify_function(*id, typ, &Destructor::Id, acc)
            });

            let (param, param_type) = iter_id_and_types.into_iter().next().unwrap();
            let predicate: Expr = Expr::Lambda {
                param,
                param_type,
                body: Box::new(predicate),
            };

            let bindings = bindings
                .into_iter()
                .filter(|(i, _)| *i != binding)
                .collect();

            let destructors = vec![destructor.clone()]
                .into_iter()
                .chain(destructors.into_iter())
                .collect();

            DecomposedExpression::Collection {
                decomp: Box::new(decomp),
                iter_type,
                predicate,
                destructors,
                bindings,
            }
        }

        // Collection-Base
        DecomposedExpression::Collection {
            decomp,
            iter_type,
            predicate,
            destructors,
            bindings,
        } => {
            let decomp = Box::new(simplify_function(
                param_name, param_type, destructor, *decomp,
            ));

            DecomposedExpression::Collection {
                decomp,
                iter_type,
                predicate,
                destructors,
                bindings,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::Expr::*;

    #[test_log::test]
    fn test_find_tuple_index_usage() {
        let expr = Tuple(vec![
            tuple_access!(sym!(x).into(), 0),
            tuple_access!(sym!(x).into(), 1),
            letv!(z, sym!(x).into(), tuple_access!(sym!(z).into(), 20)),
            tuple_access!(sym!(y).into(), 0),
        ]);

        let result = find_tuple_index_usage_in_expr(&expr, sym!(x));
        assert_eq!(result, vec![0, 1, 20].into_iter().collect());

        let result = find_tuple_index_usage_in_expr(&expr, sym!(y));
        assert_eq!(result, vec![0].into_iter().collect());
    }
}
