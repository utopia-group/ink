use std::collections::HashSet;

use tracing::{event, Level};

use crate::lang::{
    macros::*, BinOpKinds, Expr, IsCurriedFunction, LangAnalyzer, Transformable, Type, TypedEnv,
};

fn find_map_key_usage(expr: &Expr, map: &Expr) -> HashSet<Expr> {
    let mut analysis = LangAnalyzer::new(expr);

    let keys = expr
        .clone()
        .into_iter()
        .fold(HashSet::new(), |mut acc, expr| match expr {
            Expr::MapAccess { key, .. } => {
                acc.insert(*key.clone());
                acc
            }

            _ => acc,
        });

    expr.into_iter().fold(HashSet::new(), |mut acc, expr| {
        for key in keys.iter() {
            let map_access = Expr::MapAccess {
                map: Box::new(map.clone()),
                key: Box::new(key.clone()),
            };

            if analysis.check_eq(&expr, &map_access) {
                acc.insert(key.clone());
            }
        }
        acc
    })
}

fn rewrite_map_operations<T: TypedEnv>(expr: Expr, state: &T) -> Expr {
    expr.transform(&mut |expr| match expr {
        Expr::MapAssign { map, key, value } => {
            let Type::Map(ty_k, ty_v) = state.get_type(map.as_ref()) else {
                panic!("invalid map type: {:?}", map);
            };
            let key_usages = find_map_key_usage(&value, &map);
            if key_usages.len() > 1
                || (key_usages.len() == 1 && *key_usages.iter().next().unwrap() != *key)
            {
                // don't rewrite if there are multiple key usages
                return Expr::MapAssign { map, key, value };
            }

            let value_expr = if key_usages.is_empty() {
                *value
            } else {
                let orig_expr = map_access!(*map.clone(), *key.clone());
                let mut analyzer = LangAnalyzer::default();
                value.transform(&mut |expr| {
                    if analyzer.check_eq(&expr, &orig_expr) {
                        event!(Level::INFO, expr = expr.to_string(), "replacing key");
                        // (key_exists_in_map, value)
                        tuple_access!(var!(__v), 1)
                    } else {
                        expr
                    }
                })
            };

            var!(map_values).call(vec![
                ite!(
                    expr_eq!(*key.clone(), var!(__k)),
                    tuple!(true.into(), value_expr),
                    var!(__v)
                )
                .bind_params(vec![
                    param!(__k, *ty_k),
                    param!(__v, ttuple!(Type::Bool, *ty_v)),
                ]),
                rewrite_map_operations(*map, state),
            ])
        }

        _ => expr,
    })
}

pub fn preprocess<T: TypedEnv>(expr: Expr, state: &T) -> Expr {
    event!(Level::INFO, "Preprocessing expression: {}", expr);
    let funcs: Vec<(&str, fn(Expr, &T) -> Expr)> = vec![
        ("rewrite_map_operations", rewrite_map_operations),
        ("inline_let_bindings", |e, _| e.inline_let_bindings()),
    ];

    let preprocessed = funcs.into_iter().fold(expr, |acc, (name, func)| {
        event!(Level::INFO, "Applying function: {}", name);
        func(acc, state)
    });
    event!(Level::INFO, "Preprocessed expression: {}", preprocessed);
    preprocessed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::{BasicSubexprTypeMap, BinOpKinds, Type, TypeEnv};

    #[test_log::test]
    fn test_find_map_key_usage() {
        let _acc_n = tuple_access!(sym!(s).into(), 0);
        let acc_map = tuple_access!(sym!(s).into(), 1);

        let x_name = tuple_access!(sym!(x).into(), 0);
        let x_n = tuple_access!(sym!(x).into(), 1);
        let x_count = tuple_access!(sym!(x).into(), 2);

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
            ("s".into(), ttuple![Type::Num, tmap!(Type::Num, Type::Num)]),
            ("x".into(), ttuple![Type::Num, Type::Num, Type::Num]),
        ]);

        let result = find_map_key_usage(&f, &acc_map);
        assert!(result.contains(&x_name));
        assert_eq!(result.len(), 1);

        let typed_env = BasicSubexprTypeMap::new(&f, TypeEnv::default());
        let result = rewrite_map_operations(f, &typed_env);

        println!("Result: \n{}", result);
    }
}
