use super::{BinOpKinds, Expr, Type, TypedEnv, UnaryOpKinds};
use crate::consts::{as_id_str, is_builtin_ids, sygus_builtin_requires_type};

const ACCESS_MAP: &str = "get_value";
const ASSIGN_MAP: &str = "set_value";

pub trait FormatSyGuS {
    fn format(&self) -> String;
}

impl FormatSyGuS for BinOpKinds {
    fn format(&self) -> String {
        match self {
            BinOpKinds::Concat => "seq.++".to_string(),
            BinOpKinds::And => "and".to_string(),
            BinOpKinds::Or => "or".to_string(),
            BinOpKinds::Div => "div".to_string(),
            _ => self.to_string(),
        }
    }
}

impl FormatSyGuS for UnaryOpKinds {
    fn format(&self) -> String {
        match self {
            UnaryOpKinds::Neg => "-".to_string(),
            _ => self.to_string(),
        }
    }
}

impl FormatSyGuS for Type {
    fn format(&self) -> String {
        match self {
            Type::Num => "Int".to_string(),
            Type::Bool => "Bool".to_string(),
            Type::Str => "String".to_string(),
            Type::Tuple(tys) => format!(
                "(Tuple {})",
                tys.iter().map(|t| t.format()).collect::<Vec<_>>().join(" ")
            ),
            Type::Set(t) => format!("(Set {})", t.format()),
            Type::List(t) => format!("(Seq {})", t.format()),
            Type::Map(k, v) => format!("(Set (Tuple {} {}))", k.format(), v.format()),
            Type::Var(_) => panic!("Type variable should not appear in SyGuS"),
            Type::Fn(_, _) => panic!("Function type should not appear in SyGuS"),
        }
    }
}

fn get_sygus_var_name(v: &str, t: &Type) -> String {
    match v {
        // list primitives
        "length" => "str.len".into(),

        // set primitives
        "set_add" => "set.insert".into(),
        "intersection" => "set.inter".into(),
        "union" => "set.union".into(),
        "empty_set" | "empty_map" => format!("(as set.empty {})", t.format()),
        "filter_set" => "set.filter".into(),
        "map_set" => "set.map".into(),
        "is_set_empty" => format!("isSetEmpty_{}", as_id_str(t.format().as_str())),

        // map primitives
        "concat_map" => format!("concatMap_{}", as_id_str(t.format().as_str())),
        "filter_values" => format!("filterValues_{}", as_id_str(t.format().as_str())),
        "map_values" => format!("mapValues_{}", as_id_str(t.format().as_str())),
        "contains_key" => format!("containsKey_{}", as_id_str(t.format().as_str())),
        ACCESS_MAP => format!("access_{}", as_id_str(t.format().as_str())),
        ASSIGN_MAP => format!("update_{}", as_id_str(t.format().as_str())),
        v => v.into(),
    }
}

pub fn format_define_fun(id: &str, expr: &Expr, env: &dyn TypedEnv) -> String {
    let Type::Fn(_, ret_type) = env.get_type(expr) else {
        panic!("Lambda should have function type");
    };
    let (body, params) = expr.uncurry_lambda();
    let (_, ret_type) = ret_type.uncurry_fn();

    let params = params
        .iter()
        .map(|(p, t)| format!("({} {})", p, t.format()))
        .collect::<Vec<_>>()
        .join(" ");

    format!(
        "(define-fun {} ({}) {} {})",
        id,
        params,
        ret_type.format(),
        (body, env).format()
    )
}

impl FormatSyGuS for (&Expr, &dyn TypedEnv) {
    fn format(&self) -> String {
        use Expr::*;
        match self.0 {
            Num(n) => {
                if *n < 0 {
                    format!("(- {})", n.abs())
                } else {
                    n.to_string()
                }
            }
            Var(v) => get_sygus_var_name(v.as_str(), &self.1.get_type(self.0)),
            Bool(b) => b.to_string(),
            Str(s) => format!("\"{}\"", s),
            UnaryOp(op, e) => format!("({} {})", op.format(), (e.as_ref(), self.1).format()),
            BinOp(op, e1, e2) => format!(
                "({} {} {})",
                op.format(),
                (e1.as_ref(), self.1).format(),
                (e2.as_ref(), self.1).format()
            ),
            Ite {
                cond,
                then_expr,
                else_expr,
            } => format!(
                "(ite {} {} {})",
                (cond.as_ref(), self.1).format(),
                (then_expr.as_ref(), self.1).format(),
                (else_expr.as_ref(), self.1).format()
            ),
            App { .. } => {
                let (func, args) = self.0.uncurry_call();
                let args_str = args
                    .iter()
                    .map(|a| (*a, self.1).format())
                    .collect::<Vec<_>>()
                    .join(" ");

                let func_str = match func {
                    Var(v) if is_builtin_ids(v.as_str()) => {
                        let typ = args
                            .iter()
                            .filter_map(|e| {
                                let t = self.1.get_type(e);
                                if t.is_collection() {
                                    Some(t)
                                } else {
                                    None
                                }
                            })
                            .next()
                            .or(if !sygus_builtin_requires_type(v.as_str()) {
                                // placeholder
                                Some(Type::Bool)
                            } else {
                                None
                            })
                            .unwrap();

                        get_sygus_var_name(v.as_str(), &typ)
                    }
                    _ => (func, self.1).format(),
                };

                format!("({} {})", func_str, args_str)
            }
            Lambda { .. } => {
                let (body, params) = self.0.uncurry_lambda();
                let params = params
                    .iter()
                    .map(|(p, t)| format!("({} {})", p, t.format()))
                    .collect::<Vec<_>>()
                    .join(" ");

                format!("(lambda ({}) {})", params, (body, self.1).format())
            }

            Let { name, expr, body } => {
                format!(
                    "(let (({} ({}))) ({}))",
                    name,
                    (expr.as_ref(), self.1).format(),
                    (body.as_ref(), self.1).format()
                )
            }

            Nil => format!("(as seq.empty {})", self.1.get_type(self.0).format()),
            Cons { head, tail } => {
                if matches!(tail.as_ref(), Nil) {
                    format!("(seq.unit {})", (head.as_ref(), self.1).format())
                } else {
                    format!(
                        "(seq.++ (seq.unit {}) {})",
                        (head.as_ref(), self.1).format(),
                        (tail.as_ref(), self.1).format()
                    )
                }
            }
            Tuple(els) => format!(
                "(tuple {})",
                els.iter()
                    .map(|e| (e, self.1).format())
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            TupleAccess(tup, idx) => format!(
                "((_ tuple.select {}) {})",
                idx,
                (tup.as_ref(), self.1).format()
            ),
            MapAssign { map, key, value } => {
                let map_type = self.1.get_type(map.as_ref());
                let func = get_sygus_var_name(ASSIGN_MAP, &map_type);

                format!(
                    "({} {} {} {})",
                    func,
                    (map.as_ref(), self.1).format(),
                    (key.as_ref(), self.1).format(),
                    (value.as_ref(), self.1).format()
                )
            }
            MapAccess { map, key } => {
                let map_type = self.1.get_type(map.as_ref());
                let func = get_sygus_var_name(ACCESS_MAP, &map_type);

                format!(
                    "({} {} {})",
                    func,
                    (map.as_ref(), self.1).format(),
                    (key.as_ref(), self.1).format()
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::{macros::*, BasicSubexprTypeMap, TypeEnv};
    use rstest::rstest;

    #[rstest]
    #[case::acc_1(binop!(BinOpKinds::Add, Expr::Num(1), var!(s)).bind_params(vec![param!(s, Type::Num), param!(x, Type::Num)]), "(define-fun f ((s Int) (x Int)) Int (+ 1 s))")]
    fn test_simple_func(#[case] func: Expr, #[case] expected: &str) {
        let map: &dyn TypedEnv = &BasicSubexprTypeMap::new(&func, TypeEnv::default());
        let result = format_define_fun("f", &func, map);
        assert_eq!(result, expected);
    }
}
