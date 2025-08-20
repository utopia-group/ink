use crate::lang::{
    infer, simplify, Expr, IsCurriedFunction, IsFunction, Transformable, Type, TypeEnv,
};
use core::fmt;

mod preprocess;
mod simp;
mod vd;

use crate::Symbol;
pub use vd::vd_default;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Destructor {
    /// identity function
    Id,

    /// other lambda expression
    Lambda(Expr),
}

impl fmt::Display for Destructor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Destructor::Id => write!(f, "Id"),
            Destructor::Lambda(expr) => write!(f, "{}", expr),
        }
    }
}

impl IsFunction for Destructor {
    type Ret = Expr;
    fn call(self, args: Vec<Expr>) -> Self::Ret {
        assert!(
            args.len() == 1,
            "invalid number of arguments for destructor"
        );

        match self {
            Destructor::Id => args[0].clone(),
            Destructor::Lambda(body) => simplify(&body.app(args[0].clone())),
        }
    }
}

#[derive(Debug, Clone)]
pub enum InputSimplifiedFunction {
    Id(Expr),
    Abs(Box<InputSimplifiedFunction>, Destructor),
}

impl InputSimplifiedFunction {
    pub fn abs(self, name: Symbol, typ: Type, destructor: Destructor) -> Self {
        let mut current = self;
        let mut destructors = vec![destructor];
        while let InputSimplifiedFunction::Abs(func, d) = current {
            destructors.push(d);
            current = *func;
        }

        let Self::Id(func) = current else {
            panic!("unexpected ISF: {:?}", current);
        };

        let func = func.bind_params(vec![(name, typ)]);
        destructors
            .into_iter()
            .rev()
            .fold(Self::Id(func), |func, d| Self::Abs(Box::new(func), d))
    }

    pub fn get_expr(&self) -> &Expr {
        match self {
            Self::Id(expr) => expr,
            Self::Abs(func, ..) => func.get_expr(),
        }
    }

    pub fn get_destructors(&self) -> Vec<&Destructor> {
        let mut current = self;
        let mut destructors = vec![];
        while let Self::Abs(func, d) = current {
            destructors.push(d);
            current = func;
        }
        destructors
    }

    pub fn to_function(self) -> Expr {
        fn transform_name(s: Symbol) -> Symbol {
            format!("__{}", s).into()
        }

        let expr = self.get_expr().clone();
        let destructors = self.get_destructors();

        let (_, params) = expr.uncurry_lambda();

        let new_params = params
            .into_iter()
            .zip(destructors)
            .map(|((p_name, p_type), d)| match d {
                Destructor::Id => (
                    transform_name(p_name),
                    p_type.clone(),
                    Expr::from(transform_name(p_name)),
                ),
                Destructor::Lambda(expr) => {
                    let p_name = transform_name(p_name);
                    let new_expr = expr.clone().app(Expr::Var(p_name));
                    let Type::Fn(p_type, _) = infer(&TypeEnv::default(), expr).unwrap() else {
                        panic!("unexpected type: {:?}", p_type);
                    };
                    (p_name, *p_type, new_expr)
                }
            })
            .collect::<Vec<_>>();

        expr.call(new_params.iter().map(|(_, _, e)| e.clone()).collect())
            .bind_params(new_params.into_iter().map(|(n, t, _)| (n, t)).collect())
    }
}

impl From<Expr> for InputSimplifiedFunction {
    fn from(expr: Expr) -> Self {
        Self::Id(expr)
    }
}

impl fmt::Display for InputSimplifiedFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Id(expr) => write!(f, "{}", expr),
            Self::Abs(func, Destructor::Id) => write!(f, "{}", func),
            Self::Abs(func, Destructor::Lambda(body)) => write!(f, "({})*({})", func, body),
        }
    }
}

/// transform the expression of an ISF
impl Transformable for InputSimplifiedFunction {
    type Item = Expr;
    fn transform(self, f: &mut impl FnMut(Self::Item) -> Self::Item) -> Self {
        match self {
            Self::Id(expr) => Self::Id(expr.transform(f)),
            Self::Abs(func, destructor) => Self::Abs(Box::new(func.transform(f)), destructor),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Binding {
    Single(Symbol),
    KeyValuePair(Symbol, Symbol),
}

impl fmt::Display for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Binding::Single(id) => write!(f, "{}", id),
            Binding::KeyValuePair(id1, id2) => write!(f, "({}, {})", id1, id2),
        }
    }
}

#[derive(Debug, Clone)]
pub enum DecomposedExpression {
    Func(InputSimplifiedFunction),
    Product {
        typ: Type,
        decomps: Vec<DecomposedExpression>,
    },
    Collection {
        decomp: Box<DecomposedExpression>,
        iter_type: Type,
        predicate: Expr,

        /// generalization for zip and mapN; multiple destructors
        destructors: Vec<Destructor>,

        /// generalization for zip and mapN; bindings for the iterable [("x", "xs"), ...]
        bindings: Vec<(Binding, Expr)>,
    },
}

impl Transformable for DecomposedExpression {
    type Item = Expr;
    fn transform(self, f: &mut impl FnMut(Self::Item) -> Self::Item) -> Self {
        match self {
            Self::Func(func) => Self::Func(func.transform(f)),
            Self::Product { typ, decomps } => Self::Product {
                typ,
                decomps: decomps.into_iter().map(|d| d.transform(f)).collect(),
            },
            Self::Collection {
                decomp,
                iter_type,
                predicate,
                destructors,
                bindings,
            } => Self::Collection {
                decomp: Box::new(decomp.transform(f)),
                iter_type,
                predicate: predicate.transform(f),
                destructors,
                bindings: bindings
                    .into_iter()
                    .map(|(b, e)| (b, e.transform(f)))
                    .collect(),
            },
        }
    }
}

impl DecomposedExpression {
    pub fn get_isf(&self) -> &InputSimplifiedFunction {
        if let Self::Func(isf) = self {
            isf
        } else {
            panic!("DecomposedExpression is not an ISF: {}", self);
        }
    }

    fn extract_exprs(self) -> Vec<Expr> {
        fn extract_exprs_isf(isf: InputSimplifiedFunction) -> Vec<Expr> {
            match isf {
                InputSimplifiedFunction::Id(expr) => vec![expr],
                InputSimplifiedFunction::Abs(func, destructor) => {
                    let mut exprs = extract_exprs_isf(*func);
                    exprs.push(destructor.call(exprs.clone()));
                    exprs
                }
            }
        }

        match self {
            Self::Func(isf) => extract_exprs_isf(isf),
            Self::Product { decomps, .. } => decomps
                .into_iter()
                .flat_map(|d| d.extract_exprs())
                .collect(),

            Self::Collection { decomp, .. } => decomp.extract_exprs(),
        }
    }
}

impl fmt::Display for DecomposedExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Func(func) => write!(f, "{}", func),
            Self::Product { typ, decomps } => {
                let decomps_str = decomps
                    .iter()
                    .map(|d| format!("{}", d))
                    .collect::<Vec<String>>()
                    .join(", ");

                write!(f, "{}({})", typ, decomps_str)
            }

            Self::Collection {
                decomp,
                iter_type,
                predicate,
                destructors,
                bindings,
            } => {
                let destructors_str = destructors
                    .iter()
                    .map(|d| format!("{}", d))
                    .collect::<Vec<String>>()
                    .join(", ");

                let destructors_str = if !destructors_str.is_empty() {
                    format!("({})", destructors_str)
                } else {
                    "".to_string()
                };

                let bindings_str = bindings
                    .iter()
                    .map(|(id1, id2)| format!("{} in {}", id1, id2))
                    .collect::<Vec<String>>()
                    .join(", ");

                let bindings_str = if !bindings_str.is_empty() {
                    format!("({})", bindings_str)
                } else {
                    "".to_string()
                };

                let args_str = &[
                    decomp.to_string(),
                    predicate.to_string(),
                    destructors_str,
                    bindings_str,
                ]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<String>>()
                .join(", ");

                write!(f, "{}[{}]", iter_type, args_str)
            }
        }
    }
}
