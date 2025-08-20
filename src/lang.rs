use crate::Symbol;
use core::fmt;
use serde::Deserialize;
use std::{collections::HashMap, ops};

mod analysis;
mod interpreter;
pub mod json;
pub mod macros;
mod parser;
mod sygus;
mod type_inference;

pub(crate) use analysis::{simplify, LangAnalyzer};
pub use interpreter::{create_env, eval};
pub use parser::{parse_binop, parser};
pub use sygus::{format_define_fun, FormatSyGuS};
pub use type_inference::{build_subexpr_type_map, infer, BasicSubexprTypeMap, PolyType, TypeEnv};

use macros::*;

type TypeVar = Symbol;
type Id = Symbol;

pub trait TypedEnv {
    fn get_type(&self, expr: &Expr) -> Type;
}

pub struct PlaceholderEnv;
impl TypedEnv for PlaceholderEnv {
    fn get_type(&self, expr: &Expr) -> Type {
        Type::Var(Symbol::from(expr.to_string()))
    }
}

pub trait Transformable: Sized {
    type Item;
    fn transform(self, f: &mut impl FnMut(Self::Item) -> Self::Item) -> Self;
}

pub trait IsFunction {
    type Ret;
    fn call(self, args: Vec<Expr>) -> Self::Ret;
}

pub trait IsCurriedFunction: Sized {
    fn app(self, arg: Expr) -> Self;

    fn call(self, args: Vec<Expr>) -> Self {
        args.into_iter().fold(self, |acc, arg| acc.app(arg))
    }
}

impl IsCurriedFunction for Expr {
    fn app(self, arg: Expr) -> Self {
        Expr::App {
            func: Box::new(self),
            arg: Box::new(arg),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Copy, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinOpKinds {
    Add,
    Sub,
    Mul,
    Div,
    Concat,

    #[serde(rename = ">")]
    Gt,
    #[serde(rename = "<")]
    Lt,
    #[serde(rename = "=")]
    Eq,
    #[serde(rename = "&&")]
    And,
    #[serde(rename = "||")]
    Or,
}

impl fmt::Display for BinOpKinds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use BinOpKinds::*;

        let op = match self {
            Add => "+",
            Sub => "-",
            Mul => "*",
            Div => "/",
            Concat => "++",
            Gt => ">",
            Lt => "<",
            Eq => "=",
            And => "&&",
            Or => "||",
        };

        write!(f, "{}", op)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnaryOpKinds {
    Neg,
    Not,
}

impl fmt::Display for UnaryOpKinds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use UnaryOpKinds::*;

        let op = match self {
            Neg => "--",
            Not => "not",
        };

        write!(f, "{}", op)
    }
}

#[derive(Debug, PartialEq, Clone, Eq, Hash)]
pub enum Expr {
    Num(i32),
    Bool(bool),
    Str(String),
    Var(Id),

    UnaryOp(UnaryOpKinds, Box<Expr>),
    BinOp(BinOpKinds, Box<Expr>, Box<Expr>),

    Ite {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },

    App {
        func: Box<Expr>,
        arg: Box<Expr>,
    },

    Lambda {
        param: Symbol,
        param_type: Type,
        body: Box<Expr>,
    },

    Let {
        name: Symbol,
        expr: Box<Expr>,
        body: Box<Expr>,
    },

    Nil,
    Cons {
        head: Box<Expr>,
        tail: Box<Expr>,
    },
    Tuple(Vec<Expr>),
    TupleAccess(Box<Expr>, usize),

    MapAssign {
        map: Box<Expr>,
        key: Box<Expr>,
        value: Box<Expr>,
    },
    MapAccess {
        map: Box<Expr>,
        key: Box<Expr>,
    },
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Expr::*;

        match self {
            Num(n) => write!(f, "{}", n),
            Str(s) => write!(f, "\"{}\"", s),
            Var(name) => write!(f, "{}", name),
            Bool(b) => write!(f, "{}", b),

            UnaryOp(op, expr) => write!(f, "({}{})", op, expr),
            BinOp(op, lhs, rhs) => write!(f, "({} {} {})", lhs, op, rhs),

            Ite {
                cond,
                then_expr,
                else_expr,
            } => write!(f, "ITE({}, {}, {})", cond, then_expr, else_expr),

            App { .. } => {
                let uncurried = self.clone().uncurry();
                let App { func, arg } = uncurried else {
                    panic!("impossible")
                };
                match func.as_ref() {
                    Var(func) => write!(f, "{}{}", func, arg),
                    _ => write!(f, "({}){}", func, arg),
                }
            }
            Lambda {
                param,
                param_type,
                body,
            } => {
                let mut params = vec![(*param, param_type)];
                let mut f_body = body;

                while let Lambda {
                    param,
                    param_type,
                    body,
                } = f_body.as_ref()
                {
                    params.push((*param, param_type));
                    f_body = body;
                }

                let params_str = params
                    .iter()
                    .map(|(param, ty)| format!("{}: {}", param, ty))
                    .collect::<Vec<std::string::String>>()
                    .join(", ");

                write!(f, "λ({}). {}", params_str, f_body)
            }

            Let { name, expr, body } => write!(f, "let {} = {} in {}", name, expr, body),

            Nil => write!(f, "nil"),
            Cons { head, tail } => write!(f, "({} :: {})", head, tail),
            Tuple(exprs) => write!(
                f,
                "({})",
                exprs
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            TupleAccess(tuple, idx) => write!(f, "{}._{}", tuple, idx),
            MapAssign { map, key, value } => write!(f, "{}[{} <- {}]", map, key, value),
            MapAccess { map, key } => write!(f, "{}[{}]", map, key),
        }
    }
}

impl ops::Add<Expr> for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Self::Output {
        Expr::BinOp(BinOpKinds::Add, Box::new(self), Box::new(rhs))
    }
}

impl ops::Sub<Expr> for Expr {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Self::Output {
        Expr::BinOp(BinOpKinds::Sub, Box::new(self), Box::new(rhs))
    }
}

impl ops::Mul<Expr> for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Self::Output {
        Expr::BinOp(BinOpKinds::Mul, Box::new(self), Box::new(rhs))
    }
}

impl ops::Div<Expr> for Expr {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Self::Output {
        Expr::BinOp(BinOpKinds::Div, Box::new(self), Box::new(rhs))
    }
}

impl ops::Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Self::Output {
        Expr::UnaryOp(UnaryOpKinds::Neg, Box::new(self))
    }
}

impl ops::Not for Expr {
    type Output = Expr;
    fn not(self) -> Self::Output {
        Expr::UnaryOp(UnaryOpKinds::Not, Box::new(self))
    }
}

impl ops::BitAnd<Expr> for Expr {
    type Output = Expr;
    fn bitand(self, rhs: Expr) -> Self::Output {
        Expr::BinOp(BinOpKinds::And, Box::new(self), Box::new(rhs))
    }
}

impl ops::BitOr<Expr> for Expr {
    type Output = Expr;
    fn bitor(self, rhs: Expr) -> Self::Output {
        Expr::BinOp(BinOpKinds::Or, Box::new(self), Box::new(rhs))
    }
}

pub type Env = Vec<(Symbol, Value)>;
pub type ValueFn = fn(Vec<Value>) -> Result<Value, String>;

#[derive(Debug, Clone)]
pub enum Value {
    Num(i32),
    Bool(bool),
    Str(String),
    Nil,
    Cons(Box<Value>, Box<Value>),
    Tuple(Vec<Value>),
    Map(Vec<(Value, Value)>),
    Set(Vec<Value>),
    Closure {
        env: Env,
        param: Symbol,
        body: Expr,
    },
    PrimitiveFn {
        func: ValueFn,
        num_params: usize,
    },
    PrimitiveFnCall {
        func: ValueFn,
        num_params: usize,
        args: Vec<Value>,
    },
}

impl PartialEq for Value {
    fn eq(&self, rhs: &Self) -> bool {
        use Value::*;
        match (self, rhs) {
            (Num(lhs), Num(rhs)) => lhs == rhs,
            (Bool(lhs), Bool(rhs)) => lhs == rhs,
            (Str(lhs), Str(rhs)) => lhs == rhs,
            (Nil, Nil) => true,
            (Cons(lh, lt), Cons(rh, rt)) => lh == rh && lt == rt,
            (Tuple(lhs), Tuple(rhs)) => lhs.iter().zip(rhs.iter()).all(|(l, r)| l == r),
            (Map(lhs), Map(rhs)) => {
                let keys = lhs
                    .iter()
                    .chain(rhs.iter())
                    .map(|(k, _)| k)
                    .collect::<Vec<_>>();

                keys.into_iter().all(|k| {
                    let lv = lhs.iter().find(|(lk, _)| *lk == *k).map(|(_, v)| v);
                    let rv = rhs.iter().find(|(rk, _)| *rk == *k).map(|(_, v)| v);

                    match (lv, rv) {
                        (Some(lv), Some(rv)) => lv == rv,
                        _ => false,
                    }
                })
            }
            (Set(lhs), Set(rhs)) => lhs.len() == rhs.len() && lhs.iter().all(|l| rhs.contains(l)),
            _ => false,
        }
    }
}

impl Eq for Value {}

impl Value {
    /// Get the type of a concrete value.
    pub fn get_type(&self) -> Type {
        use Value::*;
        match self {
            Num(_) => Type::Num,
            Bool(_) => Type::Bool,
            Str(_) => Type::Str,
            Nil => panic!("unknown"),
            Cons(h, _) => tlist!(h.get_type()),
            Tuple(vs) => Type::Tuple(vs.iter().map(Value::get_type).collect()),
            Map(vs) => {
                let (k, v) = vs.iter().next().expect("empty map");
                tmap!(k.get_type(), v.get_type())
            }
            _ => panic!("unknown"),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Value::*;

        match self {
            Num(n) => write!(f, "{}", n),
            Bool(b) => write!(f, "{}", b),
            Str(s) => write!(f, "\"{}\"", s),
            Nil => write!(f, "nil"),
            Cons(head, tail) => write!(f, "({} :: {})", head, tail),
            Tuple(values) => write!(
                f,
                "({})",
                values
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Map(map) => write!(
                f,
                "{{{}}}",
                map.iter()
                    .map(|(k, v)| format!("{}: {}", k, v))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Set(values) => write!(
                f,
                "{{{}}}",
                values
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Closure { param, body, .. } => write!(f, "λ({}). {}", param, body),
            PrimitiveFn { .. } => write!(f, "<primitive fn>"),
            PrimitiveFnCall { .. } => write!(f, "<primitive fn call>"),
        }
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Value {
        v.into_iter()
            .map(Into::into)
            .rev()
            .fold(Value::Nil, |acc, v| Value::Cons(Box::new(v), Box::new(acc)))
    }
}

impl<K: Into<Value>, V: Into<Value>> From<HashMap<K, V>> for Value {
    fn from(value: HashMap<K, V>) -> Self {
        Value::Map(
            value
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect::<Vec<_>>(),
        )
    }
}

impl<T: Into<Expr>> From<Vec<T>> for Expr {
    fn from(v: Vec<T>) -> Expr {
        v.into_iter()
            .map(Into::into)
            .rev()
            .fold(Expr::Nil, |acc: Expr, v| Expr::Cons {
                head: Box::new(v),
                tail: Box::new(acc),
            })
    }
}

impl From<Id> for Expr {
    fn from(v: Id) -> Self {
        Expr::Var(v)
    }
}

impl From<&str> for Expr {
    fn from(v: &str) -> Self {
        if let Ok(n) = v.parse::<i32>() {
            Expr::Num(n)
        } else if v.starts_with('"') && v.ends_with('"') {
            Expr::Str(v[1..v.len() - 1].to_string())
        } else {
            Expr::Var(Id::from(v))
        }
    }
}

impl From<i32> for Expr {
    fn from(v: i32) -> Self {
        Expr::Num(v)
    }
}

impl From<bool> for Expr {
    fn from(v: bool) -> Self {
        Expr::Bool(v)
    }
}

impl From<String> for Expr {
    fn from(v: String) -> Self {
        Expr::Str(v)
    }
}

impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::Num(v)
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Str(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Str(v.to_string())
    }
}

/// Convert a value into an expression; only works for constants.
impl From<Value> for Expr {
    fn from(value: Value) -> Self {
        match value {
            Value::Num(n) => Expr::Num(n),
            Value::Bool(b) => Expr::Bool(b),
            Value::Str(s) => Expr::Str(s),
            Value::Nil => Expr::Nil,
            Value::Cons(head, tail) => Expr::Cons {
                head: Box::new((*head).into()),
                tail: Box::new((*tail).into()),
            },
            Value::Tuple(values) => Expr::Tuple(values.into_iter().map(Into::into).collect()),
            _ => panic!("Invalid conversion"),
        }
    }
}

impl From<Value> for Vec<Value> {
    fn from(v: Value) -> Vec<Value> {
        match &v {
            Value::Tuple(v) => v.clone(),
            Value::Nil => vec![],
            Value::Cons(..) => {
                let mut v = v;
                let mut result = vec![];
                while let Value::Cons(head, tail) = v {
                    result.push(*head);
                    v = *tail;
                }
                result
            }
            _ => panic!("Invalid conversion {}", v),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Eq, Hash)]
pub enum Type {
    Var(TypeVar),
    Num,
    Bool,
    Str,
    List(Box<Type>),
    Tuple(Vec<Type>),
    Map(Box<Type>, Box<Type>),
    Set(Box<Type>),
    Fn(Box<Type>, Box<Type>),
}

impl Type {
    pub fn is_collection(&self) -> bool {
        matches!(self, Type::Map(..) | Type::List(..) | Type::Set(..))
    }

    pub fn is_compound(&self) -> bool {
        matches!(self, Type::Tuple(..))
    }

    pub fn is_base(&self) -> bool {
        matches!(self, Type::Num | Type::Bool | Type::Str)
    }

    pub fn element_type(&self) -> &Type {
        match self {
            Type::List(ty) => ty,
            Type::Map(_, ty_v) => ty_v,
            Type::Set(ty) => ty,
            _ => panic!("cannot get element type of non-collection type: {}", self),
        }
    }

    pub fn with_new_element_type(&self, typ: Type) -> Type {
        match self {
            Type::List(_) => tlist!(typ),
            Type::Map(ty_k, ..) => tmap!(*ty_k.clone(), typ),
            Type::Set(_) => tset!(typ),
            _ => panic!("cannot set element type of non-collection type: {}", self),
        }
    }

    pub fn default_value(&self) -> Value {
        match self {
            Type::Num => Value::Num(0),
            Type::Bool => Value::Bool(false),
            Type::Str => Value::Str("".to_string()),
            Type::List(..) => Value::Nil,
            Type::Tuple(tys) => Value::Tuple(tys.iter().map(Type::default_value).collect()),
            Type::Map(..) => Value::Map(vec![]),
            Type::Set(..) => Value::Set(vec![]),

            _ => panic!("cannot get default value of non-primitive type: {}", self),
        }
    }

    pub fn uncurry_fn(&self) -> (Vec<&Type>, &Type) {
        let Type::Fn(arg, ret) = self else {
            panic!("attempt to uncurry a non-function type: {}", self)
        };

        let mut param_types = vec![arg.as_ref()];
        let mut ret = ret.as_ref();

        while let Type::Fn(arg, ret2) = ret {
            param_types.push(arg.as_ref());
            ret = ret2.as_ref();
        }

        (param_types, ret)
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Type::*;

        match self {
            Var(name) => write!(f, "{}", name),
            Num => write!(f, "num"),
            Bool => write!(f, "bool"),
            Str => write!(f, "str"),
            List(ty) => write!(f, "list[{}]", ty),
            Tuple(tys) => write!(
                f,
                "({})",
                tys.iter()
                    .map(|ty| ty.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Set(ty) => write!(f, "set[{}]", ty),
            Map(k, v) => write!(f, "map[{}, {}]", k, v),
            Fn(arg, ret) => write!(f, "({} -> {})", arg, ret),
        }
    }
}

impl<'a> IntoIterator for &'a Type {
    type Item = &'a Type;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        use Type::*;

        let mut types = vec![];

        let mut stack = vec![self];
        while let Some(ty) = stack.pop() {
            types.push(ty);

            match ty {
                List(ty) | Set(ty) => stack.push(ty),
                Tuple(tys) => stack.extend(tys.iter()),
                Map(k, v) => stack.extend(&[k.as_ref(), v.as_ref()]),
                Fn(arg, ret) => stack.extend(&[arg.as_ref(), ret.as_ref()]),
                _ => (),
            }
        }

        types.into_iter()
    }
}

impl<'a> IntoIterator for &'a Expr {
    type Item = &'a Expr;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    /// Returns an iterator over the expression tree.
    fn into_iter(self) -> Self::IntoIter {
        use Expr::*;

        let mut exprs = vec![];

        // dfs over the expression tree
        let mut stack = vec![self];
        while let Some(expr) = stack.pop() {
            exprs.push(expr);

            match expr {
                UnaryOp(_, expr) => stack.push(expr.as_ref()),
                BinOp(_, lhs, rhs) => stack.extend(&[lhs.as_ref(), rhs.as_ref()]),
                Ite {
                    cond,
                    then_expr,
                    else_expr,
                } => stack.extend(&[cond.as_ref(), then_expr.as_ref(), else_expr.as_ref()]),
                App { func, arg } => stack.extend(&[func.as_ref(), arg.as_ref()]),
                Lambda { body, .. } => stack.push(body.as_ref()),
                Let { expr, body, .. } => stack.extend(&[expr.as_ref(), body.as_ref()]),
                Cons { head, tail } => stack.extend(&[head.as_ref(), tail.as_ref()]),
                Tuple(exprs) => stack.extend(exprs.iter()),
                TupleAccess(tuple, _) => stack.push(tuple.as_ref()),
                MapAssign { map, value, .. } => stack.extend(&[map.as_ref(), value.as_ref()]),
                _ => {}
            }
        }

        exprs.into_iter()
    }
}

impl Expr {
    pub fn ast_size(&self) -> usize {
        self.into_iter().count()
    }

    #[inline]
    pub fn create_top_level(func_acc: Expr, init: Expr) -> Expr {
        let input_type = {
            let env = TypeEnv::default();
            let Type::Fn(_state_type, sub_fn_type) = infer(&env, &func_acc).unwrap() else {
                panic!("invalid accumulator function type when creating top-level function");
            };
            let Type::Fn(input_type, _state_type2) = *sub_fn_type else {
                panic!("invalid accumulator function type when creating top-level function");
            };
            assert!(
                *_state_type == *_state_type2,
                "invalid accumulator function type: {} != {}",
                _state_type,
                _state_type2
            );
            *input_type
        };

        var!(foldl)
            .call(vec![func_acc, init, var!(__input_list)])
            .bind_params(vec![param!(__input_list, tlist!(input_type))])
    }

    /// Uncurry the expression and return the function and its arguments.
    pub fn uncurry_call(&self) -> (&Expr, Vec<&Expr>) {
        use Expr::*;

        let App { func, arg } = self else {
            panic!("attempt to uncurry a non-call expression: {}", self)
        };

        let mut func = func.as_ref();
        let mut args = vec![arg.as_ref()];

        while let App { func: f, arg: a } = func {
            func = f.as_ref();
            args.push(a.as_ref());
        }

        args.reverse();
        (func, args)
    }

    pub fn uncurry_lambda(&self) -> (&Expr, Vec<(Symbol, &Type)>) {
        use Expr::*;

        let Lambda {
            param,
            param_type,
            body,
        } = self
        else {
            panic!("attempt to uncurry a non-lambda expression: {}", self)
        };

        let mut params = vec![*param];
        let mut param_types = vec![param_type];
        let mut body = body.as_ref();

        while let Lambda {
            param,
            param_type,
            body: b,
        } = body
        {
            params.push(*param);
            param_types.push(param_type);
            body = b.as_ref();
        }

        (body, params.into_iter().zip(param_types).collect())
    }

    pub fn uncurry(self) -> Expr {
        use Expr::*;

        match &self {
            App { .. } => {
                let (func, args) = self.uncurry_call();
                Expr::App {
                    func: Box::new(func.clone()),
                    arg: Box::new(Expr::Tuple(
                        args.into_iter().cloned().collect::<Vec<Expr>>(),
                    )),
                }
            }

            Lambda { .. } => {
                let (body, params) = self.uncurry_lambda();
                let (params, param_types): (Vec<_>, Vec<_>) = params.into_iter().unzip();

                let param_types = param_types.into_iter().cloned().collect();

                let input_name = "_input".into();

                body.clone()
                    .bind_lets(
                        params
                            .iter()
                            .enumerate()
                            .map(|(i, p)| (*p, TupleAccess(Box::new(Var(input_name)), i)))
                            .collect(),
                    )
                    .bind_params(vec![(input_name, Type::Tuple(param_types))])
            }

            _ => panic!("Invalid uncurry operation: {}", self),
        }
    }

    /// Bind the given parameters to the expression tree using lambda abstractions.
    pub fn bind_params(self, params: Vec<(Symbol, Type)>) -> Expr {
        params
            .into_iter()
            .rev()
            .fold(self, |acc, (param, param_type)| Expr::Lambda {
                param,
                param_type,
                body: Box::new(acc),
            })
    }

    /// Bind the given variables to the expression tree using let bindings.
    pub fn bind_lets(self, bindings: Vec<(Symbol, Expr)>) -> Expr {
        bindings
            .into_iter()
            .rev()
            .fold(self, |acc, (name, expr)| Expr::Let {
                name,
                expr: Box::new(expr),
                body: Box::new(acc),
            })
    }

    pub fn inline_let_bindings(self) -> Expr {
        fn substitute_var(expr: Expr, var_name: Symbol, replacement: &Expr) -> Expr {
            expr.transform(&mut |expr| match expr {
                Expr::Var(name) if name == var_name => replacement.clone(),
                _ => expr,
            })
        }

        self.transform(&mut |expr| match expr {
            Expr::Let { name, expr, body } => {
                let inlined_expr = expr.inline_let_bindings();
                let inlined_body = body.inline_let_bindings();
                substitute_var(inlined_body, name, &inlined_expr)
            }
            _ => expr,
        })
    }
}

impl Transformable for Expr {
    type Item = Expr;

    /// Transform the expression tree using the given function.
    fn transform(self, f: &mut impl FnMut(Self::Item) -> Self::Item) -> Self::Item {
        use Expr::*;

        match self {
            UnaryOp(op, expr) => {
                let expr = expr.transform(f);
                f(UnaryOp(op, Box::new(expr)))
            }
            BinOp(op, lhs, rhs) => {
                let lhs = lhs.transform(f);
                let rhs = rhs.transform(f);
                f(BinOp(op, Box::new(lhs), Box::new(rhs)))
            }
            Ite {
                cond,
                then_expr,
                else_expr,
            } => {
                let cond = cond.transform(f);
                let then_expr = then_expr.transform(f);
                let else_expr = else_expr.transform(f);
                f(Ite {
                    cond: Box::new(cond),
                    then_expr: Box::new(then_expr),
                    else_expr: Box::new(else_expr),
                })
            }
            App { func, arg } => {
                let func = func.transform(f);
                let arg = arg.transform(f);
                f(App {
                    func: Box::new(func),
                    arg: Box::new(arg),
                })
            }
            Lambda {
                param,
                param_type,
                body,
            } => {
                let body = body.transform(f);
                f(Lambda {
                    param,
                    param_type: param_type.clone(),
                    body: Box::new(body),
                })
            }
            Let { name, expr, body } => {
                let expr = expr.transform(f);
                let body = body.transform(f);
                f(Let {
                    name,
                    expr: Box::new(expr),
                    body: Box::new(body),
                })
            }
            Cons { head, tail } => {
                let head = head.transform(f);
                let tail = tail.transform(f);
                f(Cons {
                    head: Box::new(head),
                    tail: Box::new(tail),
                })
            }
            Tuple(exprs) => {
                let exprs = exprs.into_iter().map(|e| e.transform(f)).collect();
                f(Tuple(exprs))
            }
            TupleAccess(tuple, idx) => {
                let tuple = tuple.transform(f);
                f(TupleAccess(Box::new(tuple), idx))
            }
            MapAssign { map, key, value } => {
                let map = map.transform(f);
                let value = value.transform(f);
                f(MapAssign {
                    map: Box::new(map),
                    key,
                    value: Box::new(value),
                })
            }

            _ => f(self),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::macros::*;
    use super::*;
    use std::vec;

    #[test]
    fn test_bind_params() {
        use Expr::*;
        let body = BinOp(
            BinOpKinds::Mul,
            Box::new(Expr::from(sym!(x))),
            Box::new(Expr::from(sym!(y))),
        );

        let func = body.bind_params(vec![param!(x, Type::Num), param!(y, Type::Num)]);

        assert_eq!(func.to_string(), "λ(x: num, y: num). (x * y)")
    }

    #[test]
    fn test_call() {
        use Expr::*;

        let func = Expr::from(sym!(f));
        let call_expr = func.call(vec![
            Expr::from(sym!(a)),
            Expr::from(sym!(b)),
            Expr::from(sym!(c)),
        ]);

        assert_eq!(
            call_expr,
            App {
                func: Box::new(App {
                    func: Box::new(App {
                        func: Box::new(Expr::from(sym!(f))),
                        arg: Box::new(Expr::from(sym!(a))),
                    }),
                    arg: Box::new(Expr::from(sym!(b))),
                }),
                arg: Box::new(Expr::from(sym!(c))),
            }
        );

        assert_eq!(call_expr.to_string(), "f(a, b, c)")
    }

    #[test]
    fn test_uncurry_app() {
        use Expr::*;

        let func = Expr::from(sym!(f));
        let arg1 = Expr::from(sym!(x));
        let arg2 = Expr::from(sym!(y));

        let expr = func.clone().call(vec![arg1.clone(), arg2.clone()]);

        let expected = App {
            arg: Box::new(Tuple(vec![arg1, arg2])),
            func: Box::new(func),
        };

        assert_eq!(expr.uncurry(), expected)
    }

    #[test]
    fn test_uncurry_lambda() {
        use Expr::*;
        let body = BinOp(
            BinOpKinds::Mul,
            Box::new(Expr::from(sym!(x))),
            Box::new(Expr::from(sym!(y))),
        );

        let expr = body
            .clone()
            .bind_params(vec![param!(x, Type::Num), param!(y, Type::Num)]);

        let expected = Lambda {
            param: sym!(x),
            param_type: Type::Num,
            body: Box::new(Lambda {
                param: sym!(y),
                param_type: Type::Num,
                body: Box::new(body.clone()),
            }),
        };

        assert_eq!(expr, expected);

        let uncurried = expected.uncurry();

        let expected_body = body.bind_lets(vec![
            (sym!(x), TupleAccess(Box::new(Expr::from(sym!(_input))), 0)),
            (sym!(y), TupleAccess(Box::new(Expr::from(sym!(_input))), 1)),
        ]);

        let expected = Lambda {
            param: sym!(_input),
            param_type: Type::Tuple(vec![Type::Num, Type::Num]),
            body: Box::new(expected_body),
        };

        assert_eq!(uncurried, expected);
    }

    #[test]
    fn test_transform() {
        use Expr::*;

        let elem_1 = Tuple(vec![binop!(
            BinOpKinds::Add,
            tuple_access!(tuple_access!(Var(sym!(s)), 0), 0),
            Var(sym!(x))
        )]);

        let elem_2 = Tuple(vec![binop!(
            BinOpKinds::Add,
            tuple_access!(tuple_access!(Var(sym!(s)), 0), 1),
            binop!(BinOpKinds::Mul, Num(2), Var(sym!(x)))
        )]);

        let elem_3 = binop!(BinOpKinds::Add, tuple_access!(Var(sym!(s)), 1), Num(1));

        let func = Tuple(vec![Tuple(vec![elem_1, elem_2]), elem_3]).bind_params(vec![
            param!(
                s,
                Type::Tuple(vec![Type::Tuple(vec![Type::Num, Type::Num]), Type::Num])
            ),
            param!(x, Type::Num),
        ]);

        // transform s._0 into s0
        let transformed = func.transform(&mut |expr| match expr {
            TupleAccess(tuple, 0) if *tuple == Var(sym!(s)) => Var(sym!(s0)),
            _ => expr,
        });
        println!("{}", transformed);
    }
}
