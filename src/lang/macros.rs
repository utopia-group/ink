#[macro_export]
macro_rules! expr_eq {
    ($lhs:expr, $rhs:expr) => {
        binop!(BinOpKinds::Eq, $lhs, $rhs)
    };
}

#[macro_export]
macro_rules! expr_ne {
    ($lhs:expr, $rhs:expr) => {
        unaryop!(
            $crate::lang::UnaryOpKinds::Not,
            binop!(BinOpKinds::Eq, $lhs, $rhs)
        )
    };
}

#[macro_export]
macro_rules! expr_gt {
    ($lhs:expr, $rhs:expr) => {
        binop!(BinOpKinds::Gt, $lhs, $rhs)
    };
}

#[macro_export]
macro_rules! expr_lt {
    ($lhs:expr, $rhs:expr) => {
        binop!(BinOpKinds::Lt, $lhs, $rhs)
    };
}

#[macro_export]
macro_rules! expr_and {
    ($lhs:expr, $rhs:expr) => {
        binop!(BinOpKinds::And, $lhs, $rhs)
    };
}

#[macro_export]
macro_rules! expr_or {
    ($lhs:expr, $rhs:expr) => {
        binop!(BinOpKinds::Or, $lhs, $rhs)
    };
}

#[macro_export]
macro_rules! expr_add {
    ($lhs:expr, $rhs:expr) => {
        binop!(BinOpKinds::Add, $lhs, $rhs)
    };
}

#[macro_export]
macro_rules! ite {
    ($cond:expr, $then:expr, $els:expr) => {
        Expr::Ite {
            cond: Box::new($cond),
            then_expr: Box::new($then),
            else_expr: Box::new($els),
        }
    };
}

#[macro_export]
macro_rules! let_ {
    ($name:expr, $e:expr, $body:expr) => {
        Expr::Let {
            name: $name,
            expr: Box::new($e),
            body: Box::new($body),
        }
    };
}

#[macro_export]
macro_rules! letv {
    ($name:ident, $e:expr, $body:expr) => {
        Expr::Let {
            name: sym!($name),
            expr: Box::new($e),
            body: Box::new($body),
        }
    };
}

#[macro_export]
macro_rules! lambda {
    ($param:ident: $param_type:expr => $body:expr) => {
        Expr::Lambda {
            param: sym!($param),
            param_type: $param_type,
            body: Box::new($body),
        }
    };
}

#[macro_export]
macro_rules! param {
    ($x:ident, $ty:expr) => {
        (egg::Symbol::from(stringify!($x)), $ty)
    };
}

#[macro_export]
macro_rules! sym {
    ($x:ident) => {
        egg::Symbol::from(stringify!($x))
    };
}

#[macro_export]
macro_rules! binop {
    ($op:expr, $lhs:expr, $rhs:expr) => {
        Expr::BinOp($op, Box::new($lhs), Box::new($rhs))
    };
}

#[macro_export]
macro_rules! unaryop {
    ($op:expr, $expr:expr) => {
        Expr::UnaryOp($op, Box::new($expr))
    };
}

#[macro_export]
macro_rules! tuple_access {
    ($tuple:expr, $idx:expr) => {
        Expr::TupleAccess(Box::new($tuple), $idx)
    };
}

#[macro_export]
macro_rules! map_assign {
    ($map:expr, $key:expr, $value:expr) => {
        Expr::MapAssign {
            map: Box::new($map),
            key: Box::new($key),
            value: Box::new($value),
        }
    };
}

#[macro_export]
macro_rules! map_access {
    ($map:expr, $key:expr) => {
        Expr::MapAccess {
            map: Box::new($map),
            key: Box::new($key),
        }
    };
}

#[macro_export]
macro_rules! cons {
    ($lhs:expr, $rhs:expr) => {
        Expr::Cons {
            head: Box::new($lhs),
            tail: Box::new($rhs),
        }
    };
}

#[macro_export]
macro_rules! tuple {
        ($($x:expr),+ $(,)?) => {
            Expr::Tuple(vec![$($x),+])
        };
    }

#[macro_export]
macro_rules! var {
    ($x:ident) => {
        Expr::from(egg::Symbol::from(stringify!($x)))
    };
}

#[macro_export]
macro_rules! tlist {
    ($ty:expr) => {
        Type::List(Box::new($ty))
    };
}

#[macro_export]
macro_rules! tmap {
    ($k:expr, $v:expr) => {
        Type::Map(Box::new($k), Box::new($v))
    };
}

#[macro_export]
macro_rules! ttuple {
        ($($ty:expr),*) => {
            Type::Tuple(vec![$($ty),*])
        };
    }

#[macro_export]
macro_rules! tv {
    ($name:ident) => {
        crate::lang::Type::Var(stringify!($name).into())
    };
}

#[macro_export]
macro_rules! tforall {
    ($v:expr, $t:expr) => {
        PolyType::Forall($v, Box::new($t))
    };
}

#[macro_export]
macro_rules! tfunc {
    ($t:expr) => {
        $t
    };

    ($arg:expr => $($rest:tt)*) => {
        Type::Fn(Box::new($arg), Box::new(tfunc!($($rest)*)))
    };
}

#[macro_export]
macro_rules! tset {
    ($ty:expr) => {
        Type::Set(Box::new($ty))
    };
}

pub use tforall;
pub use tfunc;
pub use tlist;
pub use tmap;
pub use tset;
pub use ttuple;
pub use tv;

pub use binop;
pub use cons;
pub use expr_add;
pub use expr_and;
pub use expr_eq;
pub use expr_gt;
pub use expr_lt;
pub use expr_ne;
pub use expr_or;
pub use ite;
pub use lambda;
pub use let_;
pub use letv;
pub use map_access;
pub use map_assign;
pub use param;
pub use sym;
pub use tuple;
pub use tuple_access;
pub use unaryop;
pub use var;

#[macro_export]
macro_rules! is_func_map {
    ($ex:expr) => {
        *$ex == crate::lang::macros::sym!(map)
            || *$ex == crate::lang::macros::sym!(map_values)
            || *$ex == crate::lang::macros::sym!(map_set)
    };
}

#[macro_export]
macro_rules! is_func_filter {
    ($ex:expr) => {
        *$ex == crate::lang::macros::sym!(filter)
            || *$ex == crate::lang::macros::sym!(filter_map)
            || *$ex == crate::lang::macros::sym!(filter_set)
    };
}

#[macro_export]
macro_rules! is_func_zip {
    ($ex:expr) => {
        *$ex == crate::lang::macros::sym!(zip)
    };
}

#[macro_export]
macro_rules! is_func_foldl {
    ($ex:expr) => {
        *$ex == crate::lang::macros::sym!(foldl)
    };
}

pub use is_func_filter;
pub use is_func_map;
pub use is_func_zip;
