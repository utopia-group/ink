use crate::Symbol;
use tracing::{event, Level};

use crate::is_func_filter;
use crate::lang::{macros::*, Expr, Id, Type, TypeVar};
use crate::lang::{BinOpKinds, IsCurriedFunction, UnaryOpKinds};

use std::collections::{HashMap, HashSet};

use super::TypedEnv;

const EMPTY_LIST_TVAR: &str = "@@[]";
const CONS_TVAR: &str = "@@cons";
const ITE_TVAR: &str = "@@ite";

#[derive(Debug, PartialEq, Eq, Clone)]
struct Substitution(HashMap<TypeVar, Type>);

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TypeEnv(Vec<(Id, PolyType)>);

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum PolyType {
    Mono(Type),
    Forall(TypeVar, Box<PolyType>),
}

trait HasFreeTypeVars {
    fn type_vars(&self) -> Vec<TypeVar>;
}

trait Substitutable {
    fn apply_substitution(&self, substitution: &Substitution) -> Self;
}

impl Default for Substitution {
    fn default() -> Self {
        Substitution(HashMap::new())
    }
}

impl Substitution {
    fn get(&self, var: &TypeVar) -> Option<&Type> {
        self.0.get(var)
    }

    fn extend(&mut self, var: TypeVar, t: Type) {
        self.0.insert(var, t.clone());

        let subst = Substitution(HashMap::from([(var, t)]));
        for v in self.0.values_mut() {
            *v = v.apply_substitution(&subst);
        }
    }

    fn remove(&mut self, var: &TypeVar) {
        self.0.remove(var);
    }
}

impl HasFreeTypeVars for Type {
    fn type_vars(&self) -> Vec<TypeVar> {
        use Type::*;
        match self {
            Var(v) => vec![v.clone()],
            Num => vec![],
            Bool => vec![],
            Str => vec![],
            List(t) => t.type_vars(),
            Tuple(ts) => ts
                .iter()
                .flat_map(|t| t.type_vars())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect(),
            Map(k, v) => k
                .type_vars()
                .into_iter()
                .chain(v.type_vars())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect(),
            Fn(arg, ret) => arg
                .type_vars()
                .into_iter()
                .chain(ret.type_vars())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect(),
            Set(t) => t.type_vars(),
        }
    }
}

impl HasFreeTypeVars for PolyType {
    fn type_vars(&self) -> Vec<TypeVar> {
        use PolyType::*;
        match self {
            Mono(t) => t.type_vars(),
            Forall(v, t) => t.type_vars().into_iter().filter(|x| x != v).collect(),
        }
    }
}

impl HasFreeTypeVars for TypeEnv {
    fn type_vars(&self) -> Vec<TypeVar> {
        self.0
            .iter()
            .flat_map(|(_, t)| t.type_vars())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }
}

impl Substitutable for Type {
    fn apply_substitution(&self, substitution: &Substitution) -> Self {
        match self {
            Type::Var(v) => substitution
                .get(v)
                .map(Clone::clone)
                .unwrap_or(Type::Var(*v)),
            Type::List(t) => Type::List(Box::new(t.apply_substitution(substitution))),
            Type::Set(t) => Type::Set(Box::new(t.apply_substitution(substitution))),
            Type::Tuple(ts) => Type::Tuple(
                ts.iter()
                    .map(|t| t.apply_substitution(substitution))
                    .collect(),
            ),
            Type::Map(k, v) => Type::Map(
                Box::new(k.apply_substitution(substitution)),
                Box::new(v.apply_substitution(substitution)),
            ),
            Type::Fn(arg, ret) => Type::Fn(
                Box::new(arg.apply_substitution(substitution)),
                Box::new(ret.apply_substitution(substitution)),
            ),
            _ => self.clone(),
        }
    }
}

impl Substitutable for PolyType {
    fn apply_substitution(&self, substitution: &Substitution) -> Self {
        match self {
            PolyType::Mono(t) => PolyType::Mono(t.apply_substitution(substitution)),
            PolyType::Forall(v, t) => {
                let mut s = substitution.clone();
                s.remove(v);
                PolyType::Forall(*v, Box::new(t.apply_substitution(&s)))
            }
        }
    }
}

impl Substitutable for TypeEnv {
    fn apply_substitution(&self, substitution: &Substitution) -> Self {
        TypeEnv(
            self.0
                .iter()
                .map(|(id, t)| (*id, t.apply_substitution(substitution)))
                .collect(),
        )
    }
}

impl Substitutable for Substitution {
    fn apply_substitution(&self, substitution: &Substitution) -> Self {
        Substitution(
            self.0
                .iter()
                .map(|(k, v)| (k.clone(), v.apply_substitution(substitution)))
                .collect(),
        )
    }
}

impl TypeEnv {
    pub fn new(other: Vec<(Id, PolyType)>) -> Self {
        let mut env = Self::default();
        env.0.extend(other);
        env
    }

    pub fn empty() -> Self {
        Self(vec![])
    }

    pub fn lookup(&self, id: &Id) -> Option<&PolyType> {
        self.0.iter().rev().find(|(k, _)| k == id).map(|(_, v)| v)
    }

    pub fn extend(&mut self, id: Id, poly_type: PolyType) {
        self.0.push((id, poly_type));
    }
}

#[derive(Debug, Clone)]
struct TypeInferenceState {
    substitution: Substitution,
    num_fresh_vars: usize,
}

impl Default for TypeInferenceState {
    fn default() -> Self {
        TypeInferenceState {
            substitution: Substitution::default(),
            num_fresh_vars: 0,
        }
    }
}

impl TypeInferenceState {
    fn unify_type_variable(mut self, v: TypeVar, t: &Type) -> Result<Self, String> {
        match t {
            Type::Var(v2) if v == *v2 => Ok(self),
            _ => {
                if t.type_vars().contains(&v) {
                    Err(format!("Type variable {} occurs in type {}", v, t))
                } else {
                    self.substitution.extend(v, t.clone());
                    Ok(self)
                }
            }
        }
    }

    fn unify(self, t1: &Type, t2: &Type) -> Result<Self, String> {
        match (t1, t2) {
            (Type::Var(t1), t2) => self.unify_type_variable(*t1, t2),
            (t1, Type::Var(t2)) => self.unify_type_variable(*t2, t1),
            (Type::Num, Type::Num) => Ok(self),
            (Type::Bool, Type::Bool) => Ok(self),
            (Type::Str, Type::Str) => Ok(self),
            (Type::List(t1), Type::List(t2)) => self.unify(t1, t2),
            (Type::Set(t1), Type::Set(t2)) => self.unify(t1, t2),
            (Type::Fn(arg1, ret1), Type::Fn(arg2, ret2)) => {
                let s1 = TypeInferenceState::default()
                    .unify(arg1, arg2)?
                    .substitution;
                let s2 = TypeInferenceState::default()
                    .unify(&ret1.apply_substitution(&s1), &ret2.apply_substitution(&s1))?
                    .substitution;
                let s3 = self.substitution;

                Ok(TypeInferenceState {
                    substitution: Substitution(
                        None.into_iter()
                            .chain(s1.clone().apply_substitution(&s2).0)
                            .chain(s2.apply_substitution(&s1).0)
                            .chain(s3.0)
                            .collect::<HashMap<_, _>>(),
                    ),
                    num_fresh_vars: self.num_fresh_vars,
                })
            }

            (Type::Map(k1, v1), Type::Map(k2, v2)) => {
                let s1 = TypeInferenceState::default().unify(k1, k2)?.substitution;
                let s2 = TypeInferenceState::default()
                    .unify(&v1.apply_substitution(&s1), &v2.apply_substitution(&s1))?
                    .substitution;
                let s3 = self.substitution;

                Ok(TypeInferenceState {
                    substitution: Substitution(
                        None.into_iter()
                            .chain(s1.clone().apply_substitution(&s2).0)
                            .chain(s2.apply_substitution(&s1).0)
                            .chain(s3.0)
                            .collect::<HashMap<_, _>>(),
                    ),
                    num_fresh_vars: self.num_fresh_vars,
                })
            }

            (Type::Tuple(ts1), Type::Tuple(ts2)) => {
                if ts1.len() != ts2.len() {
                    return Err(format!(
                        "Cannot unify tuples of different lengths: {} and {}",
                        ts1.len(),
                        ts2.len()
                    ));
                }

                let mut state = self;
                for (t1, t2) in ts1.iter().zip(ts2.iter()) {
                    state = state.unify(t1, t2)?;
                }

                Ok(state)
            }

            (t1, t2) => Err(format!("Cannot unify types {} and {}", t1, t2)),
        }
    }

    fn infer(self: Self, env: TypeEnv, expr: &Expr) -> Result<(Self, Type), String> {
        match expr {
            Expr::Num(_) => Ok((self, Type::Num)),
            Expr::Bool(_) => Ok((self, Type::Bool)),
            Expr::Str(_) => Ok((self, Type::Str)),
            Expr::Var(v) => {
                let tv = env
                    .lookup(&v)
                    .map(Clone::clone)
                    // .expect(format!("Unknown variable {}", v).as_str());
                    .unwrap_or(tforall!(sym!(a), PolyType::Mono(tv!(a))));
                let (num_fresh_vars, t) = instantiate(self.num_fresh_vars, tv.clone());
                Ok((
                    TypeInferenceState {
                        num_fresh_vars,
                        ..self
                    },
                    t,
                ))
            }

            Expr::UnaryOp(op, expr) => self.infer(
                env,
                &Expr::from(op.to_string().as_str()).call(vec![*expr.clone()]),
            ),

            Expr::BinOp(op, expr1, expr2) => self.infer(
                env,
                &Expr::from(op.to_string().as_str()).call(vec![*expr1.clone(), *expr2.clone()]),
            ),

            Expr::Ite {
                cond,
                then_expr,
                else_expr,
            } => self.infer(
                env,
                &Expr::from(ITE_TVAR).call(vec![
                    *cond.clone(),
                    *then_expr.clone(),
                    *else_expr.clone(),
                ]),
            ),

            Expr::Nil => self.infer(env, &Expr::from(EMPTY_LIST_TVAR)),
            Expr::Cons { head, tail } => self.infer(
                env,
                &Expr::from(CONS_TVAR).call(vec![*head.clone(), *tail.clone()]),
            ),
            Expr::Tuple(exprs) => {
                let mut types = vec![];
                let mut state = self;

                for expr in exprs {
                    let (s, typ) = state.infer(env.clone(), expr)?;
                    types.push(typ);
                    state = s;
                }

                Ok((state, Type::Tuple(types)))
            }
            Expr::TupleAccess(tup, idx) => {
                let (s, t) = self.infer(env.clone(), tup)?;
                match t {
                    Type::Tuple(ts) => Ok((s, ts[*idx].clone())),
                    _ => Err(format!("Expected {} to be a tuple, found {}", tup, t)),
                    // _ => panic!("Expected {} to be a tuple, found {}", tup, t),
                }
            }
            Expr::MapAssign { map, value, .. } => {
                let (s, t) = self.infer(env.clone(), map)?;
                let Type::Map(k, v) = t else {
                    return Err(format!("Expected a map type, found {}", t));
                };

                let (s, vt) = s.infer(env.clone(), value)?;
                let s = s.unify(&v, &vt)?;

                Ok((s, Type::Map(Box::new(*k.clone()), Box::new(vt))))
            }
            Expr::MapAccess { map, key } => {
                let (s, t) = self.infer(env.clone(), map)?;
                let Type::Map(k, v) = t else {
                    return Err(format!("Expected a map type, found {}", t));
                };

                let (s, kt) = s.infer(env.clone(), key)?;
                let s = s.unify(&k, &kt)?;

                Ok((s, *v.clone()))
            }

            Expr::Lambda {
                param,
                param_type,
                body,
            } => {
                let mut env = env.clone();
                env.extend(param.clone(), PolyType::Mono(param_type.clone()));

                let updated_state = TypeInferenceState {
                    num_fresh_vars: self.num_fresh_vars + 1,
                    ..self
                };
                let (s, t) = updated_state.infer(env, body)?;
                Ok((s, Type::Fn(Box::new(param_type.clone()), Box::new(t))))
            }

            Expr::App { func, arg } => {
                let (st1, t1) = self.clone().infer(env.clone(), func)?;

                let updated_state = TypeInferenceState {
                    num_fresh_vars: st1.num_fresh_vars,
                    ..self
                };
                let (st2, t2) =
                    updated_state.infer(env.apply_substitution(&st1.substitution), arg)?;

                let type_x = create_fresh_type_variable(st2.num_fresh_vars);
                let sub3 = st2
                    .clone()
                    .unify(&t1, &tfunc!(t2.clone() => type_x.clone()))
                    .map_err(|e| {
                        event!(
                            Level::ERROR,
                            e = e,
                            t1 = t1.to_string(),
                            t2 = t2.to_string(),
                            func = func.to_string(),
                            arg = arg.to_string(),
                            "unification failed"
                        );
                        e
                    })?
                    .substitution;

                let subst = None
                    .into_iter()
                    .chain(sub3.clone().0)
                    .chain(st2.substitution.0)
                    .chain(st1.substitution.0)
                    .collect();

                Ok((
                    TypeInferenceState {
                        substitution: Substitution(subst),
                        num_fresh_vars: st2.num_fresh_vars + 1,
                    },
                    type_x.apply_substitution(&sub3),
                ))
            }

            Expr::Let { name, expr, body } => {
                let type_x = create_fresh_type_variable(self.num_fresh_vars);
                let mut env0 = env.clone();
                env0.extend(name.clone(), PolyType::Mono(type_x.clone()));

                let (
                    TypeInferenceState {
                        substitution: sub1,
                        num_fresh_vars: n1,
                    },
                    t1,
                ) = TypeInferenceState {
                    num_fresh_vars: self.num_fresh_vars + 1,
                    ..self
                }
                .infer(env0.clone(), expr)?;

                let type_x = type_x.apply_substitution(&sub1);
                let sub_x = TypeInferenceState::default()
                    .unify(&t1, &type_x)?
                    .substitution;

                let subst = sub_x.apply_substitution(&sub1);
                let mut env1 = env0.apply_substitution(&subst);
                let poly = generalize(&env1, t1);

                env1.extend(name.clone(), poly);

                TypeInferenceState {
                    num_fresh_vars: n1,
                    substitution: subst,
                }
                .infer(env1, body)
            }
        }
    }
}

/// Create a fresh type variable with a given index
fn create_fresh_type_variable(n: usize) -> Type {
    Type::Var(TypeVar::from(format!("a{}", n)))
}

/// Generalize type variables inside a type
fn generalize(env: &TypeEnv, typ: Type) -> PolyType {
    typ.clone()
        .type_vars()
        .into_iter()
        .filter(|v| !env.type_vars().contains(v))
        .collect::<HashSet<_>>()
        .into_iter()
        .fold(PolyType::Mono(typ), |acc, v| {
            PolyType::Forall(v, Box::new(acc))
        })
}

/// Instantiate a polymorphic type into a mono-type with fresh type variables
fn instantiate(num_fresh_vars: usize, typ: PolyType) -> (usize, Type) {
    match typ {
        PolyType::Mono(t) => (num_fresh_vars, t),
        PolyType::Forall(v, t) => {
            let subst = Substitution(HashMap::from([(
                v,
                create_fresh_type_variable(num_fresh_vars),
            )]));

            let t = t.apply_substitution(&subst);
            instantiate(num_fresh_vars + 1, t)
        }
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        use BinOpKinds::*;
        use PolyType::*;
        use UnaryOpKinds::*;

        TypeEnv(vec![
            (
                Add.to_string().into(),
                Mono(tfunc!(Type::Num => Type::Num => Type::Num)),
            ),
            (
                Sub.to_string().into(),
                Mono(tfunc!(Type::Num => Type::Num => Type::Num)),
            ),
            (
                Mul.to_string().into(),
                Mono(tfunc!(Type::Num => Type::Num => Type::Num)),
            ),
            (
                Div.to_string().into(),
                Mono(tfunc!(Type::Num => Type::Num => Type::Num)),
            ),
            (
                Concat.to_string().into(),
                tforall!(
                    sym!(a),
                    Mono(tfunc!(tlist!(tv!(a)) => tlist!(tv!(a)) => tlist!(tv!(a))))
                ),
            ),
            (
                Gt.to_string().into(),
                Mono(tfunc!(Type::Num => Type::Num => Type::Bool)),
            ),
            (
                Lt.to_string().into(),
                Mono(tfunc!(Type::Num => Type::Num => Type::Bool)),
            ),
            (
                Eq.to_string().into(),
                tforall!(sym!(a), Mono(tfunc!(tv!(a) => tv!(a) => Type::Bool))),
            ),
            (
                And.to_string().into(),
                Mono(tfunc!(Type::Bool => Type::Bool => Type::Bool)),
            ),
            (
                Or.to_string().into(),
                Mono(tfunc!(Type::Bool => Type::Bool => Type::Bool)),
            ),
            (Neg.to_string().into(), Mono(tfunc!(Type::Num => Type::Num))),
            (
                Not.to_string().into(),
                Mono(tfunc!(Type::Bool => Type::Bool)),
            ),
            (
                ITE_TVAR.into(),
                Forall(
                    sym!(a),
                    Box::new(Mono(tfunc!(Type::Bool => tv!(a) => tv!(a) => tv!(a)))),
                ),
            ),
            (
                CONS_TVAR.into(),
                Forall(
                    sym!(a),
                    Box::new(Mono(tfunc!(tv!(a) => tlist!(tv!(a)) => tlist!(tv!(a))))),
                ),
            ),
            (
                EMPTY_LIST_TVAR.into(),
                Forall(sym!(a), Box::new(Mono(tlist!(tv!(a))))),
            ),
            // primitives
            (sym!(true), Mono(Type::Bool)),
            (sym!(false), Mono(Type::Bool)),
            (
                sym!(foldl),
                tforall!(
                    sym!(a),
                    tforall!(
                        sym!(b),
                        Mono(
                            tfunc!(tfunc!(tv!(b) => tv!(a) => tv!(b)) => tv!(b) => tlist!(tv!(a)) => tv!(b))
                        )
                    )
                ),
            ),
            // other built-in functions
            (sym!(abs), Mono(tfunc!(Type::Num => Type::Num))),
            (sym!(max), Mono(tfunc!(Type::Num => Type::Num => Type::Num))),
            (sym!(min), Mono(tfunc!(Type::Num => Type::Num => Type::Num))),
            (sym!(_mn), Mono(Type::Num)),
            (sym!(_mx), Mono(Type::Num)),
            (
                sym!(zip),
                tforall!(
                    sym!(a),
                    tforall!(
                        sym!(b),
                        Mono(
                            tfunc!(tlist!(tv!(a)) => tlist!(tv!(b)) => tlist!(ttuple!(tv!(a), tv!(b))))
                        )
                    )
                ),
            ),
            (
                sym!(map),
                tforall!(
                    sym!(a),
                    tforall!(
                        sym!(b),
                        Mono(tfunc!(tfunc!(tv!(a) => tv!(b)) => tlist!(tv!(a)) => tlist!(tv!(b))))
                    )
                ),
            ),
            (
                sym!(filter),
                tforall!(
                    sym!(a),
                    Mono(tfunc!(tfunc!(tv!(a) => Type::Bool) => tlist!(tv!(a)) => tlist!(tv!(a))))
                ),
            ),
            (
                sym!(length),
                tforall!(sym!(a), Mono(tfunc!(tv!(a) => Type::Num))),
            ),
            // set primitives
            (
                sym!(union),
                tforall!(
                    sym!(a),
                    Mono(tfunc!(tset!(tv!(a)) => tset!(tv!(a)) => tset!(tv!(a))))
                ),
            ),
            (
                sym!(intersection),
                tforall!(
                    sym!(a),
                    Mono(tfunc!(tset!(tv!(a)) => tset!(tv!(a)) => tset!(tv!(a))))
                ),
            ),
            (
                sym!(map_set),
                tforall!(
                    sym!(a),
                    tforall!(
                        sym!(b),
                        Mono(tfunc!(tfunc!(tv!(a) => tv!(b)) => tset!(tv!(a)) => tset!(tv!(b))))
                    )
                ),
            ),
            (sym!(empty_set), tforall!(sym!(a), Mono(tset!(tv!(a))))),
            (
                sym!(filter_set),
                tforall!(
                    sym!(a),
                    Mono(tfunc!(tfunc!(tv!(a) => Type::Bool) => tset!(tv!(a)) => tset!(tv!(a))))
                ),
            ),
            (
                sym!(set_add),
                tforall!(
                    sym!(a),
                    Mono(tfunc!(tv!(a) => tset!(tv!(a)) => tset!(tv!(a))))
                ),
            ),
            (
                sym!(is_set_empty),
                tforall!(sym!(a), Mono(tfunc!(tset!(tv!(a)) => Type::Bool))),
            ),
            // map primitives
            (
                sym!(empty_map),
                tforall!(sym!(k), tforall!(sym!(v), Mono(tmap!(tv!(k), tv!(v))))),
            ),
            (
                sym!(contains_key),
                tforall!(
                    sym!(k),
                    tforall!(
                        sym!(v),
                        Mono(tfunc!(tmap!(tv!(k), tv!(v)) => tv!(k) => Type::Bool))
                    )
                ),
            ),
            (
                sym!(map_values),
                tforall!(
                    sym!(k),
                    tforall!(
                        sym!(v),
                        Mono(
                            tfunc!(tfunc!(tv!(k) => ttuple!(Type::Bool, tv!(v)) => ttuple!(Type::Bool, tv!(v))) => tmap!(tv!(k), tv!(v)) => tmap!(tv!(k), tv!(v)))
                        )
                    )
                ),
            ),
            (
                sym!(filter_values),
                tforall!(
                    sym!(k),
                    tforall!(
                        sym!(v),
                        Mono(
                            tfunc!(tfunc!(tv!(v) => Type::Bool) => tmap!(tv!(k), tv!(v)) => tmap!(tv!(k), tv!(v)))
                        )
                    )
                ),
            ),
            (
                sym!(concat_map),
                tforall!(
                    sym!(k),
                    tforall!(
                        sym!(v),
                        Mono(
                            tfunc!(tmap!(tv!(k), tv!(v)) => tmap!(tv!(k), tv!(v)) => tmap!(tv!(k), tv!(v)))
                        )
                    )
                ),
            ),
        ])
    }
}

pub fn infer(env: &TypeEnv, expr: &Expr) -> Result<Type, String> {
    let (_, typ) = TypeInferenceState::default().infer(env.clone(), expr)?;
    Ok(typ)
}

#[derive(Debug, Clone)]
pub struct BasicSubexprTypeMap {
    env: TypeEnv,
    map: HashMap<Expr, Type>,
}

impl TypedEnv for BasicSubexprTypeMap {
    fn get_type(&self, expr: &Expr) -> Type {
        self.map
            .get(expr)
            .cloned()
            .ok_or(format!("type not found for {}", expr))
            .or_else(|_e| {
                let result = infer(&self.env, expr);
                event!(
                    Level::INFO,
                    expr = expr.to_string(),
                    result = ?result,
                    "type not found; try inferring"
                );
                result
            })
            .unwrap_or(Type::Var(TypeVar::from("cannot_find_type")))
            .clone()
    }
}

impl BasicSubexprTypeMap {
    pub fn new(acc: &Expr, mut env: TypeEnv) -> Self {
        // TODO: remove this hack
        // Add the types of the lambda parameters to the environment
        // this is useful in inferring unknown types in the collection map expressions
        acc.into_iter().for_each(|e| {
            if let Expr::Lambda {
                param, param_type, ..
            } = e
            {
                env.extend(*param, PolyType::Mono(param_type.clone()));
            }
        });

        let typ = infer(&mut env, acc).unwrap();
        let mut map = HashMap::new();
        build_subexpr_type_map(acc, typ, &mut map);

        let map = map.into_iter().map(|(k, v)| (k.clone(), v)).collect();
        BasicSubexprTypeMap { env, map }
    }

    pub fn add_var(&mut self, var: Symbol, typ: PolyType) {
        self.env.extend(var.into(), typ);
    }

    pub fn add_expr(&mut self, expr: &Expr, typ: &Type) {
        if self.map.contains_key(expr) {
            let t = self.map.get(expr).unwrap();
            if *t != *typ {
                panic!("type mismatch for {}: expected {}, found {}", expr, t, typ);
            }
        } else {
            self.map.insert(expr.clone(), typ.clone());

            let mut map = Default::default();
            build_subexpr_type_map(expr, typ.clone(), &mut map);
            self.map
                .extend(map.into_iter().map(|(k, v)| (k.clone(), v)));
        }
    }
}

/// Given an expression and its type, build a map of subexpressions to their types.
/// We need this map to infer the type of subexpressions during vertical decomposition
/// This is a very hacky implementation...
pub fn build_subexpr_type_map<'a>(expr: &'a Expr, typ: Type, map: &mut HashMap<&'a Expr, Type>) {
    map.insert(expr, typ.clone());

    match (expr, &typ) {
        (Expr::Tuple(els), Type::Tuple(tys)) => {
            for (el, ty) in els.iter().zip(tys.iter()) {
                build_subexpr_type_map(el, ty.clone(), map);
            }
        }

        (Expr::Lambda { body, .. }, Type::Fn(_, ret)) => {
            build_subexpr_type_map(body, *ret.clone(), map);
        }

        (Expr::App { .. }, ty) if ty.is_collection() => {
            let element_type = ty.element_type();
            let (func, args) = expr.uncurry_call();
            match func {
                Expr::Var(s) if is_func_filter!(s) => {
                    let &[_, arg] = args.as_slice() else {
                        panic!("uncurry failed");
                    };

                    build_subexpr_type_map(arg, typ, map);
                }

                Expr::Var(s) if is_func_map!(s) => {
                    let &[func, arg] = args.as_slice() else {
                        panic!("uncurry failed");
                    };

                    let Expr::Lambda { param_type, .. } = func else {
                        panic!("failed to find mapper input type");
                    };

                    build_subexpr_type_map(
                        func,
                        tfunc!(param_type.clone() => element_type.clone()),
                        map,
                    );
                    build_subexpr_type_map(arg, typ.with_new_element_type(param_type.clone()), map);
                }

                _ => (),
            }
        }

        _ => (),
    }
}

#[cfg(test)]
mod tests {

    use crate::lang::macros::*;

    use super::*;

    #[test]
    fn test_base() {
        let env = TypeEnv::default();

        let expr = Expr::Num(123);
        assert!(infer(&env, &expr).unwrap() == Type::Num);

        let expr = binop!(BinOpKinds::Add, 1.into(), 2.into());
        assert!(infer(&env, &expr).unwrap() == Type::Num);

        let expr = cons!(1.into(), Expr::Nil);
        println!("{:?}", infer(&env, &expr).unwrap());

        let expr = Expr::from(123)
            .bind_params(vec![("x".into(), Type::Num)])
            .bind_params(vec![("y".into(), Type::Num)]);
        println!("{:?}", infer(&env, &expr).unwrap());
    }

    #[test]
    fn test_unify() {
        let state = TypeInferenceState::default();
        let t1 = tv!(a);
        let t2 = Type::Num;

        let state = state.unify(&t1, &t2).unwrap();
        assert!(state.substitution.0.get(&TypeVar::from("a")).unwrap() == &Type::Num);
    }

    #[test]
    fn test_unify_list() {
        let state = TypeInferenceState::default();
        let t1 = tlist!(ttuple!(tv!(a), tv!(b), Type::Num));
        let t2 = tlist!(ttuple!(Type::Num, tv!(b), tv!(c)));

        let state = state.unify(&t1, &t2).unwrap();
        assert!(state.substitution.0.get(&TypeVar::from("a")).unwrap() == &Type::Num);

        println!("{:?}", state.substitution);
    }
}
