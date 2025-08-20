use crate::lang::macros::{binop, cons, ite, let_, map_access, map_assign, tuple_access, unaryop};
use crate::lang::{Transformable, Type};

use crate::lang::{BinOpKinds, Expr, IsCurriedFunction, UnaryOpKinds};
use egg::{rewrite as rw, *};
use std::collections::HashSet;
use std::fmt::Debug;
use tracing::{event, Level};

use std::mem::take;

use super::{tfunc, tlist, tmap, tset};

define_language! {
    pub enum ExprLang {
        Num(i32),
        Bool(bool),
        "str" = Str(Id),

        "type-num" = TypeNum,
        "type-bool" = TypeBool,
        "type-str" = TypeStr,
        "type-list" = TypeList([Id; 1]),
        "type-tuple" = TypeTuple(Vec<Id>),
        "type-map" = TypeMap([Id; 2]),
        "type-set" = TypeSet([Id; 1]),
        "type-fn" = TypeFn([Id; 2]),

        "var" = Variable(Id),

        "--" = Neg([Id; 1]),
        "!" = Not([Id; 1]),

        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "++" = Concat([Id; 2]),
        ">" = Gt([Id; 2]),
        "<" = Lt([Id; 2]),
        "==" = Eq([Id; 2]),
        "&&" = And([Id; 2]),
        "||" = Or([Id; 2]),

        "ite" = Ite([Id; 3]),
        "app" = App([Id; 2]),
        "lam" = Lambda([Id; 3]),
        "let" = Let([Id; 3]),
        "nil" = Nil,
        "::" = Cons([Id; 2]),
        "tuple" = Tuple(Vec<Id>),
        "tuple-access" = TupleAccess([Id; 2]),
        "map-assign" = MapAssign([Id; 3]),
        "map-access" = MapAccess([Id; 2]),

        Symbol(Symbol),
    }
}

impl ExprLang {
    fn num(&self) -> Option<i32> {
        match self {
            ExprLang::Num(n) => Some(*n),
            _ => None,
        }
    }
}

type EGraph = egg::EGraph<ExprLang, ExprLangAnalysis>;

#[derive(Default)]
struct ExprLangAnalysis;

#[derive(Debug)]
struct ExprLangData {
    free: HashSet<Id>,
    constant: Option<(ExprLang, PatternAst<ExprLang>)>,
}

impl egg::Analysis<ExprLang> for ExprLangAnalysis {
    type Data = ExprLangData;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        let before_len = to.free.len();
        // to.free.extend(from.free);
        to.free.retain(|i| from.free.contains(i));
        // compare lengths to see if I changed to or from
        DidMerge(
            before_len != to.free.len(),
            to.free.len() != from.free.len(),
        ) | merge_option(&mut to.constant, from.constant, |a, b| {
            assert_eq!(a.0, b.0, "Merged non-equal constants");
            DidMerge(false, false)
        })
    }

    fn make(egraph: &mut EGraph, enode: &ExprLang) -> Self::Data {
        use ExprLang::*;

        let f = |i: &Id| egraph[*i].data.free.iter().cloned();
        let mut free = HashSet::default();
        match enode {
            Variable(v) => {
                free.insert(*v);
            }
            Let([v, a, b]) => {
                free.extend(f(b));
                free.remove(v);
                free.extend(f(a));
            }
            Lambda([v, _, a]) => {
                free.extend(f(a));
                free.remove(v);
            }
            _ => enode.for_each(|c| free.extend(&egraph[c].data.free)),
        }
        let constant = eval(egraph, enode);
        ExprLangData { constant, free }
    }

    fn modify(egraph: &mut egg::EGraph<ExprLang, Self>, id: Id) {
        if let Some(c) = egraph[id].data.constant.clone() {
            if egraph.are_explanations_enabled() {
                egraph.union_instantiations(
                    &c.0.to_string().parse().unwrap(),
                    &c.1,
                    &Default::default(),
                    "analysis".to_string(),
                );
            } else {
                let const_id = egraph.add(c.0);
                egraph.union(id, const_id);
            }
        }
    }
}

fn eval(egraph: &EGraph, enode: &ExprLang) -> Option<(ExprLang, PatternAst<ExprLang>)> {
    use ExprLang::*;

    let x = |i: &Id| egraph[*i].data.constant.as_ref().map(|c| &c.0);
    match enode {
        Num(n) => Some((enode.clone(), format!("{}", n).parse().unwrap())),
        Bool(b) => Some((enode.clone(), format!("{}", b).parse().unwrap())),
        Str(s) => Some((enode.clone(), format!("\"{}\"", s).parse().unwrap())),
        Add([a, b]) => Some((
            Num(x(a)?.num()?.checked_add(x(b)?.num()?)?),
            format!("(+ {} {})", x(a)?, x(b)?).parse().unwrap(),
        )),
        Eq([a, b]) => Some((
            Bool(x(a)? == x(b)?),
            format!("(== {} {})", x(a)?, x(b)?).parse().unwrap(),
        )),
        _ => None,
    }
}

fn is_const(v: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _, subst| egraph[subst[v]].data.constant.is_some()
}

fn is_not_same_var(v1: Var, v2: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _, subst| egraph.find(subst[v1]) != egraph.find(subst[v2])
}

fn var(s: &str) -> Var {
    s.parse().unwrap()
}

fn rules() -> Vec<Rewrite<ExprLang, ExprLangAnalysis>> {
    vec![
        // open term rules
        rw!("ite-true";  "(ite  true ?then ?else)" => "?then"),
        rw!("ite-false"; "(ite false ?then ?else)" => "?else"),
        rw!("ite-elim";  "(ite (== (var ?x) ?e) ?then ?else)" => "?else" if ConditionEqual::parse("(let ?x ?e ?then)", "(let ?x ?e ?else)")),
        rw!("add-comm";  "(+ ?a ?b)"        => "(+ ?b ?a)"),
        rw!("add-assoc"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
        rw!("add-ident"; "(+ ?a 0)"         => "?a"),
        rw!("sub-ident"; "(- ?a 0)"         => "?a"),
        rw!("mul-comm";  "(* ?a ?b)"        => "(* ?b ?a)"),
        rw!("mul-assoc"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),
        rw!("eq-comm";   "(== ?a ?b)"        => "(== ?b ?a)"),
        rw!("not-not";   "(! (! ?a))"       => "?a"),
        rw!("not-true";  "(! true)"         => "false"),
        rw!("not-false"; "(! false)"        => "true"),
        rw!("or-comm";   "(|| ?a ?b)"       => "(|| ?b ?a)"),
        rw!("and-comm";  "(&& ?a ?b)"       => "(&& ?b ?a)"),
        rw!("or-distr";  "(|| ?a (&& ?b ?c))" => "(&& (|| ?a ?b) (|| ?a ?c))"),
        rw!("and-distr"; "(&& ?a (|| ?b ?c))" => "(|| (&& ?a ?b) (&& ?a ?c))"),
        rw!("or-elim";   "(|| ?a ?a)"       => "?a"),
        rw!("and-elim";  "(&& ?a ?a)"       => "?a"),
        rw!("or-false-elim"; "(|| ?a false)" => "?a"),
        rw!("and-true-elim"; "(&& ?a true)" => "?a"),
        rw!("or-true-elim"; "(|| ?a true)" => "true"),
        rw!("and-false-elim"; "(&& ?a false)" => "false"),
        // subst rules
        rw!("beta";     "(app (lam ?v ?t ?body) ?e)" => "(let ?v ?e ?body)"),
        rw!("let-tuple2"; "(let ?v ?e (tuple ?a ?b))" => "(tuple (let ?v ?e ?a) (let ?v ?e ?b))"),
        rw!("let-tuple3"; "(let ?v ?e (tuple ?a ?b ?c))" => "(tuple (let ?v ?e ?a) (let ?v ?e ?b) (let ?v ?e ?c))"),
        rw!("let-tuple4"; "(let ?v ?e (tuple ?a ?b ?c ?d))" => "(tuple (let ?v ?e ?a) (let ?v ?e ?b) (let ?v ?e ?c) (let ?v ?e ?d))"),
        rw!("let-tuple5"; "(let ?v ?e (tuple ?a ?b ?c ?d ?ee))" => "(tuple (let ?v ?e ?a) (let ?v ?e ?b) (let ?v ?e ?c) (let ?v ?e ?d) (let ?v ?e ?ee))"),
        rw!("let-tuple6"; "(let ?v ?e (tuple ?a ?b ?c ?d ?ee ?f))" => "(tuple (let ?v ?e ?a) (let ?v ?e ?b) (let ?v ?e ?c) (let ?v ?e ?d) (let ?v ?e ?ee) (let ?v ?e ?f))"),
        rw!("let-tuple7"; "(let ?v ?e (tuple ?a ?b ?c ?d ?ee ?f ?g))" => "(tuple (let ?v ?e ?a) (let ?v ?e ?b) (let ?v ?e ?c) (let ?v ?e ?d) (let ?v ?e ?ee) (let ?v ?e ?f) (let ?v ?e ?g))"),
        rw!("let-tuple8"; "(let ?v ?e (tuple ?a ?b ?c ?d ?ee ?f ?g ?h))" => "(tuple (let ?v ?e ?a) (let ?v ?e ?b) (let ?v ?e ?c) (let ?v ?e ?d) (let ?v ?e ?ee) (let ?v ?e ?f) (let ?v ?e ?g) (let ?v ?e ?h))"),
        rw!("let-tuple9"; "(let ?v ?e (tuple ?a ?b ?c ?d ?ee ?f ?g ?h ?i))" => "(tuple (let ?v ?e ?a) (let ?v ?e ?b) (let ?v ?e ?c) (let ?v ?e ?d) (let ?v ?e ?ee) (let ?v ?e ?f) (let ?v ?e ?g) (let ?v ?e ?h) (let ?v ?e ?i))"),
        rw!("let-neg";    "(let ?v ?e (--   ?a))" => "(--  (let ?v ?e ?a))"),
        rw!("let-app";  "(let ?v ?e (app ?a ?b))" => "(app (let ?v ?e ?a) (let ?v ?e ?b))"),
        rw!("let-add";  "(let ?v ?e (+   ?a ?b))" => "(+   (let ?v ?e ?a) (let ?v ?e ?b))"),
        rw!("let-sub";  "(let ?v ?e (-   ?a ?b))" => "(-   (let ?v ?e ?a) (let ?v ?e ?b))"),
        rw!("let-mul";  "(let ?v ?e (*   ?a ?b))" => "(*   (let ?v ?e ?a) (let ?v ?e ?b))"),
        rw!("let-div";  "(let ?v ?e (/   ?a ?b))" => "(/   (let ?v ?e ?a) (let ?v ?e ?b))"),
        rw!("let-eq";   "(let ?v ?e (==  ?a ?b))" => "(==  (let ?v ?e ?a) (let ?v ?e ?b))"),
        rw!("let-lt";   "(let ?v ?e (<   ?a ?b))" => "(<   (let ?v ?e ?a) (let ?v ?e ?b))"),
        rw!("let-gt";   "(let ?v ?e (>   ?a ?b))" => "(>   (let ?v ?e ?a) (let ?v ?e ?b))"),
        rw!("let-and";  "(let ?v ?e (&&  ?a ?b))" => "(&&  (let ?v ?e ?a) (let ?v ?e ?b))"),
        rw!("let-or";   "(let ?v ?e (||  ?a ?b))" => "(||  (let ?v ?e ?a) (let ?v ?e ?b))"),
        rw!("let-concat"; "(let ?v ?e (++ ?a ?b))" => "(++ (let ?v ?e ?a) (let ?v ?e ?b))"),
        rw!("let-const";
            "(let ?v ?e ?c)" => "?c" if is_const(var("?c"))),
        rw!("let-ite";
            "(let ?v ?e (ite ?cond ?then ?else))" =>
            "(ite (let ?v ?e ?cond) (let ?v ?e ?then) (let ?v ?e ?else))"
        ),
        rw!("let-tuple-access";
            "(let ?v ?e (tuple-access ?t ?idx))" =>
            "(tuple-access (let ?v ?e ?t) (let ?v ?e ?idx))"
        ),
        rw!("let-var-same"; "(let ?v1 ?e (var ?v1))" => "?e"),
        rw!("let-var-diff"; "(let ?v1 ?e (var ?v2))" => "(var ?v2)"
            if is_not_same_var(var("?v1"), var("?v2"))),
        rw!("let-lam-same"; "(let ?v1 ?e (lam ?v1 ?t ?body))" => "(lam ?v1 ?t ?body)"),
        rw!("let-lam-diff";
            "(let ?v1 ?e (lam ?v2 ?t ?body))" =>
            { CaptureAvoid {
                fresh: var("?fresh"), v2: var("?v2"), e: var("?e"),
                if_not_free: "(lam ?v2 ?t (let ?v1 ?e ?body))".parse().unwrap(),

                // ?v2 could be free in ?e, so we need to rename it
                if_free: "(lam ?fresh ?t (let ?v1 ?e (let ?v2 (var ?fresh) ?body)))".parse().unwrap(),
            }}
            if is_not_same_var(var("?v1"), var("?v2"))),
        // tuple rule
        rw!("tuple-access-tuple2-0"; "(tuple-access (tuple ?a ?b) 0)" => "?a"),
        rw!("tuple-access-tuple2-1"; "(tuple-access (tuple ?a ?b) 1)" => "?b"),
        rw!("tuple-access-tuple3-0"; "(tuple-access (tuple ?a ?b ?c) 0)" => "?a"),
        rw!("tuple-access-tuple3-1"; "(tuple-access (tuple ?a ?b ?c) 1)" => "?b"),
        rw!("tuple-access-tuple3-2"; "(tuple-access (tuple ?a ?b ?c) 2)" => "?c"),
        rw!("batched-accumulator"; "(lam ?s ?t1 (lam ?x ?t2 (app (app (app (var foldl) ?f) (var ?s)) (var ?x))))" => "?f"),
    ]
}

struct CaptureAvoid {
    fresh: Var,
    v2: Var,
    e: Var,
    if_not_free: Pattern<ExprLang>,
    if_free: Pattern<ExprLang>,
}

impl Applier<ExprLang, ExprLangAnalysis> for CaptureAvoid {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<ExprLang>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let e = subst[self.e];
        let v2 = subst[self.v2];
        let v2_free_in_e = egraph[e].data.free.contains(&v2);
        if v2_free_in_e {
            let mut subst = subst.clone();
            let sym = ExprLang::Symbol(format!("_{}", eclass).into());
            subst.insert(self.fresh, egraph.add(sym));
            self.if_free
                .apply_one(egraph, eclass, &subst, searcher_ast, rule_name)
        } else {
            self.if_not_free
                .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
        }
    }
}

fn add_type(egraph: &mut EGraph, typ: &Type) -> Id {
    use ExprLang::*;
    match typ {
        Type::Num => egraph.add(TypeNum),
        Type::Bool => egraph.add(TypeBool),
        Type::Str => egraph.add(TypeStr),
        Type::List(t) => {
            let id = add_type(egraph, t);
            egraph.add(TypeList([id]))
        }
        Type::Tuple(tys) => {
            let ids = tys.iter().map(|t| add_type(egraph, t)).collect();
            egraph.add(TypeTuple(ids))
        }
        Type::Map(tk, tv) => {
            let id_k = add_type(egraph, tk);
            let id_v = add_type(egraph, tv);
            egraph.add(TypeMap([id_k, id_v]))
        }
        Type::Set(t) => {
            let id = add_type(egraph, t);
            egraph.add(TypeSet([id]))
        }
        Type::Fn(arg, ret) => {
            let arg = add_type(egraph, arg);
            let ret = add_type(egraph, ret);
            egraph.add(TypeFn([arg, ret]))
        }
        Type::Var(..) => unimplemented!(),
    }
}

fn add_expr(egraph: &mut EGraph, expr: &Expr) -> Id {
    use ExprLang::*;

    match expr {
        Expr::Var(v) => {
            let s = egraph.add(Symbol(*v));
            egraph.add(Variable(s))
        }
        Expr::Num(n) => egraph.add(Num(*n)),
        Expr::Bool(b) => egraph.add(Bool(*b)),
        Expr::Str(s) => {
            let id = egraph.add(Symbol(s.as_str().into()));
            egraph.add(Str(id))
        }

        Expr::UnaryOp(op, expr) => {
            let id = add_expr(egraph, expr);
            match op {
                crate::lang::UnaryOpKinds::Neg => egraph.add(Neg([id])),
                crate::lang::UnaryOpKinds::Not => egraph.add(Not([id])),
            }
        }

        Expr::BinOp(op, lhs, rhs) => {
            let lhs = add_expr(egraph, lhs);
            let rhs = add_expr(egraph, rhs);
            match op {
                crate::lang::BinOpKinds::Add => egraph.add(Add([lhs, rhs])),
                crate::lang::BinOpKinds::Sub => egraph.add(Sub([lhs, rhs])),
                crate::lang::BinOpKinds::Mul => egraph.add(Mul([lhs, rhs])),
                crate::lang::BinOpKinds::Div => egraph.add(Div([lhs, rhs])),
                crate::lang::BinOpKinds::Concat => egraph.add(Concat([lhs, rhs])),
                crate::lang::BinOpKinds::Gt => egraph.add(Gt([lhs, rhs])),
                crate::lang::BinOpKinds::Lt => egraph.add(Lt([lhs, rhs])),
                crate::lang::BinOpKinds::Eq => egraph.add(Eq([lhs, rhs])),
                crate::lang::BinOpKinds::And => egraph.add(And([lhs, rhs])),
                crate::lang::BinOpKinds::Or => egraph.add(Or([lhs, rhs])),
            }
        }

        Expr::Ite {
            cond,
            then_expr,
            else_expr,
        } => {
            let cond = add_expr(egraph, cond);
            let then_expr = add_expr(egraph, then_expr);
            let else_expr = add_expr(egraph, else_expr);
            egraph.add(Ite([cond, then_expr, else_expr]))
        }

        Expr::App { func, arg } => {
            let func = add_expr(egraph, func);
            let arg = add_expr(egraph, arg);
            egraph.add(App([func, arg]))
        }

        Expr::Lambda {
            param,
            body,
            param_type,
        } => {
            let param = egraph.add(Symbol(*param));
            let typ = add_type(egraph, param_type);
            let body = add_expr(egraph, body);
            egraph.add(Lambda([param, typ, body]))
        }

        Expr::Let { name, expr, body } => {
            let name = egraph.add(Symbol(*name));
            let expr = add_expr(egraph, expr);
            let body = add_expr(egraph, body);
            egraph.add(Let([name, expr, body]))
        }

        Expr::Nil => egraph.add(Nil),

        Expr::Cons { head, tail } => {
            let head = add_expr(egraph, head);
            let tail = add_expr(egraph, tail);
            egraph.add(Cons([head, tail]))
        }

        Expr::Tuple(exprs) => {
            let exprs = exprs.iter().map(|e| add_expr(egraph, e)).collect();
            egraph.add(Tuple(exprs))
        }

        Expr::TupleAccess(expr, idx) => {
            let expr = add_expr(egraph, expr);
            let idx = egraph.add(Num(*idx as i32));
            egraph.add(TupleAccess([expr, idx]))
        }

        Expr::MapAssign { map, key, value } => {
            let map = add_expr(egraph, map);
            let key = add_expr(egraph, key);
            let value = add_expr(egraph, value);
            egraph.add(MapAssign([map, key, value]))
        }

        Expr::MapAccess { map, key } => {
            let map = add_expr(egraph, map);
            let key = add_expr(egraph, key);
            egraph.add(MapAccess([map, key]))
        }
    }
}

pub(crate) struct LangAnalyzer {
    egraph: EGraph,
}

impl Default for LangAnalyzer {
    fn default() -> Self {
        Self {
            egraph: EGraph::default(),
        }
    }
}

impl LangAnalyzer {
    pub fn new(expr: &Expr) -> Self {
        let mut egraph = EGraph::default();
        add_expr(&mut egraph, expr);
        Self { egraph }
    }

    fn run(&mut self) {
        let egraph = take(&mut self.egraph);

        let runner = egg::Runner::default().with_egraph(egraph).run(&rules());
        self.egraph = runner.egraph;
    }

    pub fn check_eq(&mut self, expr1: &Expr, expr2: &Expr) -> bool {
        let id1 = add_expr(&mut self.egraph, expr1);
        let id2 = add_expr(&mut self.egraph, expr2);

        self.run();
        self.egraph.find(id1) == self.egraph.find(id2)
    }
}

fn get_symbol(rexp: &RecExpr<ExprLang>, idx: &Id) -> Symbol {
    let ExprLang::Symbol(s) = &rexp[*idx] else {
        panic!("Expected symbol");
    };

    *s
}

fn build_type(rexp: &RecExpr<ExprLang>, idx: &Id) -> Type {
    match &rexp[*idx] {
        ExprLang::TypeNum => Type::Num,
        ExprLang::TypeStr => Type::Str,
        ExprLang::TypeBool => Type::Bool,
        ExprLang::TypeList([t]) => tlist!(build_type(rexp, t)),
        ExprLang::TypeTuple(tys) => Type::Tuple(tys.iter().map(|i| build_type(rexp, i)).collect()),
        ExprLang::TypeMap([ty_k, ty_v]) => tmap!(build_type(rexp, ty_k), build_type(rexp, ty_v)),
        ExprLang::TypeSet([t]) => tset!(build_type(rexp, t)),
        ExprLang::TypeFn([ty_arg, ty_ret]) => {
            tfunc!(build_type(rexp, ty_arg) => build_type(rexp, ty_ret))
        }

        _ => panic!("invalid id for type"),
    }
}

fn build_expr(rexp: &RecExpr<ExprLang>, idx: &Id) -> Expr {
    match &rexp[*idx] {
        ExprLang::Num(v) => Expr::Num(*v),
        ExprLang::Bool(v) => Expr::Bool(*v),
        ExprLang::Str(id) => Expr::Str(get_symbol(rexp, id).to_string()),
        ExprLang::Variable(id) => Expr::Var(get_symbol(rexp, id)),
        ExprLang::Neg([v]) => unaryop!(UnaryOpKinds::Neg, build_expr(rexp, v)),
        ExprLang::Not([v]) => unaryop!(UnaryOpKinds::Not, build_expr(rexp, v)),
        ExprLang::Add([l, r]) => binop!(BinOpKinds::Add, build_expr(rexp, l), build_expr(rexp, r)),
        ExprLang::Sub([l, r]) => binop!(BinOpKinds::Sub, build_expr(rexp, l), build_expr(rexp, r)),
        ExprLang::Mul([l, r]) => binop!(BinOpKinds::Mul, build_expr(rexp, l), build_expr(rexp, r)),
        ExprLang::Div([l, r]) => binop!(BinOpKinds::Div, build_expr(rexp, l), build_expr(rexp, r)),
        ExprLang::Concat([l, r]) => {
            binop!(BinOpKinds::Concat, build_expr(rexp, l), build_expr(rexp, r))
        }
        ExprLang::Gt([l, r]) => binop!(BinOpKinds::Gt, build_expr(rexp, l), build_expr(rexp, r)),
        ExprLang::Lt([l, r]) => binop!(BinOpKinds::Lt, build_expr(rexp, l), build_expr(rexp, r)),
        ExprLang::Eq([l, r]) => binop!(BinOpKinds::Eq, build_expr(rexp, l), build_expr(rexp, r)),
        ExprLang::And([l, r]) => binop!(BinOpKinds::And, build_expr(rexp, l), build_expr(rexp, r)),
        ExprLang::Or([l, r]) => binop!(BinOpKinds::Or, build_expr(rexp, l), build_expr(rexp, r)),
        ExprLang::Ite([c, then, els]) => ite!(
            build_expr(rexp, c),
            build_expr(rexp, then),
            build_expr(rexp, els)
        ),
        ExprLang::App([func, arg]) => build_expr(rexp, func).app(build_expr(rexp, arg)),
        ExprLang::Lambda([param, typ, body]) => Expr::Lambda {
            param: get_symbol(rexp, param),
            param_type: build_type(rexp, typ),
            body: Box::new(build_expr(rexp, body)),
        },
        ExprLang::Let([name, expr, body]) => let_!(
            get_symbol(rexp, name),
            build_expr(rexp, expr),
            build_expr(rexp, body)
        ),
        ExprLang::Nil => Expr::Nil,
        ExprLang::Cons([head, tail]) => cons!(build_expr(rexp, head), build_expr(rexp, tail)),
        ExprLang::Tuple(vec) => Expr::Tuple(vec.iter().map(|i| build_expr(rexp, i)).collect()),
        ExprLang::TupleAccess([tup, idx]) => {
            let Expr::Num(idx) = build_expr(rexp, idx) else {
                panic!("Expected number");
            };
            tuple_access!(build_expr(rexp, tup), idx as usize)
        }
        ExprLang::MapAssign([map, key, value]) => map_assign!(
            build_expr(rexp, map),
            build_expr(rexp, key),
            build_expr(rexp, value)
        ),
        ExprLang::MapAccess([map, key]) => {
            map_access!(build_expr(rexp, map), build_expr(rexp, key))
        }
        ExprLang::Symbol(symbol) => panic!("Unexpected symbol {:?}", symbol),
        ExprLang::TypeBool
        | ExprLang::TypeStr
        | ExprLang::TypeList(..)
        | ExprLang::TypeMap(..)
        | ExprLang::TypeNum
        | ExprLang::TypeSet(..)
        | ExprLang::TypeTuple(..)
        | ExprLang::TypeFn(..) => panic!("invalid id"),
    }
}

fn postprocess(expr: Expr, orig: &Expr) -> Expr {
    fn simplify_tuple_access(expr: &Expr, _orig: &Expr) -> Option<Expr> {
        Some(expr.clone().transform(&mut |expr| match expr {
            Expr::TupleAccess(tup, idx) => match tup.as_ref() {
                Expr::Tuple(tups) => tups.get(idx as usize).unwrap().clone(),
                _ => Expr::TupleAccess(tup, idx),
            },
            _ => expr,
        }))
    }

    fn inline_let_bindings(expr: &Expr, _orig: &Expr) -> Option<Expr> {
        Some(expr.clone().inline_let_bindings())
    }

    vec![inline_let_bindings, simplify_tuple_access]
        .into_iter()
        .fold(expr, |expr, f| f(&expr, orig).unwrap_or(expr))
}

pub fn simplify(expr: &Expr) -> Expr {
    let mut egraph = EGraph::default();
    let id = add_expr(&mut egraph, expr);

    let runner = egg::Runner::default().with_egraph(egraph).run(&rules());

    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (best_cost, best) = extractor.find_best(id);

    let nodes = best.as_ref();
    let last = nodes.len() - 1;
    let simp_expr = postprocess(build_expr(&best, &last.into()), expr);

    event!(
        Level::INFO,
        cost = best_cost,
        expr = expr.to_string(),
        simp = simp_expr.to_string(),
        "Expression simplified",
    );

    simp_expr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::{macros::*, BinOpKinds};

    #[test]
    fn test_init() {
        let expr = Expr::BinOp(
            BinOpKinds::Add,
            Box::new(Expr::from(sym!(x))),
            Box::new(Expr::from(1)),
        )
        .bind_lets(vec![(sym!(x), Expr::from(sym!(y)))]);

        let expr2 = Expr::BinOp(
            BinOpKinds::Add,
            Box::new(Expr::from(sym!(y))),
            Box::new(Expr::from(1)),
        );

        let mut checker = LangAnalyzer::default();
        assert!(checker.check_eq(&expr, &expr2));

        let expr3 = Expr::BinOp(
            BinOpKinds::Add,
            Box::new(Expr::from(sym!(z))),
            Box::new(Expr::from(1)),
        )
        .bind_lets(vec![(sym!(z), Expr::from(sym!(y)))]);
        assert!(checker.check_eq(&expr, &expr3));
    }

    egg::test_fn! {
        base_1, rules(), "(let x (var y) (+ (var x) 1))" => "(+ (var y) 1)",
    }

    egg::test_fn! {
        tuple_access, rules(),
        "(tuple-access (tuple 1 2 3) 2)"
        =>
        "3",
    }

    egg::test_fn! {
        lambda_under, rules(),
        "(lam x type-num (+ 4
                   (app (lam y type-num (var y))
                        4)))"
        =>
        "(lam x type-num 8))",
    }

    egg::test_fn! {
        lambda_let, rules(),
        "(let f (lam x type-num (var x)) (app (var f) (var y)))"
        =>
        "(var y)",
    }

    egg::test_fn! {
        lambda_let0, rules(),
        "(app (lam x type-num (var x)) (var y))"
        =>
        "(var y)",
    }

    egg::test_fn! {
        tuple_access_let, rules(),
        "(let z (var x) (tuple-access (var z) 20))"
        =>
        "(tuple-access (var x) 20)",
    }

    egg::test_fn! {
        batched, rules(),
        "(lam s type-num (lam y type-str (app (app (app (var foldl) (var f)) (var s)) (var y))))"
        =>
        "(var f)",
    }

    egg::test_fn! {
        batched2, rules(),
        "(lam acc (type-map type-num (type-tuple type-num type-num)) (lam x (type-list (type-tuple type-num type-num)) (app (app (app (var foldl) (lam acc (type-map type-num (type-tuple type-num type-num)) (lam x_row (type-tuple type-num type-num) (map-assign (var acc) (tuple-access (var x_row) 0) (var x_row))))) (var acc)) (var x))))"
        =>
        "(lam acc (type-map type-num (type-tuple type-num type-num)) (lam x_row (type-tuple type-num type-num) (map-assign (var acc) (tuple-access (var x_row) 0) (var x_row))))",
    }

    #[test_log::test]
    fn test_basic_check_eq() {
        let expr1 = letv!(z, sym!(x).into(), tuple_access!(sym!(z).into(), 20));
        let expr2 = tuple_access!(sym!(x).into(), 20);

        let mut checker = LangAnalyzer::default();
        assert!(checker.check_eq(&expr1, &expr2));
    }

    #[test_log::test]
    fn test_simplify() {
        let expr = letv!(z, sym!(x).into(), tuple_access!(sym!(z).into(), 20));
        assert_eq!(simplify(&expr).to_string(), "x._20");

        let fold_body = var!(foldl)
            .call(vec![
                map_assign!(var!(acc), tuple_access!(var!(x_row), 0).into(), var!(x_row))
                    .bind_params(vec![
                        param!(acc, tmap!(Type::Num, ttuple![Type::Num, Type::Num])),
                        param!(x_row, ttuple![Type::Num, Type::Num]),
                    ]),
                var!(acc),
                var!(x),
            ])
            .bind_params(vec![
                param!(acc, tmap!(Type::Num, ttuple![Type::Num, Type::Num])),
                param!(x, tlist![ttuple![Type::Num, Type::Num]]),
            ]);
        assert_eq!(
            simplify(&fold_body).to_string(),
            "λ(acc: map[num, (num, num)], x_row: (num, num)). acc[x_row._0 <- x_row]"
        )
    }
}
