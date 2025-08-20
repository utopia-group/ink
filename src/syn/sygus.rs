use super::{Assumptions, FunctionHasFreeVar, NormalizerConstraint, NormalizerSynthesisFailure};
use crate::consts::{as_id_str, CVC5_TIMEOUT_SECS};
use crate::lang::macros::tuple_access;
use crate::lang::{
    format_define_fun, macros::*, parse_binop, simplify, BasicSubexprTypeMap, BinOpKinds, Expr,
    FormatSyGuS, IsCurriedFunction, PlaceholderEnv, PolyType, Type, TypeEnv, TypedEnv,
};
use crate::local_config::CVC5_PATH;

use crate::Symbol;
use sexp::{parse, Atom, Sexp};
use std::collections::HashMap;
use std::io::Read;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tracing::{event, Level};
use wait_timeout::ChildExt;

const ZIP_LIST_NUM: &str = r#"
(define-fun-rec zip ((s1 (Seq Int)) (s2 (Seq Int))) (Seq (Tuple Int Int))
  (ite (or (= (seq.len s1) 0) (= (seq.len s2) 0))
    (as seq.empty (Seq (Tuple Int Int)))
    (seq.++ 
      (seq.unit (tuple (seq.nth s1 0) (seq.nth s2 0)))
      (zip (seq.extract s1 1 (- (seq.len s1) 1)) 
           (seq.extract s2 1 (- (seq.len s2) 1))))))

(define-fun-rec map ((f (-> (Tuple Int Int) Int)) (s (Seq (Tuple Int Int)))) (Seq Int)
  (ite (= (seq.len s) 0)
    (as seq.empty (Seq Int))
    (seq.++ 
      (seq.unit (f (seq.nth s 0)))
      (map f (seq.extract s 1 (- (seq.len s) 1))))))
"#;

const ZIP_TUPLE_NESTED: &str = r#"
(define-fun-rec zip ((s1 (Seq (Tuple (Seq Int) Int Int))) (s2 (Seq (Tuple (Seq Int) Int Int)))) (Seq (Tuple (Tuple (Seq Int) Int Int) (Tuple (Seq Int) Int Int)))
  (ite (or (= (seq.len s1) 0) (= (seq.len s2) 0))
    (as seq.empty (Seq (Tuple (Tuple (Seq Int) Int Int) (Tuple (Seq Int) Int Int))))
    (seq.++ 
      (seq.unit (tuple (seq.nth s1 0) (seq.nth s2 0)))
      (zip (seq.extract s1 1 (- (seq.len s1) 1)) 
                  (seq.extract s2 1 (- (seq.len s2) 1))))))

(define-fun-rec map ((f (-> (Tuple (Tuple (Seq Int) Int Int) (Tuple (Seq Int) Int Int)) (Tuple (Seq Int) Int Int))) (s (Seq (Tuple (Tuple (Seq Int) Int Int) (Tuple (Seq Int) Int Int))))) (Seq (Tuple (Seq Int) Int Int))
  (ite (= (seq.len s) 0)
    (as seq.empty (Seq (Tuple (Seq Int) Int Int)))
    (seq.++ 
      (seq.unit (f (seq.nth s 0)))
      (map f (seq.extract s 1 (- (seq.len s) 1))))))
"#;

pub enum SynthesisMode {
    Normalizer,
    NormalizerFromPartialResult {
        ret_type: Type,
        partial_norm: Expr,
    },
    MergeOperator {
        num_left_args: usize,
        num_right_args: usize,
    },
}

impl Default for SynthesisMode {
    fn default() -> Self {
        SynthesisMode::Normalizer
    }
}

const SYGUS_FILE_EXT: &str = ".sy";

fn parse_cvc5_fn_name(v: &str) -> &str {
    match v {
        "set.insert" => "set_add",
        "set.inter" => "intersection",
        "set.union" => "union",
        "set.empty" => "empty_set",
        "set.filter" => "filter_set",
        "set.map" => "map_set",
        "set.is_empty" => "is_set_empty",

        v => v.split('_').next().unwrap(),
    }
}

struct GrammarState {
    nonterms: HashMap<Type, GrammarDesc>,
    fv_counter: usize,
}

impl GrammarState {
    fn new() -> Self {
        Self {
            nonterms: HashMap::from([(
                Type::Bool,
                GrammarDesc {
                    non_term_id: "B".into(),
                    exprs: vec!["true", "false", "(not B)", "(and B B)", "(or B B)"]
                        .into_iter()
                        .map(String::from)
                        .collect(),
                    assumptions: vec![],
                    library_functions: vec![],
                },
            )]),
            fv_counter: 0,
        }
    }

    /// get the non-terminal name for the given type
    fn get_nonterm(&self, typ: &Type) -> &str {
        self.nonterms
            .get(typ)
            .map(|desc| desc.non_term_id.as_str())
            .unwrap_or_else(|| panic!("non-terminal not found for type: {}", typ.format()))
    }

    /// create a fresh non-terminal name with the given prefix
    fn new_nonterm_name(&mut self, prefix: &str) -> String {
        let name = format!("{}{}", prefix, self.fv_counter);
        self.fv_counter += 1;
        name
    }

    /// add possible expressions using this variable
    fn add_variable(&mut self, var: Symbol, typ: &Type) {
        let desc = if let Some(desc) = self.nonterms.get_mut(typ) {
            desc
        } else {
            self.add_type(typ);
            self.nonterms.get_mut(typ).unwrap()
        };

        desc.exprs.push(var.to_string());
        if matches!(typ, Type::List(..)) {
            self.add_type(&Type::Num);
            self.nonterms
                .get_mut(&Type::Num)
                .unwrap()
                .exprs
                .push(format!("(seq.len {})", var));
        }

        if matches!(typ, Type::Str) {
            self.add_type(&Type::Num);
            self.nonterms
                .get_mut(&Type::Num)
                .unwrap()
                .exprs
                .push(format!("(str.len {})", var));
        }
    }

    fn add_boolean_expressions(&mut self, conditions: Vec<String>) {
        self.nonterms
            .get_mut(&Type::Bool)
            .unwrap()
            .exprs
            .extend(conditions);
    }

    /// add a new non-terminal for the given type
    fn add_type(&mut self, typ: &Type) {
        if self.nonterms.contains_key(typ) {
            return;
        }

        match typ {
            Type::Var(..) | Type::Fn(..) | Type::Bool => panic!("invalid type"),
            Type::Str => {
                self.nonterms.insert(
                    typ.clone(),
                    GrammarDesc {
                        non_term_id: "S".into(),
                        exprs: vec!["\"\"", "(str.++ S S)"]
                            .into_iter()
                            .map(String::from)
                            .collect(),

                        assumptions: vec![],
                        library_functions: vec![],
                    },
                );
            }
            Type::Num => {
                self.nonterms.insert(
                    typ.clone(),
                    GrammarDesc {
                        non_term_id: "N".into(),
                        exprs: vec![
                            "0",
                            "1",
                            "(+ N N)",
                            "(- N N)",
                            "(* N N)",
                            "(- N)",
                            "(max N N)",
                            "(min N N)",
                        ]
                        .into_iter()
                        .map(String::from)
                        .collect(),

                        assumptions: vec!["(and (<= _mn $var) (<= $var _mx))"]
                            .into_iter()
                            .map(|s| format!("(assume {})", s))
                            .collect(),

                        library_functions: vec![],
                    },
                );

                self.add_boolean_expressions(
                    vec!["(< N N)", "(> N N)", "(= N N)"]
                        .into_iter()
                        .map(String::from)
                        .collect(),
                );
            }
            Type::List(element_type) => {
                self.add_type(element_type);
                let element_nonterm = self.get_nonterm(element_type).to_string();

                let non_term_id = self.new_nonterm_name(format!("L_{}", element_nonterm).as_str());

                // ad hoc way to support zip/map
                let mut library_functions = vec![];
                match element_type.as_ref() {
                    Type::Num => {
                        library_functions.push(ZIP_LIST_NUM.to_string());
                    }
                    Type::Tuple(..) => {
                        library_functions.push(ZIP_TUPLE_NESTED.to_string());
                    }
                    _ => {}
                }

                self.nonterms.insert(
                    typ.clone(),
                    GrammarDesc {
                        non_term_id: non_term_id.clone(),
                        exprs: vec![
                            format!("(as seq.empty {typ})", typ = typ.format()),
                            format!("(seq.unit {})", element_nonterm),
                            format!("(seq.++ {} {})", non_term_id, non_term_id),
                        ],
                        assumptions: vec![],
                        library_functions,
                    },
                );
            }
            Type::Tuple(tys) => {
                for ty in tys {
                    self.add_type(ty);
                }

                let non_term_id = self.new_nonterm_name("T");

                // add tuple access expressions to grammar
                for (i, ty) in tys.iter().enumerate() {
                    self.nonterms.get_mut(ty).unwrap().exprs.push(
                        (
                            &tuple_access!(non_term_id.as_str().into(), i),
                            &PlaceholderEnv as &dyn TypedEnv,
                        )
                            .format(),
                    );
                }

                let assumptions = {
                    let mut assumptions = vec![];

                    // add asssumptions for each type
                    for (i, ty) in tys.iter().enumerate() {
                        for assumption in self.nonterms.get(ty).unwrap().assumptions.iter() {
                            let assumption = assumption.replace(
                                "$var",
                                (
                                    &tuple_access!(Expr::Var("$var".into()), i),
                                    &PlaceholderEnv as &dyn TypedEnv,
                                )
                                    .format()
                                    .as_str(),
                            );
                            assumptions.push(assumption);
                        }
                    }

                    assumptions
                };

                self.nonterms.insert(
                    typ.clone(),
                    GrammarDesc {
                        non_term_id: non_term_id.clone(),
                        exprs: vec![format!(
                            "(tuple {})",
                            tys.iter()
                                .map(|t| self.get_nonterm(t))
                                .collect::<Vec<_>>()
                                .join(" ")
                        )],
                        assumptions,
                        library_functions: vec![],
                    },
                );
            }
            Type::Set(element_type) => {
                self.add_type(element_type);
                let element_nonterm = self.get_nonterm(element_type).to_string();

                let non_term_id = self.new_nonterm_name(format!("S_{}", element_nonterm).as_str());
                let typ_str = typ.format();

                self.nonterms.insert(
                    typ.clone(),
                    GrammarDesc {
                        non_term_id: non_term_id.clone(),
                        exprs: vec![
                            format!("(as set.empty {})", typ_str),
                            format!("(set.union {} {})", non_term_id, non_term_id),
                            format!("(set.insert {} {})", element_nonterm, non_term_id),
                        ],
                        assumptions: vec![],
                        library_functions: vec![format_set_prelude(typ)],
                    },
                );

                self.add_boolean_expressions(vec![format!(
                    "(isSetEmpty_{} {})",
                    as_id_str(typ_str.as_str()),
                    non_term_id
                )]);
            }
            Type::Map(ty_k, ty_v) => {
                self.add_type(ty_k);
                self.add_type(ty_v);

                let key_nonterm = self.get_nonterm(ty_k).to_string();
                let value_nonterm = self.get_nonterm(ty_v).to_string();

                let nonterm_id =
                    self.new_nonterm_name(format!("M_{}_{}", key_nonterm, value_nonterm).as_str());
                let typ_str = typ.format();
                let id_str = as_id_str(typ_str.as_str());

                self.nonterms.insert(
                    typ.clone(),
                    GrammarDesc {
                        non_term_id: nonterm_id.clone(),
                        exprs: vec![
                            format!("(as set.empty {})", typ_str),
                            format!(
                                "(update_{id_str} {} {} {})",
                                nonterm_id, key_nonterm, value_nonterm
                            ),
                            format!("(concatMap_{id_str} {} {})", nonterm_id, nonterm_id),
                        ],
                        assumptions: vec![],
                        library_functions: vec![format_map_prelude(typ)],
                    },
                );
            }
        }

        let non_term_id = self.get_nonterm(typ).to_string();

        // add ite expression
        self.nonterms
            .get_mut(typ)
            .unwrap()
            .exprs
            .push(format!("(ite B {} {})", non_term_id, non_term_id));
    }
}

struct GrammarDesc {
    non_term_id: String,
    exprs: Vec<String>,
    assumptions: Vec<String>,
    library_functions: Vec<String>,
}

fn format_set_prelude(typ: &Type) -> String {
    assert!(matches!(typ, Type::Set(_)), "expected a set type");

    let typ_str = typ.format();
    let typ_id = as_id_str(typ_str.as_str());

    let is_empty_func_name = format!("isSetEmpty_{typ_id}");
    let is_empty = format!(
        r#"
(define-fun {is_empty_func_name} ((s {typ_str})) Bool
    (> (set.card s) 0)
)
    "#
    );

    vec![is_empty].join("")
}

fn format_map_prelude(typ: &Type) -> String {
    let Type::Map(typ_k, typ_v) = typ else {
        panic!("expected a map type")
    };
    let typ_k_str = typ_k.format();
    let typ_v_str = typ_v.format();
    let typ_str = typ.format();
    let typ_id = as_id_str(typ_str.as_str());

    let contains_key_func_name = format!("containsKey_{typ_id}");
    let update_func_name = format!("update_{typ_id}");
    let concat_map_func_name = format!("concatMap_{typ_id}");
    let filter_values_func_name = format!("filterValues_{typ_id}");
    let access_func_name = format!("access_{typ_id}");
    let map_values_func_name = format!("mapValues_{typ_id}");

    let contains_key = format!(
        r#"
(define-fun {contains_key_func_name} ((m {typ_str}) (k {typ_k_str})) Bool
  (exists ((t (Tuple {typ_k_str} {typ_v_str}))) (and (set.member t m) (= k ((_ tuple.select 0) t))))
)
"#
    );

    let update = format!(
        r#"
(define-fun {update_func_name} ((m {typ_str}) (k {typ_k_str}) (v {typ_v_str})) {typ_str}
  (set.union
    (set.filter (lambda ((t (Tuple {typ_k_str} {typ_v_str}))) (not (= k ((_ tuple.select 0) t)))) m)
    (set.singleton (tuple k v))
  )
)
"#
    );

    let concat_map = format!(
        r#"
(define-fun {concat_map_func_name} ((m1 {typ_str}) (m2 {typ_str})) {typ_str}
  (set.union
    m2
    (set.filter (lambda ((_val_y (Tuple {typ_k_str} {typ_v_str}))) (not ({contains_key_func_name} m2 ((_ tuple.select 0) _val_y)))) m1)
))
"#
    );

    let filter_values = format!(
        r#"
(define-fun {filter_values_func_name} ((p (-> {typ_v_str} Bool)) (m {typ_str})) {typ_str}
  (set.filter (lambda ((t (Tuple {typ_k_str} {typ_v_str}))) (p ((_ tuple.select 1) t))) m)
)
"#
    );

    let access = format!(
        r#"
(define-fun {access_func_name} ((m {typ_str}) (k {typ_k_str})) {typ_v_str}
  ((_ tuple.select 1) (set.choose (set.filter (lambda ((t (Tuple {typ_k_str} {typ_v_str}))) (= k ((_ tuple.select 0) t))) m)))
)
"#
    );

    // FIXME: it doesn't support mapping to other types
    let map_values = format!(
        r#"
(define-fun {map_values_func_name} ((p (-> {typ_k_str} (Tuple Bool {typ_v_str}) (Tuple Bool {typ_v_str}))) (m (Set (Tuple {typ_k_str} {typ_v_str})))) (Set (Tuple {typ_k_str} {typ_v_str}))
  (set.map (lambda ((t (Tuple {typ_k_str} {typ_v_str}))) (tuple ((_ tuple.select 0) t) ((_ tuple.select 1) (p ((_ tuple.select 0) t) (tuple true ((_ tuple.select 1) t)))))) m)
)
"#
    );

    vec![
        contains_key,
        update,
        concat_map,
        filter_values,
        access,
        map_values,
    ]
    .join("")
}

fn format_nonterm_grammar(typ: &Type, desc: &GrammarDesc) -> String {
    format!(
        r#"({nonterm} {typ} (
      {exprs}
    ))"#,
        nonterm = desc.non_term_id,
        typ = typ.format(),
        exprs = desc.exprs.join("\n      ")
    )
}

fn format_synth_fun(acc_type: &Type, ret_type: &Type, state: &GrammarState, id: &str) -> String {
    // always put the output type (either state type or bool) first
    let nonterms = {
        let mut nonterms = state
            .nonterms
            .iter()
            .filter(|(t, _)| **t != *ret_type)
            .collect::<Vec<_>>();
        nonterms.sort_by_key(|(t, _)| t.to_string());

        let output_nonterm = state.nonterms.get(ret_type).unwrap();
        nonterms.insert(0, (ret_type, output_nonterm));
        nonterms
    };

    fn format_nonterm(state: &GrammarState, typ: &Type) -> String {
        let desc = state.nonterms.get(typ).unwrap();
        format!("({} {})", desc.non_term_id, typ.format())
    }

    let (nonterms, nonterm_grammars): (Vec<String>, Vec<String>) = nonterms
        .into_iter()
        .map(|(k, v)| (format_nonterm(&state, k), format_nonterm_grammar(k, v)))
        .collect::<Vec<_>>()
        .into_iter()
        .unzip();

    let nonterms = nonterms.join(" ");
    let nonterm_grammars = nonterm_grammars.join("\n    ");

    let acc_type_str = acc_type.format();
    let ret_type_str = ret_type.format();
    format!(
        r#"
(synth-fun {id} ((val_l {acc_type_str}) (val_r {acc_type_str})) {ret_type_str}
  ({nonterms})
  (
    {nonterm_grammars}
  )
)"#
    )
}

fn format_prelude(
    acc_type: &Type,
    ret_type: &Type,
    item_type: &Type,
    assumptions: &Assumptions,
    variables: &Vec<(Symbol, &Type)>,
) -> String {
    let mut state = GrammarState::new();
    // parameters of the merge function to be synthesized
    state.add_variable("val_l".into(), acc_type);
    state.add_variable("val_r".into(), acc_type);
    state.add_type(item_type);

    let libraries = state
        .nonterms
        .values()
        .flat_map(|desc| desc.library_functions.iter().map(|x| x.as_str()))
        .collect::<Vec<&str>>()
        .join("\n");

    let variable_decls = variables
        .iter()
        .map(|(name, ty)| format!("(declare-var {} {})", name, ty.format()))
        .collect::<Vec<_>>()
        .join("\n");

    let variable_assumptions = variables
        .iter()
        .map(|(name, ty)| {
            state
                .nonterms
                .get(ty)
                .expect(format!("non-terminal for {} not found", ty).as_str())
                .assumptions
                .iter()
                .map(|a| a.replace("$var", name.as_str()))
                .collect::<Vec<String>>()
                .join("\n")
        })
        .chain(assumptions.0.iter().cloned())
        .collect::<Vec<String>>()
        .join("\n");

    let num_sygus_type = Type::Num.format();

    vec![
        "(set-logic HO_ALL)".to_string(),
        format!(
            "(define-fun min ((x {num_sygus_type}) (y {num_sygus_type})) {num_sygus_type} (ite (<= x y) x y))"
        ),
        format!(
            "(define-fun max ((x {num_sygus_type}) (y {num_sygus_type})) {num_sygus_type} (ite (<= x y) y x))"
        ),
        format!("(declare-var _mn {num_sygus_type})"),
        format!("(declare-var _mx {num_sygus_type})"),
        variable_decls,
        variable_assumptions,
        libraries,
        format_synth_fun(acc_type, ret_type, &mut state, "merge"),
    ]
    .join("\n")
}

fn format_normalizer_postlude(
    init: &str,
    constraint: NormalizerConstraint,
    use_partial_norm: bool,
) -> String {
    let merge_operator = if use_partial_norm { "h" } else { "merge" };

    vec![
        if constraint.intersects(NormalizerConstraint::Inductive) {
            format!("(constraint (= r1 ({merge_operator} r1 {init})))")
        } else {
            format!("")
        }
        .as_str(),
        if constraint.intersects(NormalizerConstraint::Commutative) {
            format!(
                "(constraint (= ({merge_operator} r1 (f r2 x)) (f ({merge_operator} r1 r2) x)))"
            )
        } else {
            format!("")
        }
        .as_str(),
        "(check-synth)",
    ]
    .join("\n")
}

fn format_merge_postlude(init: &str, left_args: &Vec<Symbol>, right_args: &Vec<Symbol>) -> String {
    fn build_foldl_expr(init: &str, args: &Vec<Symbol>) -> String {
        let init = format!("(f {} {})", init, args[0]);
        args.iter()
            .skip(1)
            .fold(init, |acc, arg| format!("(f {} {})", acc, arg))
    }

    let args = left_args.iter().chain(right_args.iter()).cloned().collect();
    assert!(
        left_args.len() > 0 && right_args.len() > 0,
        "expected non-empty arguments"
    );
    let bindings = vec![
        format!("(_y1 {})", build_foldl_expr(init, left_args)),
        format!("(_y2 {})", build_foldl_expr(init, right_args)),
        format!("(_y {})", build_foldl_expr(&init, &args)),
    ]
    .join("\n");

    let constraint = format!(
        r#"
(constraint (let (
{bindings})
    (= (merge _y1 _y2) _y)))
(constraint (let (
{bindings})
    (= (merge _y1 {init}) _y1)))
(constraint (let (
{bindings})
    (= (merge {init} _y2) _y2)))
(constraint (= (merge {init} {init}) {init}))
"#
    );

    vec![constraint, "(check-synth)".to_string()].join("\n")
}

fn run_cvc5(script: &str) -> Result<String, NormalizerSynthesisFailure> {
    use std::io::Write;
    use std::process::Command;
    use tempfile::Builder;

    let mut file = Builder::new()
        .suffix(SYGUS_FILE_EXT)
        .tempfile()
        .map_err(|e| e.to_string())?;
    file.write_all(script.as_bytes())
        .map_err(|e| e.to_string())?;

    let mut child = Command::new(CVC5_PATH)
        .arg(file.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let output = child
        .wait_timeout(Duration::from_secs(CVC5_TIMEOUT_SECS))
        .map_err(|e| e.to_string())?;

    match output {
        None => {
            child.kill().map_err(|e| e.to_string())?;
            Err(NormalizerSynthesisFailure::Timeout)
        }
        Some(code) if code.success() => {
            let mut s = String::new();
            child.stdout.take().unwrap().read_to_string(&mut s).unwrap();
            Ok(s)
        }
        Some(code) => {
            let mut s = String::new();
            child.stdout.take().unwrap().read_to_string(&mut s).unwrap();
            child.stderr.take().unwrap().read_to_string(&mut s).unwrap();
            s.push_str("\n");
            s.push_str(&format!("cvc5 exited with status: {:?}\n", code));
            s.push_str(&format!("script:\n{}", script));
            event!(Level::ERROR, "cvc5 error: {}", s);

            Err(NormalizerSynthesisFailure::Other(s))
        }
    }
}

fn parse_sexp_rec(s: &Sexp) -> Result<Expr, String> {
    match s {
        Sexp::Atom(Atom::S(s)) => Ok(Expr::from(s.as_str())),
        Sexp::Atom(Atom::I(i)) => Ok(Expr::Num(*i as i32)),
        Sexp::Atom(Atom::F(f)) => Ok(Expr::Num(*f as i32)),
        Sexp::List(params) => {
            let args = &params[1..];
            match &params[0] {
                Sexp::Atom(Atom::S(func)) => {
                    if let Some(binop) = parse_binop(func) {
                        assert!(
                            args.len() == 2,
                            "expected two arguments for binary operator"
                        );
                        let arg0 = parse_sexp_rec(&args[0])?;
                        let arg1 = parse_sexp_rec(&args[1])?;
                        return Ok(binop!(binop, arg0, arg1));
                    }

                    match func.as_str() {
                        "tuple" => {
                            let exprs = args
                                .iter()
                                .map(|arg| parse_sexp_rec(arg))
                                .collect::<Result<Vec<_>, _>>()?;
                            Ok(Expr::Tuple(exprs))
                        }

                        "ite" => {
                            assert!(args.len() == 3, "expected three arguments for ite");
                            let cond = parse_sexp_rec(&args[0])?;
                            let then_expr = parse_sexp_rec(&args[1])?;
                            let else_expr = parse_sexp_rec(&args[2])?;
                            Ok(Expr::Ite {
                                cond: Box::new(cond),
                                then_expr: Box::new(then_expr),
                                else_expr: Box::new(else_expr),
                            })
                        }

                        s if matches!(s, "max" | "min") => {
                            assert!(args.len() == 2, "expected two arguments for binary ops");
                            let arg0 = parse_sexp_rec(&args[0])?;
                            let arg1 = parse_sexp_rec(&args[1])?;
                            Ok(Expr::from(s).call(vec![arg0, arg1]))
                        }

                        s if matches!(s, "str.++" | "seq.++") => {
                            assert!(args.len() == 2, "expected two arguments for concatenation");
                            let arg0 = parse_sexp_rec(&args[0])?;
                            let arg1 = parse_sexp_rec(&args[1])?;
                            Ok(binop!(BinOpKinds::Concat, arg0, arg1))
                        }

                        "as" => {
                            assert!(args.len() == 2, "expected two arguments for type cast");
                            parse_sexp_rec(&args[0])
                        }

                        s => {
                            let library_func_name = parse_cvc5_fn_name(s);
                            let exprs = args
                                .iter()
                                .map(|arg| parse_sexp_rec(arg))
                                .collect::<Result<Vec<_>, _>>()?;
                            Ok(Expr::from(library_func_name).call(exprs))
                        }
                    }
                }

                // (_ cvc5-func args)
                Sexp::List(params) if matches!(&params[0], Sexp::Atom(Atom::S(s)) if s == "_") => {
                    let func_args = &params[2..];
                    let func = params[1].to_string();
                    match func.as_str() {
                        "tuple.select" => {
                            assert!(
                                func_args.len() == 1,
                                "expected one argument for tuple.select"
                            );
                            let idx = func_args[0]
                                .to_string()
                                .parse::<usize>()
                                .map_err(|e| e.to_string())?;
                            let tuple = parse_sexp_rec(&args[0])?;
                            Ok(tuple_access!(tuple, idx))
                        }
                        _ => panic!("unsupported cvc5 function: {:?}", params[1]),
                    }
                }

                _ => panic!("unsupported sexp: {:?}", s),
            }
        }
    }
}

fn parse_cvc5_result(s: &str, acc_type: &Type) -> Result<Expr, NormalizerSynthesisFailure> {
    let s = s.trim();
    if s == "infeasible" {
        return Err(NormalizerSynthesisFailure::Refuted);
    }
    if s.contains("(error") {
        return Err(NormalizerSynthesisFailure::Other(s.into()));
    }

    let sexp = parse(s).map_err(|e| e.to_string())?;
    let Sexp::List(funcs) = sexp else {
        return Err("expected a list".into());
    };
    let [Sexp::List(func)] = &funcs[..] else {
        return Err("expected a function".into());
    };

    let [Sexp::Atom(Atom::S(kw)), Sexp::Atom(Atom::S(func_name)), Sexp::List(params), _typ, body] =
        &func[..]
    else {
        return Err("expected a function definition".into());
    };

    assert!(kw == "define-fun", "expected a function definition");
    assert!(
        matches!(func_name.as_str(), "merge" | "cond"),
        "expected a function named merge or cond"
    );
    assert!(params.len() == 2, "expected two parameters");

    fn parse_param_name(s: &Sexp) -> Result<&str, String> {
        match s {
            Sexp::List(args) => match &args[..] {
                [Sexp::Atom(Atom::S(name)), _] => Ok(name.as_str()),
                _ => Err("expected a parameter name".into()),
            },
            _ => Err("expected a parameter name".into()),
        }
    }

    let param_1 = parse_param_name(&params[0])?;
    let param_2 = parse_param_name(&params[1])?;

    let body = parse_sexp_rec(body)?;

    Ok(body.bind_params(
        [param_1, param_2]
            .into_iter()
            .map(|s| (s.into(), acc_type.clone()))
            .collect::<Vec<_>>(),
    ))
}

pub fn synthesize_normalizer(
    func_acc: &Expr,
    init: &Expr,
    free_var: FunctionHasFreeVar,
    constraint: NormalizerConstraint,
    mode: SynthesisMode,
) -> Result<Expr, NormalizerSynthesisFailure> {
    let func_acc = &simplify(func_acc);

    let (type_map, acc_type, ret_type, item_type) = {
        let mut type_map = BasicSubexprTypeMap::new(func_acc, TypeEnv::default());
        // FIXME: hacky way to set type for partial_norm
        if let SynthesisMode::NormalizerFromPartialResult { partial_norm, .. } = &mode {
            let (_, params) = partial_norm.uncurry_lambda();
            for (param_name, typ) in params.iter() {
                type_map.add_var(*param_name, crate::lang::PolyType::Mono((*typ).clone()));
            }
        }
        if let FunctionHasFreeVar::Yes(name, ty) = &free_var {
            type_map.add_var(*name, crate::lang::PolyType::Mono((*ty).clone()));
        }

        let func_acc_type = type_map.get_type(func_acc);
        let (param_types, ret_type) = func_acc_type.uncurry_fn();
        let [acc_type, item_type] = param_types.as_slice() else {
            panic!("expected two parameters: {}", func_acc_type)
        };
        assert!(
            ret_type == *acc_type,
            "expected the return type to match the accumulator type; got {} != {}\n{}",
            ret_type.format(),
            acc_type.format(),
            func_acc
        );
        type_map.add_expr(init, ret_type);
        match (&init, &ret_type) {
            (Expr::Tuple(tups), Type::Tuple(tys)) if tups.len() == tys.len() => {
                for (i, tup) in tups.iter().enumerate() {
                    type_map.add_expr(tup, &tys[i]);
                }
            }
            _ => {}
        }

        let acc_type = (*acc_type).clone();
        let ret_type = match &mode {
            SynthesisMode::Normalizer | SynthesisMode::MergeOperator { .. } => acc_type.clone(),
            SynthesisMode::NormalizerFromPartialResult { ret_type, .. } => ret_type.clone(),
        };

        type_map.add_var(
            sym!(merge),
            tforall!(
                sym!(a),
                PolyType::Mono(tfunc!(tv!(a) => tv!(a) => ret_type.clone()))
            ),
        );

        (type_map, acc_type, ret_type, (*item_type).clone())
    };

    let type_map = &type_map as &dyn TypedEnv;
    let free_var_str = match free_var {
        FunctionHasFreeVar::Yes(name, ty) => format!("(declare-var {} {})", name, ty.format()),
        FunctionHasFreeVar::No => "".to_string(),
    };
    let init_str = &(init, type_map).format();

    let script = match mode {
        SynthesisMode::MergeOperator {
            num_left_args,
            num_right_args,
        } => {
            let left_vars = (0..num_left_args)
                .map(|i| Symbol::from(format!("_l{}", i)))
                .collect::<Vec<_>>();

            let right_vars = (0..num_right_args)
                .map(|i| Symbol::from(format!("_r{}", i)))
                .collect::<Vec<_>>();

            let vars = left_vars
                .iter()
                .chain(right_vars.iter())
                .map(|v| (*v, &item_type))
                .collect::<Vec<(Symbol, &Type)>>();

            let prelude =
                format_prelude(&acc_type, &ret_type, &item_type, &Default::default(), &vars);
            let postlude = format_merge_postlude(&init_str, &left_vars, &right_vars);
            vec![
                prelude,
                free_var_str,
                format_define_fun("f", func_acc, type_map),
                postlude,
            ]
            .join("\n")
        }

        SynthesisMode::Normalizer => {
            let variables = vec![
                (sym!(r1), &acc_type),
                (sym!(r2), &acc_type),
                (sym!(x), &item_type),
            ];
            let assumptions = <super::assumption::NaiveAssumptionInference as crate::syn::assumption::AssumptionInference>::infer(func_acc);
            let assumptions = assumptions.instantiate(vec!["r1", "r2"]);

            let prelude =
                format_prelude(&acc_type, &ret_type, &item_type, &assumptions, &variables);
            let postlude = format_normalizer_postlude(&init_str, constraint, false);
            vec![
                prelude,
                free_var_str,
                format_define_fun("f", func_acc, type_map),
                postlude,
            ]
            .join("\n")
        }

        SynthesisMode::NormalizerFromPartialResult {
            ret_type,
            partial_norm,
        } => {
            let variables = vec![
                (sym!(r1), &acc_type),
                (sym!(r2), &acc_type),
                (sym!(x), &item_type),
            ];
            let assumptions = <super::assumption::NaiveAssumptionInference as crate::syn::assumption::AssumptionInference>::infer(func_acc);
            let assumptions = assumptions.instantiate(vec!["r1", "r2"]);
            let prelude =
                format_prelude(&acc_type, &ret_type, &item_type, &assumptions, &variables);
            let postlude = format_normalizer_postlude(&init_str, constraint, true);
            vec![
                prelude,
                free_var_str,
                format_define_fun("f", func_acc, type_map),
                format_define_fun("h", &partial_norm, type_map),
                postlude,
            ]
            .join("\n")
        }
    };

    event!(
        Level::INFO,
        "synthesizing normalizer for {}\n{}",
        func_acc,
        script
    );

    let time_start = Instant::now();
    let output: String = run_cvc5(&script)?;
    let cvc5_duration = time_start.elapsed();
    event!(
        Level::INFO,
        "cvc5 successfully ran for {:.2} seconds\n{}",
        cvc5_duration.as_secs_f64(),
        output
    );

    parse_cvc5_result(&output, &acc_type)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_sexp_keyword() {
        let input_str = "((define-fun merge ((val_l Int) (val_r Int)) Int (+ val_l val_r)))";

        let expr = parse_cvc5_result(input_str, &Type::Num).expect("parsing failed");
        assert!(expr.to_string() == "λ(val_l: num, val_r: num). (val_l + val_r)")
    }
}
