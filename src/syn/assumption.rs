use tracing::event;
use tracing::Level;

use crate::lang::LangAnalyzer;

use super::tuple_access;
use super::Assumptions;
use super::Expr;

pub trait AssumptionInference {
    fn infer(expr: &Expr) -> Assumptions;
}

pub struct NaiveAssumptionInference;
impl AssumptionInference for NaiveAssumptionInference {
    fn infer(expr: &Expr) -> Assumptions {
        // stupid inference that checks f s x = s + 1
        let (body, params) = expr.uncurry_lambda();
        let state_name = params[0].0;
        let mut analysis = LangAnalyzer::new(expr);
        let mut assumptions = Assumptions::default();

        if let Expr::Num(n) = body {
            assumptions.0.push(format!("(assume (= $state_var {}))", n));
            return assumptions;
        }

        if analysis.check_eq(body, &(Expr::Var(state_name) + 1.into())) {
            assumptions.0.push("(assume (>= $state_var 0))".to_string());
        } else {
            let Expr::Tuple(els) = body else {
                return Default::default();
            };

            for el in els.iter() {
                for i in 0..10 {
                    let src = tuple_access!(Expr::Var(state_name), i) + 1.into();
                    if analysis.check_eq(el, &src) {
                        assumptions.0.extend(vec![format!(
                            "(assume (>= ((_ tuple.select {i}) $state_var) 0))"
                        )]);
                    }
                }
            }
        }

        event!(
            Level::INFO,
            assumption = assumptions.0.to_vec().join("\n"),
            "found assumption"
        );
        assumptions
    }
}
