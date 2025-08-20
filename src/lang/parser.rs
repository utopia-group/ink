use crate::lang::{BinOpKinds, Expr, Type, UnaryOpKinds};
use crate::Symbol;
use chumsky::prelude::*;

pub fn parse_binop(op: &str) -> Option<BinOpKinds> {
    match op {
        "+" => Some(BinOpKinds::Add),
        "-" => Some(BinOpKinds::Sub),
        "*" => Some(BinOpKinds::Mul),
        "/" => Some(BinOpKinds::Div),
        "++" => Some(BinOpKinds::Concat),
        ">" => Some(BinOpKinds::Gt),
        "<" => Some(BinOpKinds::Lt),
        "=" => Some(BinOpKinds::Eq),
        "or" | "||" => Some(BinOpKinds::Or),
        "and" | "&&" => Some(BinOpKinds::And),
        _ => None,
    }
}

pub fn parser() -> impl Parser<'static, &'static str, Expr> {
    recursive(|expr| {
        // Numbers
        let int = text::int(10)
            .map(|s: &str| Expr::Num(s.parse().unwrap()))
            .padded();

        // Booleans
        let boolean = choice((
            just("true").to(Expr::Bool(true)),
            just("false").to(Expr::Bool(false)),
        ))
        .padded();

        // Strings
        let string = just('"')
            .ignore_then(none_of('"').repeated().collect::<String>())
            .then_ignore(just('"'))
            .map(Expr::Str)
            .padded();

        // nil
        let nil = just("nil").to(Expr::Nil).padded();

        // Variables
        let variable = text::ident()
            .map(|s: &str| Expr::Var(Symbol::from(s)))
            .padded();

        // Parenthesized expressions
        let paren_expr = expr
            .clone()
            .delimited_by(just("(").padded(), just(")").padded());

        // If-then-else expressions
        let if_expr = just("if")
            .ignore_then(expr.clone().padded())
            .then_ignore(just("then").padded())
            .then(expr.clone())
            .then_ignore(just("else").padded())
            .then(expr.clone())
            .map(|((cond, then_expr), else_expr)| Expr::Ite {
                cond: Box::new(cond),
                then_expr: Box::new(then_expr),
                else_expr: Box::new(else_expr),
            });

        // Let expressions
        let let_expr = just("let")
            .ignore_then(text::ident().map(Symbol::from).padded())
            .then_ignore(just("=").padded())
            .then(expr.clone())
            .then_ignore(just("in").padded())
            .then(expr.clone())
            .map(|((name, value), body)| Expr::Let {
                name,
                expr: Box::new(value),
                body: Box::new(body),
            });

        // Lambda expressions (basic type support)
        let lambda = choice((just("λ"), just("lambda")))
            .ignore_then(just("("))
            .ignore_then(text::ident().map(Symbol::from))
            .then_ignore(just(":").padded())
            .then(choice((
                just("num").to(Type::Num),
                just("bool").to(Type::Bool),
                just("str").to(Type::Str),
            )))
            .then_ignore(just(")").padded())
            .then_ignore(just(".").padded())
            .then(expr.clone())
            .map(|((param, param_type), body)| Expr::Lambda {
                param,
                param_type,
                body: Box::new(body),
            });

        // Unary operators
        let unary_expr = choice((
            just("--").to(UnaryOpKinds::Neg),
            just("not").to(UnaryOpKinds::Not),
        ))
        .then(expr.clone())
        .map(|(op, expr)| Expr::UnaryOp(op, Box::new(expr)));

        // Basic atoms including new constructs
        let atom = choice((
            int, boolean, string, nil, if_expr, let_expr, lambda, unary_expr, paren_expr, variable,
        ));

        // Simple binary operators
        let expr_with_ops = atom.clone().foldl(
            choice((
                just("+").to(BinOpKinds::Add),
                just("-").to(BinOpKinds::Sub),
                just("*").to(BinOpKinds::Mul),
                just("/").to(BinOpKinds::Div),
                just("++").to(BinOpKinds::Concat),
                just("=").to(BinOpKinds::Eq),
                just(">").to(BinOpKinds::Gt),
                just("<").to(BinOpKinds::Lt),
                choice((just("&&"), just("and"))).to(BinOpKinds::And),
                choice((just("||"), just("or"))).to(BinOpKinds::Or),
            ))
            .padded()
            .then(atom)
            .repeated(),
            |lhs, (op, rhs)| Expr::BinOp(op, Box::new(lhs), Box::new(rhs)),
        );

        expr_with_ops
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_parser_numbers() {
        assert_eq!(parser().parse("123").into_result(), Ok(Expr::Num(123)));
    }

    #[test]
    fn test_simple_parser_booleans() {
        assert_eq!(parser().parse("true").into_result(), Ok(Expr::Bool(true)));
        assert_eq!(parser().parse("false").into_result(), Ok(Expr::Bool(false)));
    }

    #[test]
    fn test_simple_parser_variables() {
        assert_eq!(
            parser().parse("x").into_result(),
            Ok(Expr::Var(Symbol::from("x")))
        );
    }

    #[test]
    fn test_simple_parser_binary_ops() {
        assert_eq!(
            parser().parse("1 + 2").into_result(),
            Ok(Expr::BinOp(
                BinOpKinds::Add,
                Box::new(Expr::Num(1)),
                Box::new(Expr::Num(2))
            ))
        );

        assert_eq!(
            parser().parse("true && false").into_result(),
            Ok(Expr::BinOp(
                BinOpKinds::And,
                Box::new(Expr::Bool(true)),
                Box::new(Expr::Bool(false))
            ))
        );
    }

    #[test]
    fn test_if_then_else() {
        let result = parser().parse("if true then 1 else 2").into_result();
        let expected = Ok(Expr::Ite {
            cond: Box::new(Expr::Bool(true)),
            then_expr: Box::new(Expr::Num(1)),
            else_expr: Box::new(Expr::Num(2)),
        });
        assert_eq!(result, expected);
    }

    #[test]
    fn test_let_expression() {
        let result = parser().parse("let x = 5 in x + 1").into_result();
        let expected = Ok(Expr::Let {
            name: Symbol::from("x"),
            expr: Box::new(Expr::Num(5)),
            body: Box::new(Expr::BinOp(
                BinOpKinds::Add,
                Box::new(Expr::Var(Symbol::from("x"))),
                Box::new(Expr::Num(1)),
            )),
        });
        assert_eq!(result, expected);
    }

    #[test]
    fn test_lambda_expression() {
        let result = parser().parse("λ(x: num). x + 1").into_result();
        let expected = Ok(Expr::Lambda {
            param: Symbol::from("x"),
            param_type: Type::Num,
            body: Box::new(Expr::BinOp(
                BinOpKinds::Add,
                Box::new(Expr::Var(Symbol::from("x"))),
                Box::new(Expr::Num(1)),
            )),
        });
        assert_eq!(result, expected);
    }

    #[test]
    fn test_unary_operators() {
        let result = parser().parse("not true").into_result();
        let expected = Ok(Expr::UnaryOp(UnaryOpKinds::Not, Box::new(Expr::Bool(true))));
        assert_eq!(result, expected);
    }
}
