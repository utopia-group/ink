use super::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum TypeJson {
    Var {
        name: String,
    },
    Num,
    Bool,
    Str,
    List {
        elm: Box<TypeJson>,
    },
    #[serde(rename = "Tuple")]
    Tuple {
        elms: Vec<TypeJson>,
    },
    Map {
        key: Box<TypeJson>,
        value: Box<TypeJson>,
    },
    Set {
        elm: Box<TypeJson>,
    },
    Fn {
        arg: Box<TypeJson>,
        ret: Box<TypeJson>,
    },
}

impl From<TypeJson> for Type {
    fn from(type_json: TypeJson) -> Self {
        match type_json {
            TypeJson::Var { name } => Type::Var(Symbol::from(name)),
            TypeJson::Num => Type::Num,
            TypeJson::Bool => Type::Bool,
            TypeJson::Str => Type::Str,
            TypeJson::List { elm } => Type::List(Box::new((*elm).into())),
            TypeJson::Tuple { elms } => Type::Tuple(elms.into_iter().map(Into::into).collect()),
            TypeJson::Map { key, value } => {
                Type::Map(Box::new((*key).into()), Box::new((*value).into()))
            }
            TypeJson::Set { elm } => Type::Set(Box::new((*elm).into())),
            TypeJson::Fn { arg, ret } => Type::Fn(Box::new((*arg).into()), Box::new((*ret).into())),
        }
    }
}

impl From<&Type> for TypeJson {
    fn from(type_: &Type) -> Self {
        match type_ {
            Type::Var(name) => TypeJson::Var {
                name: name.to_string(),
            },
            Type::Num => TypeJson::Num,
            Type::Bool => TypeJson::Bool,
            Type::Str => TypeJson::Str,
            Type::List(elm) => TypeJson::List {
                elm: Box::new(elm.as_ref().into()),
            },
            Type::Tuple(elms) => TypeJson::Tuple {
                elms: elms.into_iter().map(Into::into).collect(),
            },
            Type::Map(key, value) => TypeJson::Map {
                key: Box::new(key.as_ref().into()),
                value: Box::new(value.as_ref().into()),
            },
            Type::Set(elm) => TypeJson::Set {
                elm: Box::new(elm.as_ref().into()),
            },
            Type::Fn(arg, ret) => TypeJson::Fn {
                arg: Box::new(arg.as_ref().into()),
                ret: Box::new(ret.as_ref().into()),
            },
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ExprJson {
    Num {
        value: i32,
    },
    Bool {
        value: bool,
    },
    Str {
        value: String,
    },
    Var {
        name: String,
    },

    UnaryOp {
        op: String,
        expr: Box<ExprJson>,
    },
    BinOp {
        op: String,
        left: Box<ExprJson>,
        right: Box<ExprJson>,
    },

    Ite {
        cond: Box<ExprJson>,
        then_expr: Box<ExprJson>,
        else_expr: Box<ExprJson>,
    },

    App {
        func: Box<ExprJson>,
        arg: Box<ExprJson>,
    },

    Lambda {
        param: String,
        param_type: TypeJson,
        body: Box<ExprJson>,
    },

    Let {
        name: String,
        expr: Box<ExprJson>,
        body: Box<ExprJson>,
    },

    Nil,
    Cons {
        head: Box<ExprJson>,
        tail: Box<ExprJson>,
    },
    Tuple {
        values: Vec<ExprJson>,
    },
    TupleAccess {
        tuple_expr: Box<ExprJson>,
        index: usize,
    },

    MapAssign {
        map: Box<ExprJson>,
        key: Box<ExprJson>,
        value: Box<ExprJson>,
    },
    MapAccess {
        map: Box<ExprJson>,
        key: Box<ExprJson>,
    },
}

impl From<ExprJson> for Expr {
    fn from(expr_json: ExprJson) -> Self {
        match expr_json {
            ExprJson::Num { value } => Expr::Num(value),
            ExprJson::Bool { value } => Expr::Bool(value),
            ExprJson::Str { value } => Expr::Str(value),
            ExprJson::Var { name } => Expr::Var(Symbol::from(name)),

            ExprJson::UnaryOp { op, expr } => {
                let op_kind = match op.as_str() {
                    "--" => UnaryOpKinds::Neg,
                    "not" => UnaryOpKinds::Not,
                    _ => panic!("Unknown unary operator: {}", op),
                };
                Expr::UnaryOp(op_kind, Box::new((*expr).into()))
            }

            ExprJson::BinOp { op, left, right } => {
                let op_kind = match op.as_str() {
                    "+" => BinOpKinds::Add,
                    "-" => BinOpKinds::Sub,
                    "*" => BinOpKinds::Mul,
                    "/" => BinOpKinds::Div,
                    "++" => BinOpKinds::Concat,
                    ">" => BinOpKinds::Gt,
                    "<" => BinOpKinds::Lt,
                    "=" => BinOpKinds::Eq,
                    "&&" => BinOpKinds::And,
                    "||" => BinOpKinds::Or,
                    _ => panic!("Unknown binary operator: {}", op),
                };
                Expr::BinOp(op_kind, Box::new((*left).into()), Box::new((*right).into()))
            }

            ExprJson::Ite {
                cond,
                then_expr,
                else_expr,
            } => Expr::Ite {
                cond: Box::new((*cond).into()),
                then_expr: Box::new((*then_expr).into()),
                else_expr: Box::new((*else_expr).into()),
            },

            ExprJson::App { func, arg } => Expr::App {
                func: Box::new((*func).into()),
                arg: Box::new((*arg).into()),
            },

            ExprJson::Lambda {
                param,
                param_type,
                body,
            } => Expr::Lambda {
                param: Symbol::from(param),
                param_type: param_type.into(),
                body: Box::new((*body).into()),
            },

            ExprJson::Let { name, expr, body } => Expr::Let {
                name: Symbol::from(name),
                expr: Box::new((*expr).into()),
                body: Box::new((*body).into()),
            },

            ExprJson::Nil => Expr::Nil,
            ExprJson::Cons { head, tail } => Expr::Cons {
                head: Box::new((*head).into()),
                tail: Box::new((*tail).into()),
            },
            ExprJson::Tuple { values } => Expr::Tuple(values.into_iter().map(Into::into).collect()),
            ExprJson::TupleAccess { tuple_expr, index } => {
                Expr::TupleAccess(Box::new((*tuple_expr).into()), index)
            }

            ExprJson::MapAssign { map, key, value } => Expr::MapAssign {
                map: Box::new((*map).into()),
                key: Box::new((*key).into()),
                value: Box::new((*value).into()),
            },
            ExprJson::MapAccess { map, key } => Expr::MapAccess {
                map: Box::new((*map).into()),
                key: Box::new((*key).into()),
            },
        }
    }
}

impl From<&Expr> for ExprJson {
    fn from(expr: &Expr) -> Self {
        match expr {
            Expr::Num(value) => ExprJson::Num { value: *value },
            Expr::Bool(value) => ExprJson::Bool { value: *value },
            Expr::Str(value) => ExprJson::Str {
                value: value.clone(),
            },
            Expr::Var(name) => ExprJson::Var {
                name: name.to_string(),
            },

            Expr::UnaryOp(op, expr) => {
                let op_str = match op {
                    UnaryOpKinds::Neg => "--",
                    UnaryOpKinds::Not => "not",
                };
                ExprJson::UnaryOp {
                    op: op_str.to_string(),
                    expr: Box::new(expr.as_ref().into()),
                }
            }

            Expr::BinOp(op, left, right) => {
                let op_str = match op {
                    BinOpKinds::Add => "+",
                    BinOpKinds::Sub => "-",
                    BinOpKinds::Mul => "*",
                    BinOpKinds::Div => "/",
                    BinOpKinds::Concat => "++",
                    BinOpKinds::Gt => ">",
                    BinOpKinds::Lt => "<",
                    BinOpKinds::Eq => "=",
                    BinOpKinds::And => "&&",
                    BinOpKinds::Or => "||",
                };
                ExprJson::BinOp {
                    op: op_str.to_string(),
                    left: Box::new(left.as_ref().into()),
                    right: Box::new(right.as_ref().into()),
                }
            }

            Expr::Ite {
                cond,
                then_expr,
                else_expr,
            } => ExprJson::Ite {
                cond: Box::new(cond.as_ref().into()),
                then_expr: Box::new(then_expr.as_ref().into()),
                else_expr: Box::new(else_expr.as_ref().into()),
            },

            Expr::App { func, arg } => ExprJson::App {
                func: Box::new(func.as_ref().into()),
                arg: Box::new(arg.as_ref().into()),
            },

            Expr::Lambda {
                param,
                param_type,
                body,
            } => ExprJson::Lambda {
                param: param.to_string(),
                param_type: param_type.into(),
                body: Box::new(body.as_ref().into()),
            },

            Expr::Let { name, expr, body } => ExprJson::Let {
                name: name.to_string(),
                expr: Box::new(expr.as_ref().into()),
                body: Box::new(body.as_ref().into()),
            },

            Expr::Nil => ExprJson::Nil,
            Expr::Cons { head, tail } => ExprJson::Cons {
                head: Box::new(head.as_ref().into()),
                tail: Box::new(tail.as_ref().into()),
            },
            Expr::Tuple(values) => ExprJson::Tuple {
                values: values.iter().map(Into::into).collect(),
            },
            Expr::TupleAccess(tuple_expr, index) => ExprJson::TupleAccess {
                tuple_expr: Box::new(tuple_expr.as_ref().into()),
                index: *index,
            },

            Expr::MapAssign { map, key, value } => ExprJson::MapAssign {
                map: Box::new(map.as_ref().into()),
                key: Box::new(key.as_ref().into()),
                value: Box::new(value.as_ref().into()),
            },
            Expr::MapAccess { map, key } => ExprJson::MapAccess {
                map: Box::new(map.as_ref().into()),
                key: Box::new(key.as_ref().into()),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_load_json_example() {
        let json_str = r#"{
  "type": "Lambda",
  "param": "buffer",
  "param_type": {
    "type": "Tuple",
    "elms": [
      {
        "type": "Str"
      },
      {
        "type": "Num"
      },
      {
        "type": "Num"
      },
      {
        "type": "Num"
      },
      {
        "type": "Num"
      }
    ]
  },
  "body": {
    "type": "Lambda",
    "param": "bus",
    "param_type": {
      "type": "Tuple",
      "elms": [
        {
          "type": "Str"
        },
        {
          "type": "Num"
        }
      ]
    },
    "body": {
      "type": "Let",
      "name": "minSpeed",
      "expr": {
        "type": "Ite",
        "cond": {
          "type": "BinOp",
          "op": "<",
          "left": {
            "type": "TupleAccess",
            "tuple_expr": {
              "type": "Var",
              "name": "bus"
            },
            "index": 1
          },
          "right": {
            "type": "TupleAccess",
            "tuple_expr": {
              "type": "Var",
              "name": "buffer"
            },
            "index": 2
          }
        },
        "then_expr": {
          "type": "TupleAccess",
          "tuple_expr": {
            "type": "Var",
            "name": "bus"
          },
          "index": 1
        },
        "else_expr": {
          "type": "TupleAccess",
          "tuple_expr": {
            "type": "Var",
            "name": "buffer"
          },
          "index": 2
        }
      },
      "body": {
        "type": "Let",
        "name": "maxSpeed",
        "expr": {
          "type": "Ite",
          "cond": {
            "type": "BinOp",
            "op": ">",
            "left": {
              "type": "TupleAccess",
              "tuple_expr": {
                "type": "Var",
                "name": "bus"
              },
              "index": 1
            },
            "right": {
              "type": "TupleAccess",
              "tuple_expr": {
                "type": "Var",
                "name": "buffer"
              },
              "index": 3
            }
          },
          "then_expr": {
            "type": "TupleAccess",
            "tuple_expr": {
              "type": "Var",
              "name": "bus"
            },
            "index": 1
          },
          "else_expr": {
            "type": "TupleAccess",
            "tuple_expr": {
              "type": "Var",
              "name": "buffer"
            },
            "index": 3
          }
        },
        "body": {
          "type": "Tuple",
          "values": [
            {
              "type": "TupleAccess",
              "tuple_expr": {
                "type": "Var",
                "name": "bus"
              },
              "index": 0
            },
            {
              "type": "BinOp",
              "op": "+",
              "left": {
                "type": "TupleAccess",
                "tuple_expr": {
                  "type": "Var",
                  "name": "buffer"
                },
                "index": 1
              },
              "right": {
                "type": "TupleAccess",
                "tuple_expr": {
                  "type": "Var",
                  "name": "bus"
                },
                "index": 1
              }
            },
            {
              "type": "Var",
              "name": "minSpeed"
            },
            {
              "type": "Var",
              "name": "maxSpeed"
            },
            {
              "type": "BinOp",
              "op": "+",
              "left": {
                "type": "TupleAccess",
                "tuple_expr": {
                  "type": "Var",
                  "name": "buffer"
                },
                "index": 4
              },
              "right": {
                "type": "Num",
                "value": 1
              }
            }
          ]
        }
      }
    }
  }
}"#;

        let expr = serde_json::from_str::<ExprJson>(json_str)
            .expect("Failed to parse JSON")
            .into();
        println!("Loaded expression: {}", expr);

        // Verify the structure
        match expr {
            Expr::Lambda {
                param,
                param_type,
                body,
            } => {
                assert_eq!(param.to_string(), "buffer");
                match param_type {
                    Type::Tuple(types) => {
                        assert_eq!(types.len(), 5);
                        assert_eq!(types[0], Type::Str);
                        assert_eq!(types[1], Type::Num);
                        assert_eq!(types[2], Type::Num);
                        assert_eq!(types[3], Type::Num);
                        assert_eq!(types[4], Type::Num);
                    }
                    _ => panic!("Expected tuple type for buffer parameter"),
                }

                // Verify the nested lambda
                match body.as_ref() {
                    Expr::Lambda {
                        param,
                        param_type,
                        body: _,
                    } => {
                        assert_eq!(param.to_string(), "bus");
                        match param_type {
                            Type::Tuple(types) => {
                                assert_eq!(types.len(), 2);
                                assert_eq!(types[0], Type::Str);
                                assert_eq!(types[1], Type::Num);
                            }
                            _ => panic!("Expected tuple type for bus parameter"),
                        }
                    }
                    _ => panic!("Expected nested lambda"),
                }
            }
            _ => panic!("Expected lambda expression"),
        }
    }
}
