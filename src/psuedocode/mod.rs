pub mod span;
use span::Span;

use nom::IResult;
use nom::bytes::complete::{tag, take_while};
use nom::combinator::peek;
use nom::character::complete::char;
use nom::branch::alt;
use nom::multi::separated_list0;
use nom::sequence::{delimited, preceded, separated_pair};
use nom::error::{Error, ErrorKind};

#[derive(Debug, PartialEq)]
pub enum Expr {
    Add(AddOperator),
    Not(NotOperator),
    Equal(EqualOperator),
    NotEqual(NotEqualOperator),
    In(InOperator),
    LessThan(LessThanOperator),
    GreaterThan(GreaterThanOperator),
    LessThanEqual(LessThanEqualOperator),
    GreaterThanEqual(GreaterThanEqualOperator),
    And(AndOperator),
    Or(OrOperator),
    Identifier(String),
    BinaryConstant(BinaryConstantExpr),
    BinaryPattern(BinaryPatternExpr),
    DecimalConstant(DecimalConstantExpr),
    Call(CallExpr),
}

#[derive(Debug, PartialEq)]
pub struct AddOperator {
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct CallExpr {
    pub identifier: Box<Expr>,
    pub arguments: Vec<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct EqualOperator {
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct NotEqualOperator {
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct LessThanOperator {
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct GreaterThanOperator {
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct LessThanEqualOperator {
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct GreaterThanEqualOperator {
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct OrOperator {
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct AndOperator {
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct NotOperator {
    pub operand: Box<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct InOperator {
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Debug, PartialEq)]
pub struct BinaryConstantExpr {
    pub value: String,
}

#[derive(Debug, PartialEq)]
pub struct DecimalConstantExpr {
    pub value: u32,
}

#[derive(Debug, PartialEq)]
pub struct BinaryPatternExpr {
    pub value: String,
}

pub fn not_whitespace(c: char) -> bool {
    c != ' '
}

pub fn is_alphanumeric(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

pub fn is_digit(c: char) -> bool {
    c.is_digit(10)
}

pub fn identifier(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (input, value) = take_while(is_alphanumeric)(code)?;
    Ok((input, Expr::Identifier(value.input.into())))
}

pub fn binary_constant(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (input, constant) = delimited(char('\''), take_while(is_alphanumeric), char('\''))(code)?;
    Ok((input, Expr::BinaryConstant(BinaryConstantExpr { value: constant.input.into() })))
}

pub fn decimal_constant(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (i, value) = take_while(is_digit)(code)?;
    // TODO: Bubble up errors here; Need to implement error conversion or define new error type
    let int_value = match u32::from_str_radix(value.input, 10) {
        Ok(v) => v,
        Err(_e) => return Err(nom::Err::Error(Error::new(i, ErrorKind::Fail))),
    };
    Ok((i, Expr::DecimalConstant(DecimalConstantExpr { value: int_value })))
}

pub fn binary_pattern(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (input, constant) = delimited(tag("{'"), take_while(is_alphanumeric), tag("'}"))(code)?;
    Ok((input, Expr::BinaryPattern(BinaryPatternExpr { value: constant.input.into() })))
}

pub fn call(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (i, (identifier, arguments)) = separated_pair(identifier, tag(" "), call_arguments)(code)?;
    Ok((i, Expr::Call(CallExpr { identifier: Box::new(identifier), arguments })))
}

pub fn call_arguments(code: Span) -> IResult<Span, Vec<Expr>, Error<Span>> {
    delimited(tag("("), separated_list0(tag(", "), identifier), tag(")"))(code)
}

//pub fn eq(code: Span) -> IResult<Span, Expr, Error<Span>> {
//    let (input, (left, right)) = separated_pair(term, tag(" == "), expr)(code)?;
//    Ok((input, Expr::Equal(EqualOperator { left: Box::new(left), right: Box::new(right) })))
//}

//pub fn and(code: Span) -> IResult<Span, Expr, Error<Span>> {
//    let (input, (left, right)) = separated_pair(term, tag(" && "), expr)(code)?;
//    Ok((input, Expr::And(AndOperator { left: Box::new(left), right: Box::new(right) })))
//}

pub fn term(code: Span) -> IResult<Span, Expr, Error<Span>> {
    alt((
        not_term,
        call,
        binary_pattern,
        binary_constant,
        decimal_constant,
        identifier,
    ))(code)
}

pub fn not_term(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (i, expr) = preceded(tag("! "), term)(code)?;
    Ok((i, Expr::Not(NotOperator { operand: Box::new(expr) })))
}

pub fn operator(code: Span) -> IResult<Span, &str, Error<Span>> {
    let (i, span) = alt((
        tag(" == "),
        tag(" != "),
        tag(" && "),
        tag(" IN "),
        tag(" > "),
        tag(" < "),
        tag(" >= "),
        tag(" <= "),
        tag(" + "),
    ))(code)?;
    Ok((i, span.input))
}

pub fn precedence_for(op: &str) -> u32 {
    match op {
        " && " | " || " => 1,
        " == " | " != " | " IN " | " > " | " < " | " >= " | " <= " => 2,
        " + " => 3,
        _ => 0, 
    }
}

pub fn build_expr(op: &str, lhs: Expr, rhs: Expr) -> Expr {
    match op {
        " == " => Expr::Equal(
            EqualOperator { 
                left: Box::new(lhs), 
                right: Box::new(rhs), 
            }
        ),
        " && " => Expr::And(
            AndOperator { 
                left: Box::new(lhs), 
                right: Box::new(rhs), 
            }
        ),
        " != " => Expr::NotEqual(
            NotEqualOperator { 
                left: Box::new(lhs), 
                right: Box::new(rhs), 
            }
        ),
        " IN " => Expr::In(
            InOperator { 
                left: Box::new(lhs), 
                right: Box::new(rhs), 
            }
        ),
        " > " => Expr::GreaterThan(
            GreaterThanOperator { 
                left: Box::new(lhs), 
                right: Box::new(rhs), 
            }
        ),
        " < " => Expr::LessThan(
            LessThanOperator { 
                left: Box::new(lhs), 
                right: Box::new(rhs), 
            }
        ),
        " >= " => Expr::GreaterThanEqual(
            GreaterThanEqualOperator { 
                left: Box::new(lhs), 
                right: Box::new(rhs), 
            }
        ),
        " <= " => Expr::LessThanEqual(
            LessThanEqualOperator { 
                left: Box::new(lhs), 
                right: Box::new(rhs), 
            }
        ),
        " + " => Expr::Add(
            AddOperator { 
                left: Box::new(lhs), 
                right: Box::new(rhs), 
            }
        ),
        _ => Expr::Identifier("ERROR".into()),
    }
}

pub fn expr(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (i, mut lhs) = term(code)?;
    let mut current = i;
    loop {
        let (i, op) = match peek(operator)(current) {
            Ok(res) => res,
            Err(nom::Err::Error(e)) => { current = e.input; break; },
            Err(e) => return Err(e),
        };
        if precedence_for(op) < i.min_precedence {
            current = i;
            break;
        }
        let (mut i, op) = operator(i)?;
        let prev_precedence = i.min_precedence;
        i.min_precedence = precedence_for(op) + 1;
        let (mut i, rhs) = expr(i)?;
        i.min_precedence = prev_precedence;
        lhs = build_expr(op, lhs, rhs); 
        current = i;
    }
    Ok((current, lhs))
}

// TODO: Use custom error type
pub fn parse_expr(code: &str) -> Result<Expr, Error<Span>> {
    let span = Span { input: code, min_precedence: 0 };
    match expr(span) {
        Ok((_i, result)) => Ok(result),
        Err(nom::Err::Error(e)) | Err(nom::Err::Failure(e)) => Err(e),
        Err(_) => Err(Error::new(Span { input: code, min_precedence: 0 }, ErrorKind::Fail)),
    }
}

mod tests {
    use crate::psuedocode::*;

    #[test]
    pub fn test_identifier() {
        let ast = parse_expr("Unconditionally").unwrap();
        assert_eq!(ast, Expr::Identifier("Unconditionally".into()));
    }

    #[test]
    pub fn test_eq_statement() {
        let ast = parse_expr("Ra == '11111'").unwrap();
        assert_eq!(ast, Expr::Equal(EqualOperator {
            left: Box::new(Expr::Identifier("Ra".into())),
            right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "11111".into() })),
        }));
    }

    #[test]
    pub fn test_precedence() {
        let ast = parse_expr("A == '0' && Rt == '11111'").unwrap();
        assert_eq!(ast, Expr::And(AndOperator {
            left: Box::new(Expr::Equal(EqualOperator {
                left: Box::new(Expr::Identifier("A".into())),
                right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "0".into() }))
            })),
            right: Box::new(Expr::Equal(EqualOperator {
                left: Box::new(Expr::Identifier("Rt".into())),
                right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "11111".into() })),
            })),
        }));
    }

    #[test]
    pub fn test_ne_statement() {
        let ast = parse_expr("Rd != '11111'").unwrap();
        assert_eq!(ast, Expr::NotEqual(NotEqualOperator {
            left: Box::new(Expr::Identifier("Rd".into())),
            right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "11111".into() })),
        }));
    }

    #[test] 
    pub fn test_not_term() {
        // AFAIK, not a expression actually seen in alias conditions
        let ast = parse_expr("! Rd").unwrap();
        assert_eq!(ast, Expr::Not(NotOperator {
            operand: Box::new(Expr::Identifier("Rd".into())),
        }));
    }

    #[test]
    pub fn test_call_arguments() {
        let (_i, arguments) = call_arguments(Span {input: "(sf, N, imms, immr)", min_precedence: 0 }).unwrap();
        assert_eq!(arguments, vec![
            Expr::Identifier("sf".into()),
            Expr::Identifier("N".into()),
            Expr::Identifier("imms".into()),
            Expr::Identifier("immr".into()), 
        ]);
    }

    #[test]
    pub fn test_call() {
        let ast = parse_expr("MoveWidePreferred (sf, N, imms, immr)").unwrap();
        assert_eq!(ast, Expr::Call(CallExpr {
            identifier: Box::new(Expr::Identifier("MoveWidePreferred".into())),
            arguments: vec![
                Expr::Identifier("sf".into()),
                Expr::Identifier("N".into()),
                Expr::Identifier("imms".into()),
                Expr::Identifier("immr".into()),
            ],
        }));
    }

    #[test]
    pub fn test_in() {
        let ast = parse_expr("cond IN {'111x'}").unwrap();
        assert_eq!(ast, Expr::In(InOperator {
            left: Box::new(Expr::Identifier("cond".into())),
            right: Box::new(Expr::BinaryPattern(BinaryPatternExpr { value: "111x".into() })),
        }));
    }

    #[test]
    pub fn test_gt() {
        let ast = parse_expr("imms > immr").unwrap();
        assert_eq!(ast, Expr::GreaterThan(GreaterThanOperator {
            left: Box::new(Expr::Identifier("imms".into())),
            right: Box::new(Expr::Identifier("immr".into())),
        }));
    }

    #[test]
    pub fn test_lt() {
        let ast = parse_expr("imms < immr").unwrap();
        assert_eq!(ast, Expr::LessThan(LessThanOperator {
            left: Box::new(Expr::Identifier("imms".into())),
            right: Box::new(Expr::Identifier("immr".into())),
        }));
    }

    #[test]
    pub fn test_gte() {
        let ast = parse_expr("imms >= immr").unwrap();
        assert_eq!(ast, Expr::GreaterThanEqual(GreaterThanEqualOperator {
            left: Box::new(Expr::Identifier("imms".into())),
            right: Box::new(Expr::Identifier("immr".into())),
        }));
    }

    #[test]
    pub fn test_lte() {
        let ast = parse_expr("imms <= immr").unwrap();
        assert_eq!(ast, Expr::LessThanEqual(LessThanEqualOperator {
            left: Box::new(Expr::Identifier("imms".into())),
            right: Box::new(Expr::Identifier("immr".into())),
        }));
    }

    #[test]
    pub fn test_compare_calls() {
        let ast = parse_expr("UInt (imms) < UInt (immr)").unwrap();
        assert_eq!(ast, Expr::LessThan(LessThanOperator {
            left: Box::new(Expr::Call(CallExpr { 
                identifier: Box::new(Expr::Identifier("UInt".into())), 
                arguments: vec![Expr::Identifier("imms".into())], 
            })),
            right: Box::new(Expr::Call(CallExpr {
                identifier: Box::new(Expr::Identifier("UInt".into())), 
                arguments: vec![Expr::Identifier("immr".into())], 
            })),
        }));
    }

    #[test]
    pub fn test_decimal_constant() {
        let ast = parse_expr("1").unwrap();
        assert_eq!(ast, Expr::DecimalConstant(DecimalConstantExpr { value: 1 }))
    }

    #[test]
    pub fn test_add() {
        let ast = parse_expr("1 + 1").unwrap();
        assert_eq!(ast, Expr::Add(AddOperator { 
            left: Box::new(Expr::DecimalConstant(DecimalConstantExpr { value: 1 })), 
            right: Box::new(Expr::DecimalConstant(DecimalConstantExpr { value: 1 })),
        }));
    }

    #[test]
    pub fn test_right_heavy() {
        let ast = parse_expr("Rn == '11111' && ! MoveWidePreferred (sf, N, imms, immr)").unwrap();
        assert_eq!(ast, Expr::And(AndOperator {
            left: Box::new(Expr::Equal(EqualOperator {
                left: Box::new(Expr::Identifier("Rn".into())),
                right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "11111".into() })),
            })),
            right: Box::new(Expr::Not(NotOperator {
                operand: Box::new(Expr::Call(CallExpr {
                    identifier: Box::new(Expr::Identifier("MoveWidePreferred".into())),
                    arguments: vec![
                        Expr::Identifier("sf".into()),
                        Expr::Identifier("N".into()),
                        Expr::Identifier("imms".into()),
                        Expr::Identifier("immr".into()),
                    ],
                })),
            })),
        }));        
    }

    #[test]
    pub fn test_add_comparison() {
        let ast = parse_expr("UInt (imms) + 1 == UInt (immr)").unwrap();
        assert_eq!(ast, Expr::Equal(EqualOperator { 
            left: Box::new(Expr::Add(AddOperator { 
                left: Box::new(Expr::Call(CallExpr { 
                    identifier: Box::new(Expr::Identifier("UInt".into())), 
                    arguments: vec![Expr::Identifier("imms".into())], 
                })),
                right: Box::new(Expr::DecimalConstant(DecimalConstantExpr { value: 1 })),
            })), 
            right: Box::new(Expr::Call(CallExpr { 
                identifier: Box::new(Expr::Identifier("UInt".into())), 
                arguments: vec![Expr::Identifier("immr".into())],
            })), 
        }));
    }

    #[test]
    pub fn test_complex_comparison() {
        let ast = parse_expr("imms != '011111' && UInt (imms) + 1 == UInt (immr)").unwrap();
        assert_eq!(ast, Expr::And(AndOperator {
            left: Box::new(Expr::NotEqual(NotEqualOperator {
                left: Box::new(Expr::Identifier("imms".into())),
                right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "011111".into() })),
            })),
            right: Box::new(Expr::Equal(EqualOperator { 
                left: Box::new(Expr::Add(AddOperator { 
                    left: Box::new(Expr::Call(CallExpr { 
                        identifier: Box::new(Expr::Identifier("UInt".into())), 
                        arguments: vec![Expr::Identifier("imms".into())], 
                    })),
                    right: Box::new(Expr::DecimalConstant(DecimalConstantExpr { value: 1 })),
                })), 
                right: Box::new(Expr::Call(CallExpr { 
                    identifier: Box::new(Expr::Identifier("UInt".into())), 
                    arguments: vec![Expr::Identifier("immr".into())],
                })), 
            })),
        }));
    }

    #[test]
    pub fn test_and_left_associative() {
        let ast = parse_expr("imms == '1' && immr == '1' && S == '1'").unwrap();
        assert_eq!(ast, Expr::And(AndOperator {
            left: Box::new(Expr::And(AndOperator {
                left: Box::new(Expr::Equal(EqualOperator {
                    left: Box::new(Expr::Identifier("imms".into())),
                    right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "1".into() })),
                })),
                right: Box::new(Expr::Equal(EqualOperator {
                    left: Box::new(Expr::Identifier("immr".into())),
                    right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "1".into() })),
                })),
            })),
            right: Box::new(Expr::Equal(EqualOperator {
                left: Box::new(Expr::Identifier("S".into())),
                right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "1".into() })),
            })),
        }));    
    }
}
