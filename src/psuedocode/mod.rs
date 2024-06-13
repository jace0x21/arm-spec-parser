pub mod span;
use span::Span;

use nom::IResult;
use nom::bytes::complete::{tag, take_while};
use nom::character::complete::char;
use nom::branch::alt;
use nom::sequence::{delimited, separated_pair};
use nom::error::Error;

#[derive(Debug, PartialEq)]
pub enum Expr {
    Equal(EqualOperator),
    NotEqual(NotEqualOperator),
    LessThan(LessThanOperator),
    GreaterThan(GreaterThanOperator),
    And(AndOperator),
    Or(OrOperator),
    Identifier(String),
    BinaryConstant(BinaryConstantExpr),
    BinaryPattern(BinaryPatternExpr),
    Call(CallExpr),
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
pub struct BinaryPatternExpr {
    pub value: String,
}

pub fn not_whitespace(c: char) -> bool {
    c != ' '
}

pub fn is_alphanumeric(c: char) -> bool {
    c.is_alphanumeric()
}

pub fn identifier(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (input, value) = take_while(not_whitespace)(code)?;
    Ok((input, Expr::Identifier(value.input.into())))
}

pub fn binary_constant(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (input, constant) = delimited(char('\''), take_while(is_alphanumeric), char('\''))(code)?;
    Ok((input, Expr::BinaryConstant(BinaryConstantExpr { value: constant.input.into() })))
}

pub fn binary_pattern(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (input, constant) = delimited(tag("{'"), take_while(is_alphanumeric), tag("'}"))(code)?;
    Ok((input, Expr::BinaryPattern(BinaryPatternExpr { value: constant.input.into() })))
}

pub fn eq(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (input, (left, right)) = separated_pair(term, tag(" == "), expr)(code)?;
    Ok((input, Expr::Equal(EqualOperator { left: Box::new(left), right: Box::new(right) })))
}

pub fn and(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (input, (left, right)) = separated_pair(term, tag(" && "), expr)(code)?;
    Ok((input, Expr::And(AndOperator { left: Box::new(left), right: Box::new(right) })))
}

pub fn term(code: Span) -> IResult<Span, Expr, Error<Span>> {
    alt((
        binary_pattern,
        binary_constant,
        identifier,
    ))(code)
}

pub fn expr(code: Span) -> IResult<Span, Expr, Error<Span>> {
    alt((
        and,
        eq,
        binary_pattern,
        binary_constant,
        identifier,
    ))(code)
}

mod tests {
    use crate::psuedocode::*;

    #[test]
    pub fn test_eq_statement() {
        let code = Span { input: "Ra == '11111'", min_precedence: 0 };
        let (input, ast) = expr(code).unwrap();
        assert_eq!(ast, Expr::Equal(EqualOperator {
            left: Box::new(Expr::Identifier("Ra".into())),
            right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "11111".into() })),
        }));
    }

    #[test]
    pub fn test_precedence() {
        let code = Span { input: "A == '0' && Rt == '11111'", min_precedence: 0 };
        let (input, ast) = expr(code).unwrap();
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
}
