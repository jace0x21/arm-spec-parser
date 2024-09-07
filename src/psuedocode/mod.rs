pub mod span;
use span::Span;

use nom::IResult;
use nom::bytes::complete::{tag, take_while, take_while1};
use nom::combinator::peek;
use nom::character::complete::char;
use nom::branch::alt;
use nom::multi::{separated_list0, separated_list1, many1};
use nom::sequence::{delimited, preceded, separated_pair, terminated, tuple};
use nom::error::{Error, ErrorKind};

#[derive(Clone, Debug, PartialEq)]
pub struct ConditionalStmt {
    blocks: Vec<ConditionalBlock>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Condition {
    If(Expr),
    ElseIf(Expr),
    Else,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConditionalBlock {
    condition: Condition,
    block: Vec<Statement>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Assignment(AssignmentStmt),
    Decl(DeclStmt),
    Cond(ConditionalStmt),
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssignmentStmt {
    pub quantifier: Option<Expr>,
    pub dest_type: Option<Expr>,
    pub dest: Expr,
    pub src: Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeclStmt {
    pub quantifier: Option<Expr>,
    pub var_type: Box<Expr>,
    pub identifier: Box<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Cond(Vec<ConditionalExpr>),
    Binary(BinaryExpr),
    Not(NotOperator),
    Identifier(String),
    BinaryConstant(BinaryConstantExpr),
    BinaryPattern(BinaryPatternExpr),
    DecimalConstant(DecimalConstantExpr),
    Register(RegisterExpr),
    Call(CallExpr),
    MemAccess(MemAccessExpr),
    Type(Type)
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConditionalExpr {
    condition: Box<Condition>,
    expr: Box<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BinaryOperator {
    Add,
    LeftShift,
    RightShift,
    Equal,
    NotEqual,
    In,
    LessThan,
    GreaterThan,
    LessThanEqual,
    GreaterThanEqual,
    LogicalAnd,
    LogicalOr,
}

impl BinaryOperator {
    fn precedence(&self) -> u32 {
        match self {
            BinaryOperator::In => 7,
            BinaryOperator::LeftShift
            | BinaryOperator::RightShift => 4,
            BinaryOperator::Add => 3,
            BinaryOperator::Equal 
            | BinaryOperator::NotEqual
            | BinaryOperator::LessThan
            | BinaryOperator::GreaterThan
            | BinaryOperator::LessThanEqual
            | BinaryOperator::GreaterThanEqual => 2,
            BinaryOperator::LogicalOr 
            | BinaryOperator::LogicalAnd => 1,
        }
    }
}

impl TryFrom<&str> for BinaryOperator {
    type Error = &'static str;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.trim() {
            "IN" => Ok(BinaryOperator::In),
            "==" => Ok(BinaryOperator::Equal),
            "!=" => Ok(BinaryOperator::NotEqual),
            ">" => Ok(BinaryOperator::GreaterThan),
            "<" => Ok(BinaryOperator::LessThan),
            ">=" => Ok(BinaryOperator::GreaterThanEqual),
            "<=" => Ok(BinaryOperator::LessThanEqual),
            "+" => Ok(BinaryOperator::Add),
            "<<" => Ok(BinaryOperator::LeftShift),
            ">>" => Ok(BinaryOperator::RightShift),
            "||" => Ok(BinaryOperator::LogicalOr),
            "&&" => Ok(BinaryOperator::LogicalAnd),
            _ => Err("Invalid operator")
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BinaryExpr {
    pub op: BinaryOperator,
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallExpr {
    pub identifier: Box<Expr>,
    pub arguments: Vec<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemAccessExpr {
    pub address: Box<Expr>,
    pub size: Box<Expr>,
    pub access_descriptor: Box<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NotOperator {
    pub operand: Box<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BinaryConstantExpr {
    pub value: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecimalConstantExpr {
    pub value: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BinaryPatternExpr {
    pub value: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegisterExpr {
    pub identifier: String,
    pub arguments: Vec<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Integer,
    Boolean,
    BitVector(u32),
    Complex(String),
}

pub fn not_whitespace(c: char) -> bool {
    c != ' '
}

pub fn is_whitespace(c: char) -> bool {
    c == ' '
}

pub fn is_alphanumeric(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '<' || c == '>'
}

pub fn is_lexpr_atom(c: char) -> bool {
    is_alphanumeric(c) || c == '(' || c == ')'
}

pub fn is_digit(c: char) -> bool {
    c.is_digit(10)
}

pub fn identifier(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (input, value) = take_while1(is_alphanumeric)(code)?;
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

pub fn register(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (i, (identifier, arguments)) = separated_pair(
        take_while(is_alphanumeric),
        take_while(is_whitespace),
        register_arguments,
    )(code)?;
    Ok((i, Expr::Register(RegisterExpr { 
        identifier: identifier.input.into(),
        arguments,
    })))
}

pub fn register_arguments(code: Span) -> IResult<Span, Vec<Expr>, Error<Span>> {
    delimited(
        tag("["),
        separated_list0(
            tag(", "),
            expr,
        ),
        tag("]")
    )(code)
}

pub fn call(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (i, (identifier, arguments)) = separated_pair(identifier, take_while(is_whitespace), call_arguments)(code)?;
    Ok((i, Expr::Call(CallExpr { identifier: Box::new(identifier), arguments })))
}

pub fn parens(mut code: Span) -> IResult<Span, Expr, Error<Span>> {
    // We are recursing into expr in a different context from our precedence climbing loop.
    // We reset precedence so we can treat the expression in parenthesis as a new one.
    let old_precedence = code.min_precedence;
    code.min_precedence = 0;
    let (mut i, expr) = delimited(tag("("), expr, tag(")"))(code)?;
    i.min_precedence = old_precedence;
    Ok((i, expr))
}

pub fn call_arguments(code: Span) -> IResult<Span, Vec<Expr>, Error<Span>> {
    delimited(tag("("), separated_list0(tag(", "), identifier), tag(")"))(code)
}

pub fn type_atom(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (input, value) = take_while(is_lexpr_atom)(code)?;
    match value.input {
        "integer" => Ok((input, Expr::Type(Type::Integer))),
        "boolean" => Ok((input, Expr::Type(Type::Boolean))),
        _ => {
            match bitvector_atom(value) {
                Ok((_, bv_atom)) => Ok((input, bv_atom)),
                Err(e) => Err(e), 
            }
        }
    }
}

pub fn bitvector_atom(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (input, value) = preceded(
        tag("bits"),
        delimited(
            tag("("), 
            take_while(is_digit),
            tag(")")
        )
    )(code)?;
    // We unwrap here because at this point our input should be guaranteed
    // to be parseable as a u32?
    Ok((input, Expr::Type(Type::BitVector(value.input.parse().unwrap()))))
}

pub fn mem_access(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (i, mut atom_list) = delimited(
        tag("Mem["),
        separated_list1(tag(", "), expr),
        tag("]"),
    )(code)?;
    if atom_list.len() != 3 {
        return Err(nom::Err::Error(Error::new(i, ErrorKind::Fail)));
    }
    let access_descriptor = Box::new(atom_list.pop().unwrap());
    let size = Box::new(atom_list.pop().unwrap());
    let address = Box::new(atom_list.pop().unwrap());
    Ok((i, Expr::MemAccess(MemAccessExpr {
        address,
        size,
        access_descriptor,
    })))
}

pub fn lexpr_atom(code: Span) -> IResult<Span, Expr, Error<Span>> {
    alt((
        mem_access,
        register,
        type_atom,
        identifier,
    ))(code)
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
    preceded(
        take_while(is_whitespace),
        alt((
            parens,
            not_term,
            call,
            binary_pattern,
            binary_constant,
            decimal_constant,
            mem_access,
            register,
            identifier,
        )),
    )(code)
}

pub fn not_term(mut code: Span) -> IResult<Span, Expr, Error<Span>> {
    let old_precedence = code.min_precedence;
    code.min_precedence = 0;
    let (mut i, expr) = alt((
        preceded(tag("! "), expr),
        delimited(tag("!("), expr, tag(")")),
    ))(code)?;
    i.min_precedence = old_precedence;
    Ok((i, Expr::Not(NotOperator { operand: Box::new(expr) })))
}

pub fn binary_operator(code: Span) -> IResult<Span, BinaryOperator, Error<Span>> {
    let (i, span) = alt((
        tag(" == "),
        tag(" != "),
        tag(" && "),
        tag(" || "),
        tag(" IN "),
        tag(" > "),
        tag(" < "),
        tag(" >= "),
        tag(" <= "),
        tag(" + "),
        tag(" << "),
        tag(" >> "),
    ))(code)?;
    // TODO: Remove unwrap? Unsure how safe it is to assume
    // this won't panic
    Ok((i, span.input.try_into().unwrap()))
}

pub fn build_expr(op: BinaryOperator, lhs: Expr, rhs: Expr) -> Expr {
    Expr::Binary(BinaryExpr {
        op,
        left: Box::new(lhs),
        right: Box::new(rhs),
    })
}

pub fn expr(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (i, mut lhs) = term(code)?;
    let mut current = i;
    loop {
        let (i, op) = match peek(binary_operator)(current) {
            Ok(res) => res,
            Err(nom::Err::Error(e)) => { current = e.input; break; },
            Err(e) => return Err(e),
        };
        if op.precedence() < i.min_precedence {
            current = i;
            break;
        }
        let (mut i, op) = binary_operator(i)?;
        let prev_precedence = i.min_precedence;
        i.min_precedence = op.precedence() + 1;
        let (mut i, rhs) = expr(i)?;
        i.min_precedence = prev_precedence;
        lhs = build_expr(op, lhs, rhs); 
        current = i;
    }
    Ok((current, lhs))
}

pub fn if_cond(code: Span) -> IResult<Span, Condition, Error<Span>> {
    let (i, cond_expr) = delimited(
        tag("if "),
        expr,
        tag(" then"),
    )(code)?;
    Ok((i, Condition::If(cond_expr)))
}

pub fn else_if_cond(code: Span) -> IResult<Span, Condition, Error<Span>> {
    let (i, cond_expr) = delimited(
        tag("else if "),
        expr,
        tag(" then"),
    )(code)?;
    Ok((i, Condition::ElseIf(cond_expr)))
}

pub fn else_cond(code: Span) -> IResult<Span, Condition, Error<Span>> {
    let (i, _) = tag("else")(code)?;
    Ok((i, Condition::Else))
}

pub fn condition(code: Span) -> IResult<Span, Condition, Error<Span>> {
    alt((
        if_cond,
        else_if_cond,
        else_cond,
    ))(code)
}

pub fn indented_stmt(code: Span) ->IResult<Span, Statement, Error<Span>> {
    preceded(
        tag("    "),
        stmt,
    )(code)
}

pub fn if_block(code: Span) -> IResult<Span, ConditionalBlock, Error<Span>> {
    let (i, (condition, block)) = separated_pair(
        condition,
        tag("\n"),
        separated_list1(
            tag("\n"),
            indented_stmt,
        ),
    )(code)?;
    Ok((i, ConditionalBlock{ condition, block }))
}

pub fn cf_keyword(code: Span) -> IResult<Span, &str, Error<Span>> {
    let (i, keyword) = alt((
        tag("if "),
        tag(" else if "),
    ))(code)?;
    Ok((i, keyword.input))
}

pub fn inline_if(code: Span) -> IResult<Span, ConditionalExpr, Error<Span>> {
    let (i, (keyword, condition, expr)) = tuple((
        cf_keyword,
        expr,
        preceded(
            tag(" then "),
            term,
        ),
    ))(code)?;
    let conditional_expr = match keyword {
        "if " => ConditionalExpr { 
            condition: Box::new(Condition::If(condition)),
            expr: Box::new(expr),
        },
        " else if " => ConditionalExpr {
            condition: Box::new(Condition::ElseIf(condition)),
            expr: Box::new(expr),
        },
        _ => return Err(nom::Err::Error(Error::new(i, ErrorKind::Fail))),
    };
    Ok((i, conditional_expr))
}

pub fn inline_else(code: Span) -> IResult<Span, ConditionalExpr, Error<Span>> {
    let (i, expr) = preceded(
        tag(" else "),
        term,
    )(code)?;
    let conditional_expr = ConditionalExpr {
        condition: Box::new(Condition::Else),
        expr: Box::new(expr),
    };
    Ok((i, conditional_expr)) 
}

pub fn inline_conditional_expr(code: Span) -> IResult<Span, Expr, Error<Span>> {
    let (i, conditional_expr_list) = many1(alt((
        inline_if,
        inline_else,
    )))(code)?;
    Ok((i, Expr::Cond(conditional_expr_list)))
}

pub fn condition_stmt(code: Span) -> IResult<Span, Statement, Error<Span>> {
    let (i, blocks) = separated_list1(tag("\n"), if_block)(code)?;
    Ok((i, Statement::Cond(ConditionalStmt { blocks })))
}

pub fn decl(code: Span) -> IResult<Span, Statement, Error<Span>> {
    let (i, mut atom_list) = terminated(
        separated_list1(tag(" "), lexpr_atom),
        tag(";"),
    )(code)?;
    if atom_list.len() < 2 {
        return Err(nom::Err::Error(Error::new(i, ErrorKind::Fail)));
    }
    let identifier = Box::new(atom_list.pop().unwrap());
    let var_type = Box::new(atom_list.pop().unwrap());
    let quantifier = atom_list.pop();
    Ok((i, Statement::Decl(DeclStmt {
        quantifier,
        var_type,
        identifier,
    })))
}

pub fn assign(code: Span) -> IResult<Span, Statement, Error<Span>> {
    let (i, (mut lhs_list, src)) = separated_pair(
        separated_list1(tag(" "), lexpr_atom),
        tag(" = "),
        terminated(
            alt((inline_conditional_expr, expr)), 
            tag(";")),
    )(code)?;
    if lhs_list.len() < 1 {
        return Err(nom::Err::Error(Error::new(i, ErrorKind::Fail)));
    }
    let dest = lhs_list.pop().unwrap();
    let dest_type = lhs_list.pop();
    let quantifier = lhs_list.pop();
    Ok((i, Statement::Assignment(AssignmentStmt { quantifier, dest_type, dest, src })))
}

pub fn stmt(code: Span) -> IResult<Span, Statement, Error<Span>> {
    alt((
        condition_stmt,
        assign,
        decl,
    ))(code)
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

pub fn parse_stmt(code: &str) -> Result<Statement, Error<Span>> {
    let span = Span { input: code, min_precedence: 0 };
    match stmt(span) {
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
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::Equal,
            left: Box::new(Expr::Identifier("Ra".into())),
            right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "11111".into() })),
        }));
    }

    #[test]
    pub fn test_precedence() {
        let ast = parse_expr("A == '0' && Rt == '11111'").unwrap();
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::LogicalAnd,
            left: Box::new(Expr::Binary(BinaryExpr {
                op: BinaryOperator::Equal,
                left: Box::new(Expr::Identifier("A".into())),
                right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "0".into() }))
            })),
            right: Box::new(Expr::Binary(BinaryExpr {
                op: BinaryOperator::Equal,
                left: Box::new(Expr::Identifier("Rt".into())),
                right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "11111".into() })),
            })),
        }));
    }

    #[test]
    pub fn test_ne_statement() {
        let ast = parse_expr("Rd != '11111'").unwrap();
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::NotEqual,
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
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::In,
            left: Box::new(Expr::Identifier("cond".into())),
            right: Box::new(Expr::BinaryPattern(BinaryPatternExpr { value: "111x".into() })),
        }));
    }

    #[test]
    pub fn test_gt() {
        let ast = parse_expr("imms > immr").unwrap();
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::GreaterThan,
            left: Box::new(Expr::Identifier("imms".into())),
            right: Box::new(Expr::Identifier("immr".into())),
        }));
    }

    #[test]
    pub fn test_lt() {
        let ast = parse_expr("imms < immr").unwrap();
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::LessThan,
            left: Box::new(Expr::Identifier("imms".into())),
            right: Box::new(Expr::Identifier("immr".into())),
        }));
    }

    #[test]
    pub fn test_gte() {
        let ast = parse_expr("imms >= immr").unwrap();
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::GreaterThanEqual,
            left: Box::new(Expr::Identifier("imms".into())),
            right: Box::new(Expr::Identifier("immr".into())),
        }));
    }

    #[test]
    pub fn test_lte() {
        let ast = parse_expr("imms <= immr").unwrap();
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::LessThanEqual,
            left: Box::new(Expr::Identifier("imms".into())),
            right: Box::new(Expr::Identifier("immr".into())),
        }));
    }

    #[test]
    pub fn test_compare_calls_leading_whitespace() {
        let ast = parse_expr("UInt (imms) < UInt (immr)").unwrap();
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::LessThan,
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
    pub fn test_compare_calls() {
        let ast = parse_expr("UInt(imms) < UInt(immr)").unwrap();
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::LessThan,
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
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::Add,
            left: Box::new(Expr::DecimalConstant(DecimalConstantExpr { value: 1 })), 
            right: Box::new(Expr::DecimalConstant(DecimalConstantExpr { value: 1 })),
        }));
    }

    #[test]
    pub fn test_right_heavy() {
        let ast = parse_expr("Rn == '11111' && ! MoveWidePreferred (sf, N, imms, immr)").unwrap();
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::LogicalAnd,
            left: Box::new(Expr::Binary(BinaryExpr {
                op: BinaryOperator::Equal,
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
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::Equal,
            left: Box::new(Expr::Binary(BinaryExpr {
                op: BinaryOperator::Add,
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
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::LogicalAnd,
            left: Box::new(Expr::Binary(BinaryExpr {
                op: BinaryOperator::NotEqual,
                left: Box::new(Expr::Identifier("imms".into())),
                right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "011111".into() })),
            })),
            right: Box::new(Expr::Binary(BinaryExpr {
                op: BinaryOperator::Equal,
                left: Box::new(Expr::Binary(BinaryExpr {
                    op: BinaryOperator::Add,
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
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::LogicalAnd,
            left: Box::new(Expr::Binary(BinaryExpr {
                op: BinaryOperator::LogicalAnd,
                left: Box::new(Expr::Binary(BinaryExpr {
                    op: BinaryOperator::Equal,
                    left: Box::new(Expr::Identifier("imms".into())),
                    right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "1".into() })),
                })),
                right: Box::new(Expr::Binary(BinaryExpr {
                    op: BinaryOperator::Equal,
                    left: Box::new(Expr::Identifier("immr".into())),
                    right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "1".into() })),
                })),
            })),
            right: Box::new(Expr::Binary(BinaryExpr {
                op: BinaryOperator::Equal,
                left: Box::new(Expr::Identifier("S".into())),
                right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "1".into() })),
            })),
        }));    
    }

    #[test]
    pub fn test_parens() {
        let ast = parse_expr("(Rn == '1')").unwrap();
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::Equal,
            left: Box::new(Expr::Identifier("Rn".into())),
            right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "1".into() }))
        }));
    }

    #[test]
    pub fn test_parens_precedence() {
        let ast = parse_expr("sh == '0' && (Rd == '1' || Rn == '1')").unwrap();
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::LogicalAnd,
            left: Box::new(Expr::Binary(BinaryExpr {
                op: BinaryOperator::Equal,
                left: Box::new(Expr::Identifier("sh".into())), 
                right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "0".into() })),
            })),
            right: Box::new(Expr::Binary(BinaryExpr {
                op: BinaryOperator::LogicalOr,
                left: Box::new(Expr::Binary(BinaryExpr {
                    op: BinaryOperator::Equal,
                    left: Box::new(Expr::Identifier("Rd".into())), 
                    right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "1".into() })),
                })),
                right: Box::new(Expr::Binary(BinaryExpr {
                    op: BinaryOperator::Equal,
                    left: Box::new(Expr::Identifier("Rn".into())), 
                    right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "1".into() })),
                })),
            })),
        }));
    }
    
    #[test]
    pub fn test_mixed_ops_left_associative() {
        let ast = parse_expr("sh == '0' && Rd == '1' || Rn == '1'").unwrap();
        assert_eq!(ast, Expr::Binary(BinaryExpr {
            op: BinaryOperator::LogicalOr,
            left: Box::new(Expr::Binary(BinaryExpr {
                op: BinaryOperator::LogicalAnd,
                left: Box::new(Expr::Binary(BinaryExpr {
                    op: BinaryOperator::Equal,
                    left: Box::new(Expr::Identifier("sh".into())), 
                    right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "0".into() })),
                })),
                right: Box::new(Expr::Binary(BinaryExpr {
                    op: BinaryOperator::Equal,
                    left: Box::new(Expr::Identifier("Rd".into())), 
                    right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "1".into() })),
                })),
            })),
            right: Box::new(Expr::Binary(BinaryExpr {
                op: BinaryOperator::Equal,
                left: Box::new(Expr::Identifier("Rn".into())), 
                right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "1".into() })),
            })),
        }));
    }

    #[test]
    pub fn test_identifier_with_brackets() {
        let ast = parse_expr("opc<1>").unwrap();
        assert_eq!(ast, Expr::Identifier("opc<1>".into()));
    }

    #[test]
    pub fn test_not_with_leading_whitespace() {
        let ast = parse_expr("!( IsZero (imm16))").unwrap();
        assert_eq!(ast, Expr::Not(NotOperator { 
            operand: Box::new(Expr::Call(CallExpr {
                identifier: Box::new(Expr::Identifier("IsZero".into())),
                arguments: vec![Expr::Identifier("imm16".into())],
            })), 
        }))
    }

    #[test]
    pub fn test_not_expr() {
        let ast = parse_expr("!( S == '1')").unwrap();
        assert_eq!(ast, Expr::Not(NotOperator { 
            operand: Box::new(Expr::Binary(BinaryExpr {
                op: BinaryOperator::Equal,
                left: Box::new(Expr::Identifier("S".into())), 
                right: Box::new(Expr::BinaryConstant(BinaryConstantExpr { value: "1".into() })), 
            })), 
        }));
    }

    #[test]
    pub fn test_stmt() {
        let ast = parse_stmt("integer n = UInt(Rn);").unwrap();
        assert_eq!(ast, Statement::Assignment(AssignmentStmt { 
            quantifier: None, 
            dest_type: Some(Expr::Type(Type::Integer)), 
            dest: Expr::Identifier("n".into()), 
            src: Expr::Call(CallExpr { 
                identifier: Box::new(Expr::Identifier("UInt".into())), 
                arguments: vec![Expr::Identifier("Rn".into())],
            }), 
        }));
    }

    #[test]
    pub fn test_stmt_bitvector() {
        let ast = parse_stmt("bits(4) options = op2;").unwrap();
        assert_eq!(ast, Statement::Assignment(AssignmentStmt {
            quantifier: None,
            dest_type: Some(Expr::Type(Type::BitVector(4))),
            dest: Expr::Identifier("options".into()),
            src: Expr::Identifier("op2".into()),
        }));
    }

    #[test]
    pub fn test_sp_register() {
        let ast = parse_stmt("address = SP[];").unwrap();
        assert_eq!(ast, Statement::Assignment(AssignmentStmt {
            quantifier: None,
            dest_type: None,
            dest: Expr::Identifier("address".into()),
            src: Expr::Register(RegisterExpr {
                identifier: "SP".into(),
                arguments: vec![],
            }),
        }));

        let lexpr_ast = parse_stmt("SP[] = address;").unwrap();
        assert_eq!(lexpr_ast, Statement::Assignment(AssignmentStmt {
            quantifier: None,
            dest_type: None,
            dest: Expr::Register(RegisterExpr {
                identifier: "SP".into(),
                arguments: vec![],
            }),
            src: Expr::Identifier("address".into()),
        }));
    }

    #[test]
    pub fn test_gp_register() {
        let ast = parse_stmt("address = X[n, 64];").unwrap();
        assert_eq!(ast, Statement::Assignment(AssignmentStmt {
            quantifier: None,
            dest_type: None,
            dest: Expr::Identifier("address".into()),
            src: Expr::Register(RegisterExpr {
                identifier: "X".into(),
                arguments: vec![
                    Expr::Identifier("n".into()),
                    Expr::DecimalConstant(DecimalConstantExpr {
                        value: 64
                    }),
                ],
            }),
        }));

        let lexpr_ast = parse_stmt("X[s, 32] = n;").unwrap();
        assert_eq!(lexpr_ast, Statement::Assignment(AssignmentStmt {
            quantifier: None,
            dest_type: None,
            dest: Expr::Register(RegisterExpr {
                identifier: "X".into(),
                arguments: vec![
                    Expr::Identifier("s".into()),
                    Expr::DecimalConstant(DecimalConstantExpr {
                        value: 32,
                    }),
                ],
            }),
            src: Expr::Identifier("n".into()),
        }));
    }

    #[test]
    pub fn test_shift_left() {
        let ast = parse_stmt("constant integer datasize = 8 << scale;").unwrap();
        assert_eq!(ast, Statement::Assignment(AssignmentStmt {
            quantifier: Some(Expr::Identifier("constant".into())),
            dest_type: Some(Expr::Type(Type::Integer)),
            dest: Expr::Identifier("datasize".into()),
            src: Expr::Binary(BinaryExpr {
                op: BinaryOperator::LeftShift,
                left: Box::new(Expr::DecimalConstant(DecimalConstantExpr {
                    value: 8,
                })),
                right: Box::new(Expr::Identifier("scale".into())),
            }),
        }));
    }

    #[test]
    pub fn test_decl() {
        let ast = parse_stmt("bits(64) address;").unwrap();
        assert_eq!(ast, Statement::Decl(DeclStmt {
            quantifier: None,
            var_type: Box::new(Expr::Type(Type::BitVector(64))),
            identifier: Box::new(Expr::Identifier("address".into())),
        }));
    }

    #[test]
    pub fn test_mem_access() {
        let ast = parse_stmt("data = Mem[address, 2, accdesc];").unwrap();
        assert_eq!(ast, Statement::Assignment(AssignmentStmt {
            quantifier: None,
            dest_type: None,
            dest: Expr::Identifier("data".into()),
            src: Expr::MemAccess(MemAccessExpr {
                address: Box::new(Expr::Identifier("address".into())),
                size: Box::new(Expr::DecimalConstant(DecimalConstantExpr {
                    value: 2
                })),
                access_descriptor: Box::new(Expr::Identifier("accdesc".into())),
            }),
        }));
    }

    #[test] 
    pub fn test_if_block() {
        let ast = parse_stmt("if n == 31 then\n    \
            address = SP[];\n"
        ).unwrap();
        assert_eq!(ast, Statement::Cond(ConditionalStmt {
            blocks: vec![ConditionalBlock {
                condition: Condition::If(Expr::Binary(BinaryExpr {
                    op: BinaryOperator::Equal,
                    left: Box::new(Expr::Identifier("n".into())),
                    right: Box::new(Expr::DecimalConstant(DecimalConstantExpr {
                        value: 31,
                    })),
                })),
                block: vec![
                    Statement::Assignment(AssignmentStmt {
                        quantifier: None,
                        dest_type: None,
                        dest: Expr::Identifier("address".into()),
                        src: Expr::Register(RegisterExpr {
                            identifier: "SP".into(),
                            arguments: vec![],
                        }),
                    })
                ],
            }],
        }))
    }

    #[test] 
    pub fn test_else_if_block() {
        let ast = parse_stmt("if n == 31 then\n    \
            address = SP[];\n\
            else if n == 30 then\n    \
            address = SP[];\n"
        ).unwrap();
        assert_eq!(ast, Statement::Cond(ConditionalStmt {
            blocks: vec![
                ConditionalBlock {
                    condition: Condition::If(Expr::Binary(BinaryExpr {
                        op: BinaryOperator::Equal,
                        left: Box::new(Expr::Identifier("n".into())),
                        right: Box::new(Expr::DecimalConstant(DecimalConstantExpr {
                            value: 31
                        })),
                    })),
                    block: vec![
                        Statement::Assignment(AssignmentStmt {
                            quantifier: None,
                            dest_type: None,
                            dest: Expr::Identifier("address".into()),
                            src: Expr::Register(RegisterExpr {
                                identifier: "SP".into(),
                                arguments: vec![],
                            }),
                        }),
                    ],
                },
                ConditionalBlock {
                    condition: Condition::ElseIf(Expr::Binary(BinaryExpr {
                        op: BinaryOperator::Equal,
                        left: Box::new(Expr::Identifier("n".into())),
                        right: Box::new(Expr::DecimalConstant(DecimalConstantExpr {
                            value: 30,
                        })),
                    })),
                    block: vec![
                        Statement::Assignment(AssignmentStmt {
                            quantifier: None,
                            dest_type: None,
                            dest: Expr::Identifier("address".into()),
                            src: Expr::Register(RegisterExpr {
                                identifier: "SP".into(),
                                arguments: vec![],
                            }),
                        }),
                    ]
                }
            ]
        }));
    }

    #[test]
    pub fn test_else_block() {
        let ast = parse_stmt("if n == 31 then\n    \
            address = SP[];\n\
            else\n    \
            address = 0;\n"
        ).unwrap();
        assert_eq!(ast, Statement::Cond(ConditionalStmt {
            blocks: vec![
                ConditionalBlock {
                    condition: Condition::If(Expr::Binary(BinaryExpr {
                        op: BinaryOperator::Equal,
                        left: Box::new(Expr::Identifier("n".into())),
                        right: Box::new(Expr::DecimalConstant(DecimalConstantExpr {
                            value: 31,
                        })),
                    })),
                    block: vec![
                        Statement::Assignment(AssignmentStmt {
                            quantifier: None,
                            dest_type: None,
                            dest: Expr::Identifier("address".into()),
                            src: Expr::Register(RegisterExpr {
                                identifier: "SP".into(),
                                arguments: vec![],
                            }),
                        }),
                    ]
                },
                ConditionalBlock {
                    condition: Condition::Else,
                    block: vec![
                        Statement::Assignment(AssignmentStmt {
                            quantifier: None,
                            dest_type: None,
                            dest: Expr::Identifier("address".into()),
                            src: Expr::DecimalConstant(DecimalConstantExpr {
                                value: 0,
                            }),
                        }),
                    ]
                },
            ]
        }));
    }

    #[test]
    pub fn test_inline_conditional() {
        let ast = parse_stmt("constant integer esize = \
            if immh IN {'1xxx'} then 64 \
            else if immh IN {'01xx'} then 32 \
            else 16;"
        ).unwrap();
        assert_eq!(ast, Statement::Assignment(AssignmentStmt {
            quantifier: Some(Expr::Identifier("constant".into())),
            dest_type: Some(Expr::Type(Type::Integer)),
            dest: Expr::Identifier("esize".into()),
            src: Expr::Cond(vec![
                ConditionalExpr {
                    condition: Box::new(Condition::If(Expr::Binary(BinaryExpr {
                        op: BinaryOperator::In,
                        left: Box::new(Expr::Identifier("immh".into())),
                        right: Box::new(Expr::BinaryPattern(BinaryPatternExpr {
                            value: "1xxx".into() 
                        })),
                    }))),
                    expr: Box::new(Expr::DecimalConstant(DecimalConstantExpr {
                        value: 64
                    })),
                },
                ConditionalExpr {
                    condition: Box::new(Condition::ElseIf(Expr::Binary(BinaryExpr {
                        op: BinaryOperator::In,
                        left: Box::new(Expr::Identifier("immh".into())),
                        right: Box::new(Expr::BinaryPattern(BinaryPatternExpr {
                            value: "01xx".into()
                        })),
                    }))),
                    expr: Box::new(Expr::DecimalConstant(DecimalConstantExpr {
                        value: 32
                    })),
                },
                ConditionalExpr {
                    condition: Box::new(Condition::Else),
                    expr: Box::new(Expr::DecimalConstant(DecimalConstantExpr {
                        value: 16
                    })),
                },
            ])
        }));
    }
}
