use crate::{
    net::Id,
    vm::{ConstRef, Instruction},
};

#[derive(Clone, Copy)]
pub struct ExpRef(u16);

#[derive(Clone)]
pub enum Expression {
    Share(ConstRef),
    SymShare(ConstRef),
    Recv(Id),
    Open(ExpRef),
    MulCon(ExpRef, ConstRef),
    BinOp {
        op: BinOp,
        left: ExpRef,
        right: ExpRef,
    },
}

impl Expression {
    pub fn translate(&self) -> Instruction {
        match self {
            Expression::Share(c) => Instruction::Share(*c),
            Expression::SymShare(_) => todo!(),
            Expression::Recv(id) => Instruction::Recv(*id),
            Expression::Open(_) => Instruction::Recombine,
            Expression::MulCon(_, c) => Instruction::MulCon(*c),
            Expression::BinOp {
                op,
                left: _,
                right: _,
            } => match op {
                BinOp::Add => Instruction::Add,
                BinOp::Sub => Instruction::Sub,
                BinOp::Mul => Instruction::Mul,
            },
        }
    }
}

#[derive(Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
}

#[derive(Clone)]
pub struct Ast {
    // Consider a constant pool?
    expressions: Vec<Expression>,
}

impl Ast {
    pub fn parse_bytecode(insts: &[Instruction]) -> Self {
        let mut expressions = Vec::with_capacity(insts.len());
        let mut stack = Vec::new();
        for (i, opcode) in insts.iter().enumerate() {
            let exp = match opcode {
                Instruction::Share(c) => {
                    stack.push(i);
                    Expression::Share(*c)
                }
                Instruction::SymShare(_) => {
                    todo!("Need to know party size")
                }
                Instruction::Recv(id) => {
                    stack.push(i);
                    Expression::Recv(*id)
                }
                Instruction::MulCon(c) => {
                    let exp_ref = ExpRef(stack.pop().unwrap() as u16);
                    Expression::MulCon(exp_ref, *c)
                }
                Instruction::RecvVec(_) => todo!("Handle vector operations better"),
                Instruction::Recombine => {
                    let exp_ref = ExpRef(stack.pop().unwrap() as u16);
                    Expression::Open(exp_ref)
                }
                Instruction::Add => {
                    let right = ExpRef(stack.pop().unwrap() as u16);
                    let left = ExpRef(stack.pop().unwrap() as u16);
                    let op = BinOp::Add;
                    Expression::BinOp { op, left, right }
                }
                Instruction::Sub => {
                    let right = ExpRef(stack.pop().unwrap() as u16);
                    let left = ExpRef(stack.pop().unwrap() as u16);
                    let op = BinOp::Add;
                    Expression::BinOp { op, left, right }
                }
                Instruction::Mul => {
                    let right = ExpRef(stack.pop().unwrap() as u16);
                    let left = ExpRef(stack.pop().unwrap() as u16);
                    let op = BinOp::Add;
                    Expression::BinOp { op, left, right }
                }
                Instruction::Sum(_) => todo!("Need to know party size"),
            };
            expressions.push(exp);
        }
        Self { expressions }
    }

    fn sort(&mut self) {
        self.expressions.sort_by_key(|exp| match exp {
            Expression::Share(_) => -1,
            Expression::SymShare(_) => -1,
            Expression::Recv(_) => -1,
            Expression::Open(exp) => exp.0 as i32,
            Expression::MulCon(exp, _) => exp.0 as i32,
            Expression::BinOp { op: _, left, right } => {
                let a = left.0 as i32;
                let b = right.0 as i32;
                a.max(b)
            }
        })
    }

    pub fn to_bytecode_fast(&self) -> Vec<Instruction> {
        // Assume epxressions are in order.
        self.traverse_fast().map(|e| e.translate()).collect()
    }

    pub fn to_bytecode_safe(&self) -> Vec<Instruction> {
        self.traverse().map(|e| e.translate()).collect()
    }

    pub fn traverse(&self) -> impl Iterator<Item = &'_ Expression> {
        let mut stack = Vec::new();
        let mut to_visit = Vec::new();
        // Consider what the last expression might be.
        let last = ExpRef((self.expressions.len() - 1) as u16);
        stack.push(last);
        while let Some(exp) = stack.pop() {
            to_visit.push(exp);
            let exp = &self.expressions[exp.0 as usize];
            match exp {
                Expression::Share(_) | Expression::SymShare(_) | Expression::Recv(_) => {}
                Expression::Open(exp) => {
                    stack.push(*exp);
                }
                Expression::MulCon(exp, _) => {
                    stack.push(*exp);
                }
                Expression::BinOp { op: _, left, right } => {
                    // TODO: Order
                    stack.push(*left);
                    stack.push(*right);
                }
            };
        }
        to_visit
            .into_iter()
            .rev()
            .map(|i| &self.expressions[i.0 as usize])
    }

    pub fn traverse_fast(&self) -> impl Iterator<Item = &'_ Expression> {
        // Assume they are in order.
        self.expressions.iter()
    }
}

#[cfg(test)]
mod test {
    use crate::vm::{ast::Ast, Instruction};

    fn check_identity(bytecode: &[Instruction]) {
        let ast = Ast::parse_bytecode(bytecode);
        let translated = ast.to_bytecode_safe();
        assert_eq!(bytecode, translated.as_slice())
    }

    fn check_identity_fast(bytecode: &[Instruction]) {
        let ast = Ast::parse_bytecode(bytecode);
        let translated = ast.to_bytecode_safe();
        assert_eq!(bytecode, translated.as_slice())
    }
}
