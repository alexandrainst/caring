use std::ops::{Add, Mul, Sub};

use crate::vm::{Instruction, Script};

struct Exp<F> {
    exp: Vec<Instruction<F>>,
}

impl<F> Exp<F> {
    pub fn share(secret: F) -> Self {
        Self {
            exp: vec![Instruction::Share(secret)],
        }
    }

    pub fn finalize(self) -> Script<F> {
        Script(self.exp)
    }
}

impl<F> Add for Exp<F> {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        self.exp.append(&mut rhs.exp);
        self.exp.push(Instruction::Add);
        self
    }
}

impl<F> Mul for Exp<F> {
    type Output = Self;

    fn mul(mut self, mut rhs: Self) -> Self::Output {
        self.exp.append(&mut rhs.exp);
        self.exp.push(Instruction::Mul);
        self
    }
}

impl<F> Mul<F> for Exp<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self::Output {
        self.exp.push(Instruction::MulCon(rhs));
        self
    }
}

impl<F> Sub for Exp<F> {
    type Output = Self;

    fn sub(mut self, mut rhs: Self) -> Self::Output {
        self.exp.append(&mut rhs.exp);
        self.exp.push(Instruction::Sub);
        self
    }
}
