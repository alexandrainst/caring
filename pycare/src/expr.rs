use pyo3::{exceptions::PyTypeError, prelude::*};
use wecare::vm;

#[pyclass]
pub struct Expr(vm::Expr);

#[pyclass(frozen)]
pub struct Opened(pub(crate) vm::Opened);

#[pyclass(frozen)]
#[derive(Debug, Clone, Copy)]
pub struct Id(pub(crate) vm::Id);

#[allow(non_snake_case, reason = "We are doing python")]
#[pymethods]
impl Expr {
    /// Construct a new share expression
    #[staticmethod]
    fn share(num: &Bound<'_, PyAny>) -> PyResult<Self> {
        let res = if let Ok(num) = num.extract::<f64>() {
            println!("sharing as float");
            let num = vm::Number::Float(num);
            vm::Expr::share(num)
        } else if let Ok(num) = num.extract::<u64>() {
            println!("sharing as int");
            // TODO: Consider signedness
            let num = vm::Number::Integer(num);
            vm::Expr::share(num)
        } else if let Ok(num) = num.extract::<Vec<f64>>() {
            println!("sharing as float");
            let num: Vec<_> = num.into_iter().map(vm::Number::Float).collect();
            vm::Expr::share_vec(num)
        } else if let Ok(num) = num.extract::<Vec<u64>>() {
            println!("sharing as int");
            // TODO: Consider signedness
            let num: Vec<_> = num.into_iter().map(vm::Number::Integer).collect();
            vm::Expr::share_vec(num)
        } else {
            return Err(PyTypeError::new_err("num is not a number"));
        };
        Ok(Self(res))
    }

    #[staticmethod]
    fn symmetric_share(num: &Bound<'_, PyAny>, id: Id, size: usize) -> PyResult<Vec<Expr>> {
        let res = if let Ok(num) = num.extract::<f64>() {
            let num = vm::Number::Float(num);
            vm::Expr::symmetric_share(num)
        } else if let Ok(num) = num.extract::<u64>() {
            // TODO: Consider signedness
            let num = vm::Number::Integer(num);
            vm::Expr::symmetric_share(num)
        } else if let Ok(num) = num.extract::<Vec<f64>>() {
            let num: Vec<_> = num.into_iter().map(vm::Number::Float).collect();
            vm::Expr::symmetric_share_vec(num)
        } else if let Ok(num) = num.extract::<Vec<u64>>() {
            // TODO: Consider signedness
            let num: Vec<_> = num.into_iter().map(vm::Number::Integer).collect();
            vm::Expr::symmetric_share_vec(num)
        } else {
            return Err(PyTypeError::new_err("num is not a number"));
        };

        let res = res.concrete(id.0 .0, size);
        let res = res.into_iter().map(Expr).collect();
        Ok(res)
    }

    /// recv from a given party
    #[staticmethod]
    fn recv(id: &Id) -> Self {
        Self(vm::Expr::receive_input(id.0))
    }

    fn open(&self) -> Opened {
        Opened(self.0.clone().open())
    }

    fn __iadd__(&mut self, other: &Self) {
        let rhs: vm::Expr = other.0.clone();
        self.0 += rhs;
    }

    fn __add__(&self, other: &Self) -> Self {
        let lhs: vm::Expr = self.0.clone();
        let rhs: vm::Expr = other.0.clone();
        Self(lhs + rhs)
    }

    fn __sub__(&self, other: &Self) -> Self {
        let lhs: vm::Expr = self.0.clone();
        let rhs: vm::Expr = other.0.clone();
        Self(lhs - rhs)
    }

    fn __isub__(&mut self, other: &Self) {
        let rhs: vm::Expr = other.0.clone();
        self.0 -= rhs;
    }

    fn __mul__(&self, other: &Self) -> Self {
        let lhs: vm::Expr = self.0.clone();
        let rhs: vm::Expr = other.0.clone();
        Self(lhs * rhs)
    }

    fn __imul__(&mut self, other: &Self) {
        let rhs: vm::Expr = other.0.clone();
        self.0 *= rhs;
    }
}
