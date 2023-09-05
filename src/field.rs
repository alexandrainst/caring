use ff::{Field, PrimeField};
#[derive(PrimeField)]
#[PrimeFieldModulus = "4294967291"]
#[PrimeFieldGenerator = "2"]
#[PrimeFieldReprEndianness = "little"]
pub struct Element([u64; 1]);

impl From<Element> for u64 {
    fn from(val: Element) -> Self {
        val.0[0]
    }
}
