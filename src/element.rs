//! Concrete mathematical field example.
//! Here we have a prime field that is very close to 2^32.

use ff::PrimeField;

#[derive(PrimeField)]
#[PrimeFieldModulus = "4294967291"]
#[PrimeFieldGenerator = "2"]
#[PrimeFieldReprEndianness = "little"]
pub struct Element32([u64; 1]);

impl From<Element32> for u32 {
    /// Convert a element into u32
    ///
    /// * `val`: Element to convert
    fn from(val: Element32) -> Self {
        let arr = val.to_repr().0;
        let arr = [arr[0], arr[1], arr[2], arr[3]];
        u32::from_le_bytes(arr)
        // val.0[0] as u32
    }
}

impl From<u32> for Element32 {
    /// Create a element element from a u32
    ///
    /// * `val`: Element to convert
    fn from(val: u32) -> Self {
        let val = val.to_le_bytes();
        // NOTE: Should probably mention that this is vartime.
        // TODO: Maybe this fails if the integer is bigger than the modulus?
        Element32::from_repr_vartime(Element32Repr([val[0], val[1], val[2], val[3], 0, 0, 0, 0])).unwrap()
    }
}

impl From<Element32> for u64 {
    fn from(val: Element32) -> Self {
        val.0[0]
    }
}
