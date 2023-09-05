use ff::PrimeField;
#[derive(PrimeField)]
#[PrimeFieldModulus = "4294967291"]
#[PrimeFieldGenerator = "2"]
#[PrimeFieldReprEndianness = "little"]
pub struct Element([u64; 1]);

impl From<Element> for u32 {
    fn from(val: Element) -> Self {
        let arr = val.to_repr().0;
        let arr = [arr[0], arr[1], arr[2], arr[3]];
        u32::from_le_bytes(arr)
        // val.0[0] as u32
    }
}

impl From<u32> for Element {
    fn from(val: u32) -> Self {
        let val = val.to_le_bytes();
        // NOTE: Should probably mention that this is vartime.
        Element::from_repr_vartime(ElementRepr([val[0], val[1], val[2], val[3], 0, 0, 0, 0])).unwrap()
    }
}

impl From<Element> for u64 {
    fn from(val: Element) -> Self {
        val.0[0]
    }
}
