pub fn run_for_each<A: Clone, B>(
    values: &[impl AsRef<[A]>],
    mut func: impl FnMut(&[A]) -> B,
) -> Vec<B> {
    // This is ugly and a bit inefficient.
    let n = values[0].as_ref().len();
    let m = values.len();
    let mut output = Vec::with_capacity(n);
    for i in 0..n {
        let mut buf = Vec::with_capacity(m);
        for party in values {
            buf.push(party.as_ref()[i].clone());
        }
        let res: B = func(&buf);
        output.push(res);
    }
    output
}

pub fn transpose<T>(original: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!original.is_empty());
    let mut transposed = (0..original[0].len()).map(|_| vec![]).collect::<Vec<_>>();

    for original_row in original {
        for (item, transposed_row) in original_row.into_iter().zip(&mut transposed) {
            transposed_row.push(item);
        }
    }

    transposed
}
