use criterion::{criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup, BenchmarkId, Criterion};
use wecare::vm::blocking::Engine;
use std::{hint::black_box, thread};


mod spdz25519 {
    include!("./spdz-25519.rs");
}
mod spdz32 {
    include!("./spdz-32.rs");
}
mod shamir25519 {
    include!("./shamir-25519.rs");
}
mod shamir32 {
    include!("./shamir-32.rs");
}
mod feldman25519 {
    include!("./feldman-25519.rs");
}


fn bench<M: Measurement>(g: &mut BenchmarkGroup<'_, M>, label: &'static str, engines: impl Fn() -> (Engine, Engine)) {
    let (mut e1, mut e2) = engines();
    for size in [1, 8, 16, 32, 64] {
        g.bench_with_input(BenchmarkId::new(label, size), &size, |b, n| {
            let input1 = vec![7.0; *n];
            let input2 = vec![3.0; *n];
            b.iter(|| {
                thread::scope(|scope| {
                    let t1 = scope.spawn(|| {
                        black_box(e1.sum(&input1));
                    });
                    let t2 = scope.spawn(|| {
                        black_box(e2.sum(&input2));
                    });
                    t1.join().unwrap();
                    t2.join().unwrap();
                });
            })
        });
    }

    let _ = e1.shutdown();
    let _ = e2.shutdown();
}


fn criterion_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("sum-2");
    bench(&mut g, "spdz-25519", spdz25519::build_spdz_engines);
    bench(&mut g, "spdz-32", spdz32::build_spdz_engines);
    bench(&mut g, "shamir-25519", shamir25519::build_shamir_engines);
    bench(&mut g, "shamir-32", shamir32::build_shamir_engines);
    bench(&mut g, "feldman-25519", feldman25519::build_feldman_engines);
}


criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
