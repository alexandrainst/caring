use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::{hint::black_box, thread};
use std::{io::Write, time::Duration};
use wecare::vm::{blocking, FieldKind};
use wecare::{vm::Engine, vm::SchemeKind};

fn build_feldman_engines() -> (blocking::Engine, blocking::Engine) {
    let clock = std::time::Instant::now();
    print!("Setting up engines...");
    let _ = std::io::stdout().flush();
    let (e1, e2) = thread::scope(|scope| {
        let e2 = scope.spawn(|| {
            Engine::builder()
                .address("127.0.0.1:1234")
                .participant("127.0.0.1:1235")
                .scheme(SchemeKind::Feldman)
                .field(FieldKind::Curve25519)
                .single_threaded_runtime()
                .connect_blocking()
                .unwrap()
                .build()
                .unwrap()
        });
        let e1 = scope.spawn(|| {
            thread::sleep(Duration::from_millis(200));
            Engine::builder()
                .address("127.0.0.1:1235")
                .participant("127.0.0.1:1234")
                .scheme(SchemeKind::Feldman)
                .field(FieldKind::Curve25519)
                .single_threaded_runtime()
                .connect_blocking()
                .unwrap()
                .build()
                .unwrap()
        });
        (e1.join().unwrap(), e2.join().unwrap())
    });
    println!(" Complete! (took {:#?})", clock.elapsed());
    (e1, e2)
}

fn criterion_benchmark(c: &mut Criterion) {
    let (mut e1, mut e2) = build_feldman_engines();
    let mut group = c.benchmark_group("feldman-25519");
    for n in [1, 8, 16, 32, 64, 128] {
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            let input1 = vec![7.0; n];
            let input2 = vec![3.0; n];
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

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
