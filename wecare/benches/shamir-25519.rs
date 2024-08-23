use wecare::vm::blocking;
use criterion::{criterion_group, criterion_main, Criterion};
use wecare::{vm::SchemeKind, vm::Engine};
use std::{io::Write, time::Duration};
use std::{hint::black_box, thread};

fn build_shamir_engines() -> (blocking::Engine, blocking::Engine) {
    let clock = std::time::Instant::now();
    print!("Setting up engines...");
    let _ = std::io::stdout().flush();
    let (e1, e2) = thread::scope(|scope| {
        let e2 = scope.spawn(|| {
            Engine::builder()
                .address("127.0.0.1:1234")
                .participant("127.0.0.1:1235")
                .scheme(SchemeKind::Shamir)
                .single_threaded_runtime()
                .connect_blocking().unwrap()
                .build()

        });
        let e1 = scope.spawn(|| {
            thread::sleep(Duration::from_millis(200));
            Engine::builder()
                .address("127.0.0.1:1235")
                .participant("127.0.0.1:1234")
                .scheme(SchemeKind::Shamir)
                .single_threaded_runtime()
                .connect_blocking().unwrap()
                .build()
        });
        (e1.join().unwrap(), e2.join().unwrap())
    });
    println!(" Complete! (took {:#?})", clock.elapsed());
    (e1, e2)
}

fn criterion_benchmark(c: &mut Criterion) {
    let (mut e1, mut e2) = build_shamir_engines();
    c.bench_function("shamir single", |b| {
        let input1 = vec![7.0];
        let input2 = vec![3.0];
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
        });
    });
    c.bench_function("shamir vec32", |b| {
        let input1 = vec![7.0; 32];
        let input2 = vec![3.0; 32];
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
        });
    });
    c.bench_function("shamir vec64", |b| {
        let input1 = vec![7.0; 64];
        let input2 = vec![3.0; 64];
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
        });
    });
    c.bench_function("shamir vec128", |b| {
        let input1 = vec![7.0; 128];
        let input2 = vec![3.0; 128];
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
        });
    });
    let _ = e1.shutdown();
    let _ = e2.shutdown();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
