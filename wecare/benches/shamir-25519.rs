use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;
use std::{hint::black_box, thread};
use wecare::*;

fn build_shamir_engines() -> (Engine, Engine) {
    print!("\nSetting up engines...");
    let (e1, e2) = thread::scope(|scope| {
        let e2 = scope.spawn(|| {
            Engine::setup("127.0.0.1:1234")
                .add_participant("127.0.0.1:1235")
                .threshold(2)
                .build_shamir()
                .unwrap()
        });
        let e1 = scope.spawn(|| {
            thread::sleep(Duration::from_millis(200));
            Engine::setup("127.0.0.1:1235")
                .add_participant("127.0.0.1:1234")
                .threshold(2)
                .build_shamir()
                .unwrap()
        });
        (e1.join().unwrap(), e2.join().unwrap())
    });
    println!(" Complete!");
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
                    black_box(e1.mpc_sum(&input1));
                });
                let t2 = scope.spawn(|| {
                    black_box(e2.mpc_sum(&input2));
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
                    black_box(e1.mpc_sum(&input1));
                });
                let t2 = scope.spawn(|| {
                    black_box(e2.mpc_sum(&input2));
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
                    black_box(e1.mpc_sum(&input1));
                });
                let t2 = scope.spawn(|| {
                    black_box(e2.mpc_sum(&input2));
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
                    black_box(e1.mpc_sum(&input1));
                });
                let t2 = scope.spawn(|| {
                    black_box(e2.mpc_sum(&input2));
                });
                t1.join().unwrap();
                t2.join().unwrap();
            });
        });
    });
    e1.shutdown();
    e2.shutdown();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
