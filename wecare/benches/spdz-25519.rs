use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;
use std::{fs::File, hint::black_box, io::Seek, thread};
use wecare::*;

fn precompute(n: usize) -> (File, File) {
    print!("\nPrecomputing...");
    let ctx1 = tempfile::tempfile().unwrap();
    let ctx2 = tempfile::tempfile().unwrap();
    let mut files = [ctx1, ctx2];
    do_preproc(&mut files, vec![n, n], false);
    let [mut ctx1, mut ctx2] = files;
    ctx1.rewind().unwrap();
    ctx2.rewind().unwrap();
    println!(" Complete!");
    (ctx1, ctx2)
}

fn build_spdz_engines() -> (Engine, Engine) {
    let (mut ctx1, mut ctx2) = precompute(2000000);
    print!("\nSetting up engines...");
    let (e1, e2) = thread::scope(|scope| {
        let e2 = scope.spawn(|| {
            Engine::setup("127.0.0.1:1234")
                .add_participant("127.0.0.1:1235")
                .file_to_preprocessed(&mut ctx1)
                .build_spdz()
                .unwrap()
        });
        let e1 = scope.spawn(|| {
            thread::sleep(Duration::from_millis(200));
            Engine::setup("127.0.0.1:1235")
                .add_participant("127.0.0.1:1234")
                .file_to_preprocessed(&mut ctx2)
                .build_spdz()
                .unwrap()
        });
        (e1.join().unwrap(), e2.join().unwrap())
    });
    println!(" Complete!");
    (e1, e2)
}

fn criterion_benchmark(c: &mut Criterion) {
    let (mut e1, mut e2) = build_spdz_engines();
    c.bench_function("spdz single", |b| {
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
    c.bench_function("spdz vec32", |b| {
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
    c.bench_function("spdz vec64", |b| {
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
    c.bench_function("spdz vec128", |b| {
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
