use criterion::{criterion_group, criterion_main, Criterion};
use std::time;
use std::{fs::File, hint::black_box, io::Seek, thread};
use std::{io::Write, time::Duration};
use wecare::{
    do_preproc,
    vm::{blocking, Engine, FieldKind, SchemeKind},
};

fn precompute(n: usize) -> (File, File) {
    let clock = time::Instant::now();
    print!("\nPrecomputing...");
    let _ = std::io::stdout().flush();
    let ctx1 = tempfile::tempfile().unwrap();
    let ctx2 = tempfile::tempfile().unwrap();
    let mut files = [ctx1, ctx2];
    do_preproc(&mut files, &[n, n], 0, false).unwrap();
    let [mut ctx1, mut ctx2] = files;
    ctx1.rewind().unwrap();
    ctx2.rewind().unwrap();
    println!(" Complete! (took {:#?})", clock.elapsed());
    (ctx1, ctx2)
}

fn build_spdz_engines() -> (blocking::Engine, blocking::Engine) {
    let (ctx1, ctx2) = precompute(10000000);
    let clock = time::Instant::now();
    print!("Setting up engines...");
    let _ = std::io::stdout().flush();
    let (e1, e2) = thread::scope(|scope| {
        let e2 = scope.spawn(|| {
            Engine::builder()
                .address("127.0.0.1:1234")
                .participant("127.0.0.1:1235")
                .preprocessed(ctx1)
                .scheme(SchemeKind::Spdz)
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
                .preprocessed(ctx2)
                .scheme(SchemeKind::Spdz)
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
    let (mut e1, mut e2) = build_spdz_engines();
    c.bench_function("spdz single", |b| {
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
    c.bench_function("spdz vec32", |b| {
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
    c.bench_function("spdz vec64", |b| {
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
    c.bench_function("spdz vec128", |b| {
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
