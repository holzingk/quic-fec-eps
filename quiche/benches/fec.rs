use criterion::{
    //black_box,
    criterion_group,
    criterion_main,
    Criterion,
    BenchmarkId,
    Throughput,
    BatchSize,
};
use quiche::fec::{
    Tetrys,
    RepairSymbol,
    Symbol
};
use rand::Rng;
enum SendSymbol {
    Source((u64, Vec<u8>)),
    Repair(RepairSymbol),
}
use std::time::Instant;


fn gen_random_vec(min_len: usize, max_len: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    let len: usize = rng.gen_range(min_len..=max_len);
    let mut res = vec![0_u8; len];
    for e in res.iter_mut() {
        *e = rng.gen();
    }
    res
}

fn prepare_decoder(num: usize, count_loss: usize) -> (Tetrys, Vec<Vec<u8>>){
    assert!(num > count_loss);
    let mut fec = Tetrys::new(1500).unwrap();
    let mut ss = Vec::new();
    let mut missing_sids = Vec::new();
    for _i in 0..num {
	ss.push(gen_random_vec(500, 1500));
    }

    let mut out: Vec<SendSymbol> = Vec::new();
    for (i, s) in ss.iter().enumerate() {
	let sid = fec.encoder.add_source_symbol(s.as_slice()).unwrap();
	if i < count_loss {
	    out.push(SendSymbol::Source((sid, s.clone())));
	} else {
	    missing_sids.push(s.clone());
	}
    }
    for _i in 0..count_loss {
	let rs = fec.encoder.generate_repair_symbol().unwrap();
	out.push(SendSymbol::Repair(rs));
    }

    // put to decoder
    for rcv in out {
        match rcv {
            SendSymbol::Source((sid, s)) => {
                fec.decoder.add_source_symbol(sid, s.as_slice()).unwrap();
            },
            SendSymbol::Repair(rs) => {
                fec.decoder
                    .add_repair_symbol(
                        rs.get_smallest_symbol_id(),
                        rs.get_largest_symbol_id(),
                        rs.get_seed(),
                        &rs.get_payload_aligned_as_ref()[..1502],
                    )
                    .unwrap();
            },
        }
    }
    (fec, missing_sids)
}

fn tetrys_decode(fec: &mut Tetrys, missing_sids: &Vec<Vec<u8>>) {
    let decoded = fec.decoder.try_decode();
    for (missing_ss,  decoded_ss) in missing_sids.iter().zip(decoded.iter()) {
	assert_eq!(missing_ss, decoded_ss);
    }
}

fn prepare_encoder(num: usize) -> Tetrys {
    let mut fec = Tetrys::new(1500).unwrap();
    let mut ss = Vec::new();
    for _i in 0..num {
	ss.push(gen_random_vec(500, 1500));
    }
    for s in ss {
	let _sid = fec.encoder.add_source_symbol(s.as_slice()).unwrap();
    }
    fec
}

fn tetrys_encode_symbols(fec: &mut Tetrys) {
    let _rs = fec.encoder.generate_repair_symbol().unwrap();
}

fn bench_decoder_matrix_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoder_matrix");
    for size in (10_usize..2500_usize).step_by(10) {
	// about 5 percent loss
	let loss = size.div_ceil(20_usize);
	group.throughput(Throughput::Elements(size as u64));
	group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
	    b.iter_batched_ref(|| prepare_decoder(size, loss),
			       |(ref mut fec,  missing_sids)| tetrys_decode(fec, &missing_sids),
			       BatchSize::SmallInput)
	});
    }
    group.finish();
}

fn bench_encoder_matrix_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoder_matrix");
    for size in (10..2500).step_by(10) {
	group.throughput(Throughput::Elements(size as u64));
	group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
	    b.iter_batched_ref(|| prepare_encoder(size),
			       |mut fec| tetrys_encode_symbols(&mut fec),
			       BatchSize::SmallInput)
	});
    }
    group.finish();
}

criterion_group!(benches, bench_decoder_matrix_size, bench_encoder_matrix_size);
criterion_main!(benches);
