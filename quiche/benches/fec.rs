use criterion::{
    //black_box,
    criterion_group,
    criterion_main,
    Criterion,
    BenchmarkId,
    Throughput,	
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


fn gen_random_vec(min_len: usize, max_len: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    let len: usize = rng.gen_range(min_len..=max_len);
    let mut res = vec![0_u8; len];
    for e in res.iter_mut() {
        *e = rng.gen();
    }
    res
}

fn tetrys_send_symbols(num: usize, count_loss: usize) {
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

    let decoded = fec.decoder.try_decode();
    for (missing_ss,  decoded_ss) in missing_sids.iter().zip(decoded.iter()) {
	assert_eq!(missing_ss, decoded_ss);
    }
}

fn bench_matrix_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoder_matrix");
    for size in (50..250).step_by(50) {
	group.throughput(Throughput::Elements(size as u64));
	group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
	    b.iter(|| tetrys_send_symbols(size, size - 5));
	});
    }
    group.finish();
}

criterion_group!(benches, bench_matrix_size);
criterion_main!(benches);
