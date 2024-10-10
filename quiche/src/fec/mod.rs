//! Library for a [Tetrys](https://datatracker.ietf.org/doc/rfc9407/)-like network protocol

mod decoder;
mod encoder;
mod error;
mod symbol;

pub use decoder::Decoder;
pub use encoder::{Encoder, SymbolKind, ReliabilityLevel, EncodingStats};
pub use error::FecError;

pub use symbol::Symbol;

use std::mem::MaybeUninit;

use moeprlnc::bindings::{
    moepgf, moepgf_init, MOEPGF_ALGORITHM_MOEPGF_ALGORITHM_BEST,
    MOEPGF_MAX_ALIGNMENT, MOEPGF_TYPE_MOEPGF256,
};

use aligned_vec::{AVec, ConstAlign};

/// Contains the config for FEC
#[derive(Debug, Copy, Clone)]
pub struct FecConfig {
    max_payload_length: usize,
    alignment: usize,
}

impl FecConfig {
    /// Constructs a new FEC config
    pub fn new(
        max_payload_length: usize, alignment: usize,
    ) -> Self {
        Self {
            max_payload_length,
            alignment,
        }
    }

    /// Returns the maximum payload length
    pub fn get_max_payload_length(&self) -> usize {
        self.max_payload_length
    }

    /// Returns the payload buffer size (2 bytes longer than max payload length)
    pub fn get_payload_buffer_size(&self) -> usize {
        2 + self.max_payload_length
    }

    /// Returns the alignment of symbols
    pub fn get_alignment(&self) -> usize {
        self.alignment
    }

    /// Returns the aligned payload buffer size
    pub fn get_payload_buffer_size_aligned(&self) -> usize {
        self.get_payload_buffer_size()
            .next_multiple_of(self.get_alignment())
    }
}

/// The implementation of a variant of the tetrys protocol
pub struct Tetrys {
    /// The encoder
    pub encoder: Encoder,
    /// The decoder
    pub decoder: Decoder,
}

impl Tetrys {
    /// Constructs a new Tetrys
    pub fn new(
        max_payload_length: u16
    ) -> Result<Self, FecError> {
        let fec_config = FecConfig::new(
            max_payload_length.into(),
            MOEPGF_MAX_ALIGNMENT as usize,
        );
        Ok(Self {
            encoder: Encoder::new(fec_config.clone())?,
            decoder: Decoder::new(fec_config.clone())?,
        })
    }
}

#[derive(Debug)]
pub(super) struct Gf {
    gf: moepgf,
}

impl Gf {
    pub fn new() -> Result<Self, FecError> {
        Ok(Self {
            gf: {
                let mut uninit: MaybeUninit<moepgf> = MaybeUninit::zeroed();
                let ptr = uninit.as_mut_ptr();
                match unsafe {
                    moepgf_init(
                        ptr,
                        MOEPGF_TYPE_MOEPGF256,
                        MOEPGF_ALGORITHM_MOEPGF_ALGORITHM_BEST,
                    )
                } {
                    0 => unsafe { uninit.assume_init() },
                    _ => Err(FecError::MoepInitError)?,
                }
            },
        })
    }

    pub fn multiply_symbol(
        &self, symbol: &mut dyn Symbol, constant: u8, smallest_sid: u64,
        largest_sid: u64,
    ) {
        let payload = symbol.get_payload_aligned_as_mut_ref();
        unsafe {
            self.gf.mulrc.unwrap()(payload.as_mut_ptr(), constant, payload.len());
        }
        let coefficients = symbol.get_coefficients_mut(smallest_sid, largest_sid);
        unsafe {
            self.gf.mulrc.unwrap()(
                coefficients.as_mut_ptr(),
                constant,
                coefficients.len(),
            );
        }
    }

    pub fn multiply_u8(&self, a: u8, b: u8) -> u8 {
        let mut dest: AVec<u8, ConstAlign<{ MOEPGF_MAX_ALIGNMENT as usize }>> =
            AVec::with_capacity(
                MOEPGF_MAX_ALIGNMENT as usize,
                MOEPGF_MAX_ALIGNMENT as usize,
            );
        dest.push(a);
        dest.resize(MOEPGF_MAX_ALIGNMENT as usize, 0);
        unsafe {
            self.gf.mulrc.unwrap()(
                dest.as_mut_ptr(),
                b,
                MOEPGF_MAX_ALIGNMENT as usize,
            );
        }
        dest[0]
    }

    pub fn inverse(&self, constant: u8) -> u8 {
        unsafe { self.gf.inv.unwrap()(constant) }
    }

    pub fn add_n_times(
        &self, dest_symbol: &mut dyn Symbol, src_symbol: &mut dyn Symbol,
        constant: u8, smallest_sid: u64, largest_sid: u64,
    ) {
        let dest_payload = dest_symbol.get_payload_aligned_as_mut_ref();
        let src_payload = src_symbol.get_payload_aligned_as_ref();
        assert_eq!(dest_payload.len(), src_payload.len());
        unsafe {
            self.gf.maddrc.unwrap()(
                dest_payload.as_mut_ptr(),
                src_payload.as_ptr(),
                constant,
                dest_payload.len(),
            );
        }

        let dest_coefficients =
            dest_symbol.get_coefficients_mut(smallest_sid, largest_sid);
        let src_coefficients =
            src_symbol.get_coefficients(smallest_sid, largest_sid);
        assert_eq!(dest_coefficients.len(), src_coefficients.len());
        unsafe {
            self.gf.maddrc.unwrap()(
                dest_coefficients.as_mut_ptr(),
                src_coefficients.as_ptr(),
                constant,
                dest_coefficients.len(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Once;
    static LOGGER_INIT: Once = Once::new();
    
    fn logger_setup() {
	LOGGER_INIT.call_once(|| { env_logger::try_init().ok(); });
    }

    use self::symbol::RepairSymbol;

    #[allow(unused_imports)]
    use super::*;
    use env_logger;

    use rand::seq::SliceRandom;
    use rand::Rng;
    use rand_core::RngCore;

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

    #[test]
    fn tetrys_two_symbols() {
	logger_setup();
        let mut fec = Tetrys::new(1150).unwrap();
	let mut ss = Vec::new();
        ss.push(gen_random_vec(4, 10));
        ss.push(gen_random_vec(4, 10));

	// first one dropped
	let _sid0 = fec.encoder.add_source_symbol(ss[0].as_slice()).unwrap();
	let sid1 = fec.encoder.add_source_symbol(ss[1].as_slice()).unwrap();
	let _ignored = fec.encoder.generate_repair_symbol().unwrap();
	let rs = fec.encoder.generate_repair_symbol().unwrap();

	fec.decoder.add_source_symbol(sid1, ss[1].as_slice())
	    .unwrap();
	fec.decoder.add_repair_symbol(rs.smallest_symbol_id,
				      rs.largest_symbol_id,
				      rs.seed,
				      &rs.get_payload_aligned_as_ref()[..1152])
	    .unwrap();
	let decoded = fec.decoder.try_decode();
	assert_eq!(decoded[0], ss[0]);
	
	
    }

    #[test]
    fn tetrys_loss_out_of_order() {
	logger_setup();
        let mut rng = rand::thread_rng();

        let count: u32 = 200;
        let loss_rate: f64 = 0.1;
        // init tetrys
        let mut fec = Tetrys::new(1500).unwrap();
	fec.encoder.set_reliability_level(
	    ReliabilityLevel::FixedRedundancyRatio(0.2));
	
        let mut ss = Vec::new();
        for _i in 0..=count {
            ss.push(gen_random_vec(500, 1500));
        }

        let mut out: Vec<SendSymbol> = Vec::new();
        let mut missing_sids = Vec::new();

        for i in 0..count {
            if fec.encoder.should_send_next() == SymbolKind::Repair {
                // push RS to out
                let s =
                    fec.encoder.generate_repair_symbol().unwrap();
                out.push(SendSymbol::Repair(s));
            }
            // push SS to out
            let sid = fec
                .encoder
                .add_source_symbol(ss[i as usize].as_slice())
                .unwrap();

	    // no losses at the very end, we do not have any loss information for recovery...
	    if i + 20 > count || (rng.next_u64() as f64 / u64::MAX as f64) > loss_rate {
		trace!("added sid {sid} length: {}", ss[i as usize].len());
                out.push(SendSymbol::Source((sid, ss[i as usize].clone())));
            } else {
		trace!("dropping sid {sid}");
                missing_sids.push(sid);
            }
        }

        let num_received_source_symbols = out
            .iter()
            .filter(|s| matches!(s, SendSymbol::Source(_)))
            .count();

        let num_received_repair_symbols = out
            .iter()
            .filter(|s| matches!(s, SendSymbol::Repair(_)))
            .count();

        trace!("{} source symbols and {} repair symbols have been delivered, {:?} are missing",
	       num_received_source_symbols,
	       num_received_repair_symbols,
	       missing_sids
	);

        // some bad reordering going on
        out.shuffle(&mut rng);

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

        let ack_sid = fec
            .decoder
            .get_ack();
        assert_eq!(ack_sid, missing_sids[0]);
        fec.encoder.handle_ack(ack_sid);

	trace!("Decoder is missing {:?}", fec.decoder.get_missing_sids());
	
	// TODO: Last symbol could be missing, then this assertion fails, should be prevented somehow
	assert_eq!(
            num_received_source_symbols,
            fec.decoder.num_source_symbols()
        );
        assert_eq!(
            num_received_repair_symbols,
            fec.decoder.num_repair_symbols()
        );
        assert_eq!(missing_sids.len(), fec.decoder.num_missing_sids());

        let decoded = fec.decoder.try_decode();

        for (i, sid) in missing_sids.iter().enumerate() {
            trace!(
                "SS {i} {sid} is {} long, decoded is {}",
                ss[*sid as usize].len(),
                decoded[i].len()
            );
            trace!("SS {:?}", ss[*sid as usize]);
            trace!("Restored {:?}", decoded[i]);
            assert!(ss[*sid as usize] == decoded[i])
        }

        let ack_sid_new = fec
            .decoder
            .get_ack();
        assert_eq!(ack_sid_new, count as u64);

        fec.encoder.handle_ack(ack_sid_new);
        assert_eq!(fec.encoder.buffered_symbols(), 0);
        assert_eq!(fec.decoder.num_missing_sids(), 0);
    }
}
