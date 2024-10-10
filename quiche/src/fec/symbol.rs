/// Represents coding symbols such as Source Symbol, Repair Symbols and their interna such as coefficients
use super::error::FecError;

use rand_core::{RngCore, SeedableRng};
use rand_pcg::Pcg64Mcg;

use aligned_vec::{avec, AVec, ConstAlign};
use moeprlnc::bindings::MOEPGF_MAX_ALIGNMENT;
//use std::mem::MaybeUninit;
use super::{FecConfig, Gf};

use std::collections::VecDeque;
use std::convert::TryFrom;
use std::convert::TryInto;

type MoepAVec = AVec<u8, ConstAlign<{ MOEPGF_MAX_ALIGNMENT as usize }>>;

#[derive(Debug)]
pub(super) struct Coefficients {
    coefficients: Vec<u8>,
    smallest_sid: u64,
    largest_sid: u64,
}

impl Coefficients {
    pub fn new_from_source_symbol(sid: u64) -> Self {
        Self {
            coefficients: vec![1],
            smallest_sid: sid,
            largest_sid: sid,
        }
    }

    pub fn new_from_seed(
        seed_offset: u16, smallest_sid: u64, largest_sid: u64,
    ) -> Self {
        let seed = ((seed_offset as u64) << 32) + largest_sid as u64;
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let mut coefficients = vec![0; (largest_sid - smallest_sid + 1) as usize];
	    
        rng.fill_bytes(coefficients.as_mut_slice());
        trace!(
            "Created coefficients vector from seed {seed}: {:?}",
            coefficients
        );
        Self {
            coefficients,
            smallest_sid,
            largest_sid,
        }
    }

    pub fn to_avec(&self, new_smallest_sid: u64, new_largest_sid: u64) -> MoepAVec {
	let len = (new_largest_sid - new_smallest_sid + 1) as usize;
	let len_aligned = len.next_multiple_of(MOEPGF_MAX_ALIGNMENT as usize);
	let mut new: MoepAVec = AVec::with_capacity(MOEPGF_MAX_ALIGNMENT as usize,
						    len_aligned);
	let zeros_front = self.smallest_sid.saturating_sub(new_smallest_sid);
	for _i in 0..zeros_front {
	    new.push(0);
	}

	let skip_front = new_smallest_sid.saturating_sub(self.smallest_sid) as usize;
	let trim_end = self.largest_sid.saturating_sub(new_largest_sid) as usize;
	assert!(trim_end + skip_front < self.coefficients.len());
	let pos_end = self.coefficients.len() - trim_end;
	new.extend_from_slice(&self.coefficients[skip_front..pos_end]);

	new.resize(len_aligned, 0);
	new
    }
}

/// Describes some functionality a symbol (e.g. source or repair) needs to fulfil
pub trait Symbol {
    /// Returns a ref to the actual symbol data
    fn get_payload_as_ref(&self) -> &[u8] {
        let buf = self.get_payload_aligned_as_ref();
        let len = u16::from_be_bytes(buf[..2].try_into().unwrap()) as usize;
        &buf[2..(len + 2)]
    }

    /// Returns the actual symbol data as mut ref
    fn get_payload_as_mut_ref(&mut self) -> &mut [u8] {
        let buf = self.get_payload_aligned_as_mut_ref();
        let len = u16::from_be_bytes(buf[..2].try_into().unwrap()) as usize;
        &mut buf[2..(len + 2)]
    }

    /// Returns a ref to the full payload including first 2 bytes indicating the length but without extra alignment space
    fn get_payload_buffer_as_ref(&self) -> &[u8] {
        let buf = self.get_payload_aligned_as_ref();
        let len = u16::from_be_bytes(buf[..2].try_into().unwrap()) as usize;
        &buf[..(len + 2)]
    }

    /// Returns a mut ref to the full payload including first 2 bytes indicating the length but without extra alignment space
    fn get_payload_buffer_as_mut_ref(&mut self) -> &mut [u8] {
        let buf = self.get_payload_aligned_as_mut_ref();
        let len = u16::from_be_bytes(buf[..2].try_into().unwrap()) as usize;
        &mut buf[..(len + 2)]
    }

    /// Returns a ref to the payload including the extra space for alignment
    fn get_payload_aligned_as_ref(&self) -> &[u8];

    /// Returns a mut to the payload including the extra space for alignment
    fn get_payload_aligned_as_mut_ref(&mut self) -> &mut [u8];

    /// Returns a ref to the coefficients
    fn get_coefficients(&mut self, smallest_sid: u64, largest_sid: u64) -> &[u8];

    /// Returns a mut ref to the coefficients
    fn get_coefficients_mut(&mut self, smallest_sid: u64, largest_sid: u64) -> &mut [u8];
}

#[derive(Debug)]
pub(super) struct SourceSymbol {
    pub(super) source_symbol_id: u64,
    pub(super) payload: MoepAVec,
    pub(super) coefficients: Coefficients,
    pub(super) coefficients_avec: MoepAVec,
}

#[derive(Debug)]
pub struct RepairSymbol {
    pub(super) smallest_symbol_id: u64,
    pub(super) largest_symbol_id: u64,
    pub(super) seed: u16,
    pub(super) payload: MoepAVec,
    pub(super) coefficients: Coefficients,
    pub(super) coefficients_avec: MoepAVec,
}

#[derive(Debug)]
pub(super) struct SymbolRow {
    pub(super) coefficients: MoepAVec,
    pub(super) data: MoepAVec,
}

impl SymbolRow {
    pub fn from_ss(ss: &SourceSymbol, smallest_sid: u64, largest_sid: u64) -> Self {
	Self {
	    coefficients: ss.coefficients.to_avec(smallest_sid, largest_sid),
	    data: ss.get_payload_avec(),
	}
    }

    pub fn from_rs(rs: &RepairSymbol, smallest_sid: u64, largest_sid: u64) -> Self {
	Self {
	    coefficients: rs.coefficients.to_avec(smallest_sid, largest_sid),
	    data: rs.get_payload_avec(),
	}
    }
}

impl Symbol for SymbolRow {
    fn get_payload_aligned_as_ref(&self) -> &[u8] {
        self.data.as_slice()
    }

    fn get_payload_aligned_as_mut_ref(&mut self) -> &mut [u8] {
        self.data.as_mut_slice()
    }

    fn get_coefficients(&mut self, _smallest_sid: u64, _largest_sid: u64) -> &[u8] {
        self.coefficients.as_slice()
    }

    fn get_coefficients_mut( &mut self, _smallest_sid: u64, _largest_sid: u64) -> &mut [u8] {
        self.coefficients.as_mut_slice()
    }
}

impl Symbol for SourceSymbol {
    fn get_payload_aligned_as_ref(&self) -> &[u8] {
        self.payload.as_slice()
    }

    fn get_payload_aligned_as_mut_ref(&mut self) -> &mut [u8] {
        self.payload.as_mut_slice()
    }

    fn get_coefficients(&mut self, smallest_sid: u64, largest_sid: u64) -> &[u8] {
        self.coefficients_avec = self.coefficients.to_avec(smallest_sid, largest_sid);
	self.coefficients_avec
	    .as_slice()
    }

    fn get_coefficients_mut(&mut self, smallest_sid: u64, largest_sid: u64) -> &mut [u8] {
	self.coefficients_avec = self.coefficients.to_avec(smallest_sid, largest_sid);
	self.coefficients_avec
	    .as_mut_slice()
    }
}

impl Symbol for &mut SourceSymbol {
    fn get_payload_aligned_as_ref(&self) -> &[u8] {
        self.payload.as_slice()
    }

    fn get_payload_aligned_as_mut_ref(&mut self) -> &mut [u8] {
        self.payload.as_mut_slice()
    }

    fn get_coefficients(&mut self, smallest_sid: u64, largest_sid: u64) -> &[u8] {
	self.coefficients_avec = self.coefficients.to_avec(smallest_sid, largest_sid);
	self.coefficients_avec
	    .as_slice()
    }

    fn get_coefficients_mut(&mut self, smallest_sid: u64, largest_sid: u64) -> &mut [u8] {
        self.coefficients_avec = self.coefficients.to_avec(smallest_sid, largest_sid);
	self.coefficients_avec
	    .as_mut_slice()
    }
}

impl SourceSymbol {
    pub(crate) fn new(
        source_symbol_id: u64, src: &[u8], fec_config: &FecConfig,
    ) -> Result<Self, FecError> {
        Ok(Self {
            source_symbol_id,
            payload: {
                if src.len() + 2 > fec_config.get_payload_buffer_size() {
                    Err(FecError::PayloadLongerThanMaximum)?;
                }
                let mut data = AVec::with_capacity(
                    MOEPGF_MAX_ALIGNMENT as usize,
                    fec_config.get_payload_buffer_size_aligned(),
                );
                data.extend_from_slice(
                    &u16::try_from(src.len()).unwrap().to_be_bytes(),
                );
                data.extend_from_slice(src);
                data.resize(fec_config.get_payload_buffer_size_aligned(), 0);
                data
            },
            coefficients: Coefficients::new_from_source_symbol(source_symbol_id),
	    coefficients_avec: AVec::new(MOEPGF_MAX_ALIGNMENT as usize),
        })
    }

    pub(crate) fn get_source_symbol_id(&self) -> u64 {
        self.source_symbol_id
    }

    fn get_payload_avec(&self) -> MoepAVec {
	self.payload.clone()
    }
}

impl RepairSymbol {
    /// Called by encoder usually
    pub(super) fn new_from_sliding_window(
        smallest_symbol_id: u64, largest_symbol_id: u64, seed: u16,
        sliding_window: &mut VecDeque<SourceSymbol>, gf: &Gf,
        fec_config: &FecConfig,
    ) -> Self {
        let mut new = Self {
            smallest_symbol_id,
            largest_symbol_id,
            seed,
            payload: avec![ [ {MOEPGF_MAX_ALIGNMENT as usize}  ] | 0; fec_config.get_payload_buffer_size_aligned()],
            coefficients: Coefficients::new_from_seed(
                seed,
                smallest_symbol_id,
                largest_symbol_id,
            ),
	    coefficients_avec: AVec::new(MOEPGF_MAX_ALIGNMENT as usize),
        };

	trace!("Creating new RepairSymbol from sliding window, payload length is {}",
	       fec_config.get_payload_buffer_size_aligned());

	let mut sr = SymbolRow::from_rs(&new, smallest_symbol_id, largest_symbol_id);
	
        // encode
        for (i, source_symbol) in sliding_window.iter_mut().enumerate() {
            let coefficient = sr.coefficients[i];
            gf.add_n_times(
                &mut sr,
                source_symbol,
                coefficient,
                smallest_symbol_id,
                largest_symbol_id,
            );
        }
        new.payload = sr.data;
	//trace!("Created new Repairsymbol {:#?}", new);
	new
    }

    /// Called by decoder usually
    pub(super) fn new_from_encoded(
        smallest_symbol_id: u64, largest_symbol_id: u64, seed: u16,
        fec_config: &FecConfig, src: &[u8],
    ) -> Result<Self, FecError> {
        assert_eq!(src.len(), fec_config.get_payload_buffer_size());
        let mut data = AVec::with_capacity(
            MOEPGF_MAX_ALIGNMENT as usize,
            fec_config.get_payload_buffer_size_aligned(),
        );
        data.extend_from_slice(src);
        data.resize(fec_config.get_payload_buffer_size_aligned(), 0);
        Ok(Self {
            smallest_symbol_id,
            largest_symbol_id,
            seed,
            payload: data,
            coefficients: Coefficients::new_from_seed(
                seed,
                smallest_symbol_id,
                largest_symbol_id,
            ),
	    coefficients_avec: AVec::new(MOEPGF_MAX_ALIGNMENT as usize),
        })
    }

    fn get_payload_avec(&self) -> MoepAVec {
	self.payload.clone()
    }

    pub fn get_smallest_symbol_id(&self) -> u64 {
        self.smallest_symbol_id
    }

    pub fn get_largest_symbol_id(&self) -> u64 {
        self.largest_symbol_id
    }

    pub fn protecting(&self, sid: u64) -> bool {
        sid >= self.get_smallest_symbol_id()
            && sid <= self.get_largest_symbol_id()
    }

    pub fn get_seed(&self) -> u16 {
        self.seed
    }
}

impl Symbol for RepairSymbol {
    fn get_payload_aligned_as_ref(&self) -> &[u8] {
        self.payload.as_slice()
    }

    fn get_payload_aligned_as_mut_ref(&mut self) -> &mut [u8] {
        self.payload.as_mut_slice()
    }

    fn get_coefficients(&mut self, smallest_sid: u64, largest_sid: u64) -> &[u8] {
	self.coefficients_avec = self.coefficients.to_avec(smallest_sid, largest_sid);
	self.coefficients_avec.as_slice()
    }

    fn get_coefficients_mut(&mut self, smallest_sid: u64, largest_sid: u64) -> &mut [u8] {
	self.coefficients_avec = self.coefficients.to_avec(smallest_sid, largest_sid);
	self.coefficients_avec.as_mut_slice()
    }
}

impl Symbol for &mut RepairSymbol {
    fn get_payload_aligned_as_ref(&self) -> &[u8] {
        self.payload.as_slice()
    }

    fn get_payload_aligned_as_mut_ref(&mut self) -> &mut [u8] {
        self.payload.as_mut_slice()
    }

    fn get_coefficients(&mut self, smallest_sid: u64, largest_sid: u64) -> &[u8] {
	self.coefficients_avec = self.coefficients.to_avec(smallest_sid, largest_sid);
	self.coefficients_avec.as_slice()
    }

    fn get_coefficients_mut(&mut self, smallest_sid: u64, largest_sid: u64) -> &mut [u8] {
	self.coefficients_avec = self.coefficients.to_avec(smallest_sid, largest_sid);
	self.coefficients_avec.as_mut_slice()
    }
}
