use super::error::FecError;
use super::symbol::{RepairSymbol, SourceSymbol};
use super::{FecConfig, Gf};
use std::collections::{VecDeque, BTreeMap};

/// The kind of the symbol
#[derive(Debug, PartialEq)]
pub enum SymbolKind {
    /// Source Symbol
    Source,
    /// Repair Symbol
    Repair,
}

/// The desired reliability level
#[derive(Debug, PartialEq)]
pub enum ReliabilityLevel {
    /// Only send repair information in case loss was detected
    RecoveryOnly,
    /// Only send so much repair information so that burst losses of a certain count of source symbols can be restored
    BurstLossTolerance(u64),
    /// Fixed ratio of the in flight symbols are protected
    FixedRedundancyRatio(f64),
}

/// Contains stats about the encoding
#[derive(Debug)]
pub struct EncodingStats {
    /// Count of in flight repair symbols
    pub in_flight_rs: u64,
    /// Count of in flight source symbols
    pub in_flight_ss: u64,
    /// Count of in flight symbols
    pub in_flight: u64,
    /// Current ratio of protected symbols
    pub protection_ratio: f64,
    /// recovered
    pub recovered: u64,
}

/// The Encoder runs on the sending side of the protocol.
/// It adds redundant packets -- called coded packets -- that help the [Decoder] recover all packets even if the network is lossy.
pub struct Encoder {
    // the coding configuration
    config: FecConfig,
    // the galois field libarary
    gf: Gf,
    // id to use for the next packet
    next_id: u64,
     // all packets that the receiver has not yet acknowledged
    sliding_window: VecDeque<SourceSymbol>,
    // pair of packet id and associated seed offset for the next coded
    seed_offset: (u64, u16),
    // the reliability level
    reliability_level: ReliabilityLevel,
    // count of currently lost symbols (information comes from pkt ACKs)
    lost: u64,
    // SIDs in flight
    //in_flight_ss: BTreeSet<u64>,
    // rids in flight, highest protected sid, count
    in_flight_rs: BTreeMap<u64, u64>,
    // count of recovered symbols
    recovered: u64,
    //    rsrc: StaticRepairSymbolRateController,
    app_limited: bool,
    left_for_tail_protection: u64,
    incremental: bool,
}

impl Encoder {

    /// Informs the encoder about a detected lost repair symbol
    pub fn on_detected_loss(&mut self) {
	trace!("Encoder has been notified about loss");
	self.lost += 1;
    }

    /// Returns statistics about the state of the encoder
    pub fn get_stats(&self) -> EncodingStats {
	let in_flight_rs = self.in_flight_rs.iter()
	    .fold(0, |acc, (_sid, &count)| acc + count);
	let in_flight_ss = self.sliding_window.len() as u64;
	let in_flight = in_flight_rs + in_flight_ss;
	EncodingStats {
	    in_flight_rs,
	    in_flight_ss,
	    in_flight,
	    protection_ratio: in_flight_rs as f64 / in_flight as f64,
	    recovered: self.recovered,
	}
    }

    /// Informs that the stream as no data left to send, might be the begin of a new app limited phase
    pub fn notify_flushed(&mut self) {
	if self.app_limited == false {
	    self.app_limited = true;

	    let stats = self.get_stats();

	    self.left_for_tail_protection = match self.reliability_level {
		ReliabilityLevel::RecoveryOnly => { 0 },
		ReliabilityLevel::BurstLossTolerance(b) => {
		    b
		},
		ReliabilityLevel::FixedRedundancyRatio(p) => {
		    (p * stats.in_flight as f64) as u64
		}
	    };
	}
    }

    /// Returns the kind of symbol that should be sent in the next packet
    pub fn should_send_next(&self) -> SymbolKind {
	// recovery has highest priority
	if self.lost > 0 {
	    return SymbolKind::Repair;
	}
	
	let stats = self.get_stats();

	// we cannot generate repair symbols out of thin air
	if stats.in_flight == 0 {
	    return SymbolKind::Source;
	}

	if self.left_for_tail_protection > 0 {
	    return SymbolKind::Repair;
	}

	// we only send repair symbols for recovery and tail protection in the non-incremental case
	if !self.incremental {
	    return SymbolKind::Source;
	}

	// we cannot generate repair symbols out of thin air
	if stats.in_flight == 0 {
	    return SymbolKind::Source;
	}
	
	let should_p = match self.reliability_level {
	    ReliabilityLevel::RecoveryOnly => {
		return SymbolKind::Source;
	    },
	    ReliabilityLevel::BurstLossTolerance(b) => {
		b as f64 / stats.in_flight as f64
	    },
	    ReliabilityLevel::FixedRedundancyRatio(p) => {
		p
	    },
	};
	
	if stats.protection_ratio < should_p {
	    return SymbolKind::Repair
	} else {
	    return SymbolKind::Source
	}
    }

    /// Sets the reliability level that is to be respected by the redundancy scheduler
    pub fn set_reliability_level(&mut self, lvl: ReliabilityLevel) {
	self.reliability_level = lvl;
    }

    pub fn set_incremental(&mut self, i: bool) {
	self.incremental = i;
    }
  
    /// Initialize the Encoder with the maximum number of packets that can be simultaneously part of the sliding window.
    pub fn new(config: FecConfig) -> Result<Self, FecError> {
        Ok(Encoder {
            config,
            gf: Gf::new()?,
            next_id: 0,
            sliding_window: VecDeque::with_capacity(64),
            seed_offset: (u64::MAX, 0),
	    reliability_level: ReliabilityLevel::RecoveryOnly,
	    lost: 0,
	    in_flight_rs: BTreeMap::new(),
	    recovered: 0,
	    app_limited: true,
	    left_for_tail_protection: 0,
	    incremental: true,
        })
    }

    /// Get the next source symbol ID
    pub fn get_next_sid(&self) -> u64 {
        self.next_id
    }

    /// Add a source symbol to the sliding window.
    /// Creates and returns the source packet.
    ///
    /// Decode with [Decoder::decode_packet].
    pub fn add_source_symbol(&mut self, src: &[u8]) -> Result<u64, FecError> {
        if src.len() > self.config.get_max_payload_length() {
            Err(FecError::PayloadLongerThanMaximum)?;
        }

        self.sliding_window.push_back(SourceSymbol::new(
            self.next_id,
            src,
            &self.config,
        )?);
        let res = self.next_id;
        self.next_id += 1;
	self.app_limited = false;
	self.left_for_tail_protection = 0;
        //self.rsrc.add_source_symbol();
        Ok(res)
    }

    /// Encodes current symbols of the sliding window (added via [self.make_source_packet]) into a coded packet.
    ///
    /// Decode with [Decoder::decode_packet].
    pub fn generate_repair_symbol(
        &mut self,
    ) -> Result<RepairSymbol, FecError> {
        let smallest_symbol_id = self
            .sliding_window
            .front()
            .ok_or(FecError::SlidingWindowIsEmpty)?
            .source_symbol_id;
        let largest_symbol_id = self
            .sliding_window
            .back()
            .ok_or(FecError::SlidingWindowIsEmpty)?
            .source_symbol_id;

        let seed_offset = {
            if self.seed_offset.0 == largest_symbol_id {
                self.seed_offset.1 += 1;
                self.seed_offset.1
            } else {
                self.seed_offset = (largest_symbol_id, 0);
                0_u16
            }
        };

        Ok(RepairSymbol::new_from_sliding_window(
            smallest_symbol_id,
            largest_symbol_id,
            seed_offset,
            &mut self.sliding_window,
            &self.gf,
            &self.config,
        ))
    }

    /// To be called if repair symbol actually is sent
    pub fn put_repair_symbol_in_flight(&mut self, largest_symbol_id: u64) {
	if self.lost >  0 {
	    self.lost = self.lost.saturating_sub(1);
	    self.recovered += 1;
	    trace!("Encoder recovering loss with a repair symbol");
	} else if self.left_for_tail_protection > 0 {
	    self.left_for_tail_protection -= 1;
	}

	self.in_flight_rs.entry(largest_symbol_id)
	    .and_modify(|c| *c += 1)
	    .or_insert(1);
    }

    /// Handles acknowledgement packets from decoder ([Decoder::make_ack]).
    pub fn handle_ack(&mut self, next_missing_symbol_id: u64) {
	// remove from in flight sids
        self.sliding_window.retain(
            |SourceSymbol {
                 source_symbol_id, ..
             }| *source_symbol_id >= next_missing_symbol_id,
        );
	// remove all in flight rids protecting only smaller sids
	self.in_flight_rs.retain(|&sid, _count| sid >= next_missing_symbol_id);
    }

    /// Returns the current size of the sliding window.
    pub fn buffered_symbols(&self) -> usize {
        self.sliding_window.len()
    }

    /// Returns the length of repair symbols
    pub fn get_repair_symbol_len(&self) -> usize {
	self.config.get_payload_buffer_size()
    }
}
