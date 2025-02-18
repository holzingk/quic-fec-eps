use super::error::FecError;
use super::symbol::{RepairSymbol, SourceSymbol, Symbol, SymbolRow};
use super::{FecConfig, Gf};

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::cmp;

use std::convert::TryInto;
use ::gf256::*;

use rand::prelude::*;

#[cfg(feature = "qlog")]
use qlog::events::EventData;

#[derive(Debug)]
struct SymbolAckFrequencyController {
    // last_ack_send_time: Option<Instant>,
    // last_acked_sid: Option<u64>,
    // last_ack_buffered_symbols: usize,
    ack_elicited: bool,
    // eliciting_factor: f32,
    // ack_delay_factor: f32,
}

impl Default for SymbolAckFrequencyController {
    fn default() -> Self {
        Self {
            // last_ack_send_time: None,
            // last_acked_sid: None,
            // last_ack_buffered_symbols: 0,
            ack_elicited: false,
            // eliciting_factor: 0.1,
            // ack_delay_factor: 0.25,
        }
    }
}

impl SymbolAckFrequencyController {
    /// Creates a new controller using default values
    pub fn new() -> Self {
        Default::default()
    }

    // Informs the controller that a symbol ack is about to be sent
    // fn add_symbol_ack(&mut self, sid: u64, num_buffered_symbols: usize) {
    //     self.last_ack_send_time = Some(Instant::now());
    //     assert!(self.last_acked_sid.unwrap_or(0) <= sid);
    //     self.last_acked_sid = Some(sid);
    //     self.ack_elicited = false;
    //     self.last_ack_buffered_symbols = num_buffered_symbols;
    // }

    /// Informs the controller that an ack is elicited (e.g. due to ACK loss detected)
    fn set_ack_elicited(&mut self) {
        self.ack_elicited = true;
    }

    // Returns `true` if an ACK should be sent now
    // fn should_send_symbol_ack(
    //     &self, next_missing_sid: u64, rtt: Duration, num_buffered_symbols: usize,
    // ) -> bool {
    //     if self.last_ack_send_time.is_none() || self.last_acked_sid.is_none() {
    //         if num_buffered_symbols > 0 {
    //             return true;
    //         } else {
    //             return false;
    //         }
    //     }
    //     let buffered_symbols_changed_by_factor = num_buffered_symbols as f32
    //         * (1.0 + self.eliciting_factor)
    //         > self.last_ack_buffered_symbols as f32
    //         || num_buffered_symbols as f32 * (1.0 - self.eliciting_factor)
    //             < self.last_ack_buffered_symbols as f32;
    //     let rtt_elapsed = next_missing_sid > self.last_acked_sid.unwrap_or(0)
    //         && Instant::now().duration_since(self.last_ack_send_time.unwrap())
    //             > rtt.mul_f32(self.ack_delay_factor);

    //     if buffered_symbols_changed_by_factor || rtt_elapsed || self.ack_elicited
    //     {
    //         true
    //     } else {
    //         false
    //     }
    // }
}

#[derive(Debug, Default)]
struct SourceSymbols {
    source_buffer: BTreeMap<u64, SourceSymbol>,
    missing_sids: BTreeSet<u64>,
    upper_bound: u64,
}

impl SourceSymbols {
    fn new() -> Self {
        Default::default()
    }

    fn add(&mut self, source_symbol: SourceSymbol) {
        // check if it's the next expected one
        if match self.source_buffer.last_key_value() {
            // empty
            None => {
                for missing_sid in self.upper_bound..source_symbol.source_symbol_id {
		    trace!("Gap between current upper bound {} and source symbol id {}, treating {missing_sid} SID as missing",
			   self.upper_bound,
			   source_symbol.source_symbol_id);
		    self.missing_sids.insert(missing_sid);
                }
                true
            },
            // there is a gap
            Some((last_sid, _)) if source_symbol.source_symbol_id > *last_sid + 1 =>
            {
                // everything within the gap is missing
		trace!("{} is a non-consecutive SID, adding everything from {last_sid} as missing",
		       source_symbol.source_symbol_id);
                for missing_sid in (*last_sid + 1)..source_symbol.source_symbol_id
                {
                    self.missing_sids.insert(missing_sid);
                }
                true
            },
            // there is something out of order
            Some((last_sid, _)) if source_symbol.source_symbol_id <= *last_sid =>
            {
                if self
                    .source_buffer
                    .contains_key(&source_symbol.source_symbol_id)
                {
                    assert!(
                        !self
                            .missing_sids
                            .contains(&source_symbol.source_symbol_id),
                        "Duplicated source symbol was missing"
                    );
                    warn!("Duplicated source symbol detected");
                    false
                } else {
                    // redact SID from missing
                    let res =
                        self.missing_sids.remove(&source_symbol.source_symbol_id);
                    assert!(res, "Out of order SID was not missing");
                    true
                }
            },
            // normal
            Some(_) => true,
            } {
		if source_symbol.get_source_symbol_id() > self.upper_bound {
		    self.upper_bound = source_symbol.get_source_symbol_id();
		    trace!("Setting upper bound to {}", self.upper_bound);
		}
		// in any case, it's not longer missing
		let _res = self.missing_sids.remove(&source_symbol.source_symbol_id);
		self.source_buffer
                    .insert(source_symbol.source_symbol_id, source_symbol);
            }
	}

    fn inform_rs_largest_symbol_id(&mut self, largest_symbol_id: u64) {
	// check if it's the next expected one
	match self.source_buffer.last_key_value() {
            // empty
            None => {
                for missing_sid in 0..largest_symbol_id {
                    self.missing_sids.insert(missing_sid);
                }
            },
            // referencing soms SS which we have not yet seen
            Some((last_sid, _))
                if largest_symbol_id > *last_sid =>
            {
                // everything within the gap is missing
                for missing_sid in (*last_sid + 1)..=largest_symbol_id  {
                    self.missing_sids.insert(missing_sid);
                }
            },
	    Some(_) => {}
	};
	// if largest_symbol_id > self.upper_bound {
        //     self.upper_bound = largest_symbol_id;
	//     trace!("Setting upper bound to {}, due to rs", self.upper_bound);
        // }

	
    }

    fn drop_symbols_older_than(&mut self, lower_bound: u64) {
	//trace!("Source buffer was {:?}", self.source_buffer);
        self.source_buffer.retain(|sid, _ss| *sid >= lower_bound);
	//trace!("Source buffer currently is {:?}", self.source_buffer);
    }

    fn count_missing(&self) -> usize {
        self.missing_sids.len()
    }

    fn get_symbols(&mut self) -> impl Iterator<Item = &mut SourceSymbol> {
        self.source_buffer.values_mut()
    }

    fn get_smallest_sid(&self) -> Option<u64> {
	self.source_buffer.first_key_value()
	    .map(|(k, _v)| k.to_owned())
    }

    fn get_largest_sid(&self) -> Option<u64> {
	self.source_buffer.last_key_value()
	    .map(|(k, _v)| k.to_owned())
    }

    fn get_missing_sids(&self) -> impl Iterator<Item = &u64> {
        self.missing_sids.iter()
    }

    fn next_missing_sid(&self) -> u64 {
        self.missing_sids
            .first()
            .cloned()
            .unwrap_or_else(|| self.get_sid_upper_bound() + 1)
    }

    fn successfully_restored(&mut self, highest_reconstructed: u64) {
        self.missing_sids.clear();
	self.upper_bound = highest_reconstructed;
    }

    fn get_sid_upper_bound(&self) -> u64 {
	trace!("Using sid upper bound");
        self.upper_bound
    }

    fn get_source_buffer_size(&self) -> usize {
        self.source_buffer.len()
    }
}

#[derive(Debug, Default)]
struct RepairSymbols {
    // key: RID
    repair_buffer: BTreeMap<u64, RepairSymbol>,
    // (SID, RIDs)
    window: BTreeMap<u64, HashSet<u64>>,
    next_rid: u64,
}

impl RepairSymbols {
    pub fn new() -> Self {
        Default::default()
    }

    fn add(&mut self, rs: RepairSymbol) {
        // add references to window
        for sid in rs.smallest_symbol_id..=rs.largest_symbol_id {
            match self.window.get_mut(&sid) {
                None => {
                    let mut hs = HashSet::new();
                    hs.insert(self.next_rid);
                    self.window.insert(sid, hs);
                },
                Some(hs) => {
                    hs.insert(self.next_rid);
                },
            }
        }
        // add to repair buffer
        let res = self.repair_buffer.insert(self.next_rid, rs);
        assert!(res.is_none());
        self.next_rid += 1;
    }

    /// Returns the smallest source symbol that is still referenced after the GC
    fn collect_garbage(&mut self, smallest_missing_sid: u64) -> Option<u64> {
	//trace!("Window before GC is {:?}", self.window);
        // neues hashset anlegen: alle rids rausfinden die unterhalb der kleinsten vermissten sid liegen
        let mut below: HashSet<u64> = HashSet::new();
        // alle sids kleiner als gesuchte im fenster iterieren
        for (_sid, rids) in self.window.range(..smallest_missing_sid) {
            for rid in rids {
                below.insert(*rid);
            }
        }
	trace!("RIDs below missing sid {smallest_missing_sid} are: {:?}", below);

        // rids, die noch für smallest missing sid benötigt werden von below abziehen
        let unneeded_rids = match self.window.get_mut(&smallest_missing_sid) {
            None => {
		trace!("No rids are needed anymore");
		below.clone()
	    },
            Some(remaining_rids) => {
		trace!("The following rids are still needed, e.g. for {smallest_missing_sid}: {:?}", remaining_rids);
                below.difference(remaining_rids).cloned().collect()
            },
        };
	trace!("The following rids are no longer needed and are removed: {:?}", unneeded_rids);
        for unneeded_rid in unneeded_rids {
            let rs = self.repair_buffer.remove(&unneeded_rid).unwrap();
	    assert!(rs.largest_symbol_id < smallest_missing_sid);
            for (_sid, rids) in self.window.range_mut(rs.smallest_symbol_id..=rs.largest_symbol_id)
            {
                rids.remove(&unneeded_rid);
            }
        }
        // leere einträge löschen
        self.window.retain(|_sid, rids| !rids.is_empty());
	//trace!("Window after GC is {:?}", self.window);
        self.first_sid_in_window()
    }

    fn first_sid_in_window(&self) -> Option<u64> {
        self.window
            .first_key_value()
	    .map(|(k, _v)| *k)
    }

    // fn get_symbols(&mut self) -> impl Iterator<Item = &mut RepairSymbol> {
    //     self.repair_buffer.values_mut()
    // }

    fn get_rid_symbols(&mut self) -> impl Iterator<Item = (&u64, &mut RepairSymbol)> {
        self.repair_buffer.iter_mut()
    }

    fn get_ref(&self, rid: u64) -> &RepairSymbol {
	self.repair_buffer.get(&rid).unwrap()	
    }

    fn len(&self) -> usize {
        self.repair_buffer.len()
    }
}

#[derive(Debug)]
pub struct DecodingStats {
    pub recovered: u64,
    pub received_source_symbols: u64,
    pub received_repair_symbols: u64,
}

/// The decoder for the Tetrys protocol
#[derive(Debug)]
pub struct Decoder {
    config: FecConfig,
    gf: Gf,
    source_symbols: SourceSymbols,
    repair_symbols: RepairSymbols,
    recovered: u64,
    received_source_symbols: u64,
    received_repair_symbols: u64,
    safc: SymbolAckFrequencyController,
    encoder_lower_bound: Option<u64>,
}

impl Decoder {
    /// New
    pub fn new(config: FecConfig) -> Result<Self, FecError> {
        Ok(Self {
            config,
            gf: Gf::new()?,
            source_symbols: SourceSymbols::new(),
            repair_symbols: RepairSymbols::new(),
	    recovered: 0,
	    received_source_symbols: 0,
	    received_repair_symbols: 0,
            safc: SymbolAckFrequencyController::new(),
	    encoder_lower_bound: None,
        })
    }

    pub fn get_stats(&self) -> DecodingStats {
	DecodingStats {
	    recovered: self.recovered,
	    received_source_symbols: self.received_source_symbols,
	    received_repair_symbols: self.received_repair_symbols
	}
    }

    /// Returns qlog event
    pub fn qlog_event(&self, fec_id: u64) -> EventData {
	qlog::events::EventData::DecoderMetricsUpdated(qlog::events::quic::DecoderMetricsUpdated {
	    fec_id,
	    recovered: self.recovered,
	    rx_ss: self.received_source_symbols,
	    rx_rs: self.received_repair_symbols,
	    window_ss: self.num_source_symbols() as u64,
	    window_rs: self.num_repair_symbols() as u64,
	})
    }
    
    /// Add new received source symbol
    pub fn add_source_symbol(
        &mut self, source_symbol_id: u64, src: &[u8],
    ) -> Result<(), FecError> {
	trace!("Received source symbol {source_symbol_id}");
        let ss = SourceSymbol::new(source_symbol_id, src, &self.config)?;
        self.source_symbols.add(ss);
	self.received_source_symbols += 1;
	// Collect garbage as repair symbols might have become unnecessary
	self.collect_garbage();
        Ok(())
    }

    /// Add new received repair symbol
    pub fn add_repair_symbol(
        &mut self, smallest_symbol_id: u64, largest_symbol_id: u64, seed: u16,
        src: &[u8],
    ) -> Result<(), FecError> {
        let rs = RepairSymbol::new_from_encoded(
            smallest_symbol_id,
            largest_symbol_id,
            seed,
            &self.config,
            src,
        )?;
	self.source_symbols.inform_rs_largest_symbol_id(largest_symbol_id);
        self.repair_symbols.add(rs);
	self.received_repair_symbols += 1;
        Ok(())
    }

    /// Returns the next missing SID
    pub fn get_ack(&mut self) -> u64 {
        let ack_sid = self.source_symbols.next_missing_sid();
        ack_sid
    }
    
    /// Tries to decode something
    pub fn try_decode(&mut self) -> Vec<Vec<u8>> {
        let mut ret = Vec::new();
        if !self.decoding_possible() {
            return ret;
        }

        let smallest_sid_in_ss = self.source_symbols.get_smallest_sid();
	    // get_symbols()
	    // .min_by_key(|ss| ss.get_source_symbol_id())
	    // .map(|ss| ss.get_source_symbol_id());
	let largest_sid_in_ss = self.source_symbols.get_largest_sid();
	    // .get_symbols()
	    // .max_by_key(|ss| ss.get_source_symbol_id())
	    // .map(|ss| ss.get_source_symbol_id());

        let missing_sids: Vec<u64> =
            self.source_symbols.get_missing_sids().copied().collect();
        trace!(
            "{} sids are missing, {:?}",
            missing_sids.len(),
            missing_sids
        );

	trace!("Sid range in ss buffer is {:?}-{:?}", smallest_sid_in_ss, largest_sid_in_ss);
	// could be that the first symbols are missing
	let smallest_sid = cmp::min(
	    missing_sids.iter().min().expect("there are missing sids at this point").to_owned(),
	    smallest_sid_in_ss.unwrap_or(0).to_owned());
	//similar last sid could be missing
	let largest_sid = cmp::max(
	    missing_sids.iter().max().expect("idem").to_owned(),
	    largest_sid_in_ss.unwrap_or(0).to_owned());
	
	// foreach missing sid, collect all protecting RSs in a set
	// we assume: youngest missing source symbols have fewest repair symbols
	// so we build differences backwards
	//let mut rids_used: HashSet<u64> = HashSet::new();
	let mut rs_for_missing_sid: Vec<HashSet<u64>> = vec![HashSet::new(); missing_sids.len()];
	for (i,  missing_sid) in missing_sids.iter().enumerate() {
	    for (rid, rs) in self.repair_symbols.get_rid_symbols() {
		if rs.protecting(*missing_sid) {
//		    let was_free = rids_used.insert(*rid);
//		    if was_free {
		    if !rs_for_missing_sid[i].insert(*rid) {
			trace!("Adding {rid}, protecting {missing_sid}");
		    }
//		    }
		}
	    }
	}
	//println!("blub");
	let mut rng = thread_rng();
	// just try to choose random selection and get it solved
	'combinations: for attempt in 0..256 {
	    let mut rs_combination: Vec<u64> = Vec::new();
	    let mut rs_for_missing_sid_temp = rs_for_missing_sid.clone();
	    for missing_sid in rs_for_missing_sid_temp.iter_mut().rev() {
		for e in rs_combination.iter() {
		    missing_sid.remove(e);
		}
		if missing_sid.len() == 0 {
		    trace!("Missing sid is empty");
		    continue 'combinations;
		}
		let element_id = rng.gen_range(0..missing_sid.len());
//		println!("rng yielded {element_id}, lem is {}", missing_sid.len());
		let new_element: u64 = missing_sid.iter()
		    .nth(element_id)
		    .unwrap().clone();
		if rs_combination.contains(&new_element) {
		    trace!("Attempt {attempt}, element {new_element} is already there in {:?}", rs_combination);
		    continue 'combinations;
		}
		rs_combination.push(new_element);
	    }
	    trace!("Attempt {attempt}, found combination {:?}", rs_combination);
	    
	    let mut matrix: Vec<SymbolRow> = Vec::new();
	    
	    for ss in self.source_symbols.get_symbols() {
		matrix.push(SymbolRow::from_ss(ss, smallest_sid, largest_sid));
            }

	    let num_source_symbols = matrix.len();
            trace!("{} ss are available", num_source_symbols);
	    
	    for rid in rs_combination {
		let rs = self.repair_symbols.get_ref(rid);
		trace!("Adding {rid} with smallest sid {} and largest sid {}. Smallest sid and largest sid for this iteration are {} - {}",
		       rs.smallest_symbol_id, rs.largest_symbol_id,
		       smallest_sid, largest_sid);
		assert!(smallest_sid <= rs.smallest_symbol_id);
		assert!(largest_sid >= rs.largest_symbol_id);
		matrix.push(SymbolRow::from_rs(rs, smallest_sid, largest_sid));
	    }
	    
            // get to row-echelon form
            let mut h = 0; //pivot row
            let mut k = 0; // pivot column
            let n = matrix.len();
            while h < n && k < n {
		// find the k-th pivot: row with the first non zero entry in that column
		let mut i_pivot = None;
		for i in h..n {
                    trace!(
			"h {h}, n {n}, i {i}, sm {}, lg {}",
			smallest_sid,
			largest_sid
                    );
                    let coeff = matrix[i].get_coefficients_mut(
			smallest_sid,
			largest_sid,
                    );
                    trace!("{:?}", coeff);
                    if coeff[k] > 0 {
			i_pivot = Some(i);
			break;
                    }
		}
		if i_pivot.is_none() {
                    trace!("Could not find pivot, giving up at h {h}, n {n}, k {k}");
                    continue 'combinations;
		}
		trace!("Found pivot {}", i_pivot.unwrap() as u64);

		matrix.swap(h, i_pivot.unwrap());

		trace!(
                    "Coefficients of pivot row h {h} {:?}",
                    matrix[h].get_coefficients_mut(
			smallest_sid,
			largest_sid
                    )
		);
		trace!(
                    "pivot) element {}",
                    matrix[h].get_coefficients_mut(
			smallest_sid,
			largest_sid
                    )[k]
		);

		let pivot = matrix[h].get_coefficients_mut(
                    smallest_sid,
                    largest_sid,
		)[k];
		// calculates the multiplicative inverse / reciprocal element
		let inv_pivot = self.gf.inverse(pivot);
		trace!("Inverted pivot {inv_pivot}");
		// for all rows below pivot
		for i in (h + 1)..n {
                    trace!(
			"Coefficients of row i {i} (below pivot) {:?}",
			matrix[i].get_coefficients_mut(
                            smallest_sid,
                            largest_sid
			)
                    );
                    // multiplication is bitwise XOR
		    let factor = (
			gf256(
			    matrix[i].get_coefficients_mut(
				smallest_sid,
				largest_sid)[k]) *
			    gf256(inv_pivot)
		    ).try_into().unwrap();
                    // let factor = self.gf.multiply_u8(
		    // 	matrix[i].get_coefficients_mut(
                    //         smallest_sid,
                    //         largest_sid,
		    // 	)[k],
		    // 	inv_pivot,
                    // );
                    trace!("Factor {factor}");
                    // h < i we cannot borrow two entries of the matrix vector as mut, so it gets ugly here
                    let (above_pivot, below_pivot) =
			matrix.as_mut_slice().split_at_mut(i);
                    let symbol_h = &mut above_pivot[h];
                    let symbol_i = &mut below_pivot[0];
                    self.gf.add_n_times(
			symbol_i,
			symbol_h,
			factor,
			smallest_sid,
			largest_sid,
                    );
                    trace!(
			"After operation {:?}",
			matrix[i].get_coefficients_mut(
                            smallest_sid,
                            largest_sid
			)
                    );

                    assert_eq!(
			matrix[i].get_coefficients_mut(
                            smallest_sid,
                            largest_sid
			)[k],
			0
                    );
		}
		// increase pivot row and column
		h += 1;
		k += 1;
            }
            trace!("Printing coefficients in euler form");
            for e in matrix.iter_mut() {
		trace!(
                    "{:?}",
                    e.get_coefficients(smallest_sid, largest_sid)
		);
            }

            trace!("Starting backward substitution");
            // backward substitution:
            for i in (0..n).rev() {
		let elem_i_i = matrix[i].get_coefficients_mut(
                    smallest_sid,
                    largest_sid,
		)[i];
		if elem_i_i == 0 {
                    // no solution
                    return ret;
		}
		let inverse_i_i = self.gf.inverse(elem_i_i);
		trace!(
                    "Element {},{}: {} (inverse {})",
                    i,
                    i,
                    elem_i_i,
                    inverse_i_i
		);

		// invert / solve this row
		self.gf.multiply_symbol(
                    &mut matrix[i],
                    inverse_i_i,
                    smallest_sid,
                    largest_sid,
		);
		trace!(
                    "Coefficients of row {i} solved are {:?}",
                    matrix[i].get_coefficients(
			smallest_sid,
			largest_sid
                    )
		);
		assert_eq!(
                    matrix[i].get_coefficients_mut(
			smallest_sid,
			largest_sid
                    )[i],
                    1
		);

		for j in 0..i {
                    // row(j) = row(j) - row(j,i) * row(j))
                    let factor = matrix[j].get_coefficients_mut(
			smallest_sid,
			largest_sid,
                    )[i];
                    trace!("Factor is {factor}");
                    let (below_i, above_i) = matrix.as_mut_slice().split_at_mut(i);
                    let symbol_j = &mut below_i[j];
                    trace!(
			"Coefficients of symbol j {:?}",
			symbol_j.get_coefficients_mut(
                            smallest_sid,
                            largest_sid
			)
                    );
                    let symbol_i = above_i.first_mut().unwrap();
                    trace!(
			"Coefficients of symbol i {:?}",
			symbol_i.get_coefficients_mut(
                            smallest_sid,
                            largest_sid
			)
                    );
                    self.gf.add_n_times(
			symbol_j,
			symbol_i,
			factor,
			smallest_sid,
			largest_sid,
                    );

                    trace!(
			"symbol {j} after operation {:?}",
			matrix[j].get_coefficients_mut(
                            smallest_sid,
                            largest_sid
			)
                    );

                    assert_eq!(
			matrix[j].get_coefficients_mut(
                            smallest_sid,
                            largest_sid
			)[i],
			0
                    );
		}
            }

            // extract missing source symbols and push to ret vector
            for missing_sid in &missing_sids {
		trace!("pushing missing sid {missing_sid} to return vec, smallest sid {}, offset {}",
		       smallest_sid,
		       *missing_sid as usize - smallest_sid as usize);
		let new = matrix
                    [*missing_sid as usize - smallest_sid as usize]
                    .get_payload_as_ref()
                    .to_vec();
		self.add_source_symbol(*missing_sid, &new.as_slice()).unwrap();
		ret.push(new);
		self.recovered += 1;
            }
            // inform source symbols that something is no longer missing
            self.source_symbols.successfully_restored(largest_sid);
	    break;
	}
	ret
    }

    /// Informs decoder that no SIDs lower than lower_bounds are expected anymore
    pub fn add_encoder_lower_bound(&mut self, lower_bound: u64) {
	self.encoder_lower_bound = Some(lower_bound);
    }
    
    /// cleans up repair symbols and source symbols that cannot be used anymore to return SIDs >= next symbol id
    /// todo: noop if no substantial change occurred
    fn collect_garbage(&mut self) {
	let ss_next_missing = self.source_symbols.next_missing_sid();
        match self.repair_symbols.collect_garbage(ss_next_missing) {
	    Some(smallest_needed) => {
		trace!("Dropping source symbols smaller than {smallest_needed}, next_missing ss is {ss_next_missing}");
		self.source_symbols.drop_symbols_older_than(smallest_needed);
	    },
	    None => {
		self.source_symbols.drop_symbols_older_than(self.encoder_lower_bound.unwrap_or(0));
	    },
	}
    }

    fn decoding_necessary(&self) -> bool {
        self.source_symbols.count_missing() > 0
    }

    fn decoding_possible(&mut self) -> bool {
        if self.decoding_necessary() {
            self.collect_garbage();
            self.repair_symbols.len() >= self.source_symbols.count_missing()
        } else {
            false
        }
    }

    /// Informs the decoder that sneding an ACK is desired ASAP, e.g. due to a detected loss
    pub fn set_ack_elicited(&mut self) {
        self.safc.set_ack_elicited();
    }

    /// Returns the count of stored source symbols
    pub fn num_source_symbols(&self) -> usize {
        self.source_symbols.get_source_buffer_size()
    }

    /// Returns the count of stored repair symbols
    pub fn num_repair_symbols(&self) -> usize {
        self.repair_symbols.len()
    }

    /// Returns the count of missing source symbols
    pub fn num_missing_sids(&self) -> usize {
        self.source_symbols.count_missing()
    }

    /// Returns the missing source symbols
    pub fn get_missing_sids(&self) -> Vec<u64> {
	self.source_symbols.get_missing_sids().cloned().collect()
    }
}
