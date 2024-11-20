use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::fmt::Formatter;
use itertools::Itertools;

/// Implementation of the Hierarchical Link Sharing (HLS) Scheduling algorithm.
#[derive(Clone, Debug)]
pub struct HLSClass {
    /// Unique identifier of the class.
    pub(crate) id: u64,

    /// Parent class of the class.
    pub parent: Option<u64>,

    /// Set of children of the class.
    pub children: HashSet<u64>,

    /// The class' weight (in promille).
    pub weight: u32,

    /// Number of bytes the class is allowed to transmit.
    pub balance: i64,

    /// Permits for the transmission of bytes. Collected from descendants.
    pub residual: i64,

    /// Number of bytes that a child class with weight set to one can transmit.
    /// `None` if not yet computed.
    fair_quota: Option<i64>,

    /// The stream id of this class. `None` for classes that are not leaves.
    pub stream_id: Option<u64>,

    /// Whether the class became idle during a round.
    idle: bool,

    /// Whether the class' balance has already been updated.
    pub ticked: bool,

    /// How many bytes this stream has emitted in the current round.
    pub(crate) emitted: i64,

    /// A guarantee in bytes per round, derived from relative weights and the root's capacity.
    pub guarantee: i64,

    /// The urgency of the class.
    pub urgency: u8,

    /// Whether the class is marked as incremental.
    pub incremental: bool,

    /// The class' burst loss tolerance.
    pub burst_loss_tolerance: u32,

    /// The class' protection ratio.
    pub protection_ratio: u32,

    /// Tolerated waiting time for protection repair symbols.
    pub repair_delay_tolerance: u32,
}

/// Represents a class hierarchy in the context of the HLS paper.
pub struct HLSHierarchy {
    /// Map of class identifiers to class objects.
    pub(crate) classes: HashMap<u64, HLSClass>,

    /// Associates an EPS id with a class ID used by the HLS scheduler.
    pub eps_id_to_hls_id: HashMap<String, u64>,

    /// Identifier of the root class.
    pub root: u64,

    /// The capacity of the hierarchy in bytes.
    pub capacity: u64,

    /// Next class identifier to be assigned.
    next_id: u64,
}

impl HLSHierarchy {
    fn guarantee_from_weight(&mut self, node_id: u64) {
        let root = self.root;

        let mut ancestors_or_self = self.ancestors(node_id);
        ancestors_or_self.insert(node_id);

        let mut product: Vec<f64> = vec![];

        for j in ancestors_or_self {
            if j != root {
                let node_weight = self.class(j).weight as f64;
                let node_siblings = self.siblings(j);

                // Sum up the weights of the siblings
                let sum_siblings_weights = node_siblings
                    .iter()
                    .map(|k| self.class(*k).weight)
                    .sum::<u32>()
                    as f64;

                match sum_siblings_weights {
                    x if x <= 0.0 => product.push(0.0),
                    _ => product.push(node_weight / sum_siblings_weights),
                }
            }
        }

        let guarantee = (self.capacity as f64) * product.iter().fold(1.0, |acc, x| acc * x);

        if let Some(n) = self.classes.get_mut(&node_id) {
            n.guarantee = guarantee as i64;
        }
    }

    /// Converts the relative weights in a hierarchy to absolute ones.
    /// Requires a capacity to have been set.
    pub fn generate_guarantees(&mut self) {
        let node_ids: Vec<u64> = self.classes.keys().copied().collect();

        for node_id in node_ids {
            self.guarantee_from_weight(node_id);
        }
    }

    /// Removes a stream that has finished from the hierarchy.
    pub fn delete_class(&mut self, stream_id: u64, mtu: usize) {
        let root = self.root;
        let capacity = self.capacity;
        let leaves = self.leaf_descendants(root);

        let mut capacity_decrease: u64 = 0;

        // Find the leaf with the corresponding stream ID
        if let Some(class_id) = leaves.iter().find_or_first(|l| self.class(**l).stream_id == Option::from(stream_id)) {
            let mut class = self.class(*class_id);

            // Remove children bottom-up
            while let Some(parent_id) = class.parent {
                let parent_class = self.mut_class(parent_id);
                parent_class.children.remove(class_id);

                if parent_class.children.is_empty() {
                    // No children left, remove this class now too
                    class = parent_class;

                    // Reduce the capacity.
                    capacity_decrease += mtu as u64;
                } else {
                    break;
                }
            }
        }

        if capacity >= capacity_decrease {
            self.capacity -= capacity_decrease
        } else {
            self.capacity = 0
        };

        // Done. Generate new guarantees.
        self.generate_guarantees();
    }
}

impl Eq for HLSClass {}

impl PartialEq<Self> for HLSClass {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl PartialOrd<Self> for HLSClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Ignore priority if class ID matches. Default EPS would use the stream ID here.
        if self.id == other.id {
            return Some(Ordering::Equal);
        }

        // First, order by urgency...
        if self.urgency != other.urgency {
            return self.urgency.partial_cmp(&other.urgency);
        }

        // ...when the urgency is the same, and both are not incremental, order
        // by class ID. Default EPS would use the stream ID here.
        if !self.incremental && !other.incremental {
            return self.id.partial_cmp(&other.id);
        }

        // ...non-incremental takes priority over incremental...
        if self.incremental && !other.incremental {
            return Some(Ordering::Greater);
        }

        if !self.incremental && other.incremental {
            return Some(Ordering::Less);
        }

        // ...finally, when both are incremental, `other` takes precedence (so
        // `self` is always sorted after other same-urgency incremental
        // entries).
        Some(Ordering::Greater)
    }
}

impl Ord for HLSClass {
    fn cmp(&self, other: &Self) -> Ordering {
        // `partial_cmp()` never returns `None`, so this should be safe.
        self.partial_cmp(other).unwrap()
    }
}

impl HLSClass {
    /// Instantiates a new HLS Class.
    pub fn new(id: u64,
               parent: Option<u64>,
               urgency: u8,
               incremental: bool,
               weight: u32,
               burst_loss_tolerance: u32,
               protection_ratio: u32,
               repair_delay_tolerance: u32
    ) -> HLSClass {
        HLSClass {
            id,
            parent,
            children: HashSet::new(),
            weight,
            balance: 0,
            residual: 0,
            fair_quota: None,
            stream_id: None,
            idle: false,
            emitted: 0,
            ticked: false,
            guarantee: 0,
            urgency,
            incremental,
            burst_loss_tolerance,
            protection_ratio,
            repair_delay_tolerance,
        }
    }
}

impl HLSHierarchy {
    /// Creates a hierarchy consisting of a default root node only.
    pub fn new() -> HLSHierarchy {
        let root = 0;
        let mut hierarchy = HLSHierarchy {
            classes: HashMap::new(),
            eps_id_to_hls_id: HashMap::new(),
            root,
            capacity: 0,
            next_id: root,
        };

        hierarchy.insert(3, false, 1, 0, 0, 0, None);
        hierarchy
    }

    /// Returns a reference to the class with the given identifier.
    pub fn class(&self, class_id: u64) -> &HLSClass {
        self.classes.get(&class_id).unwrap()
    }

    /// Returns a mutable reference to the class with the given identifier.
    pub fn mut_class(&mut self, class_id: u64) -> &mut HLSClass {
        self.classes.get_mut(&class_id).unwrap()
    }

    /// Returns the next class identifier to be assigned.
    fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Sets the stream id for a given leaf class.
    pub fn set_stream_id(&mut self, class_id: u64, stream_id: u64) {
        let class = self.mut_class(class_id);

        // Only leaf classes should have a stream id.
        if class.children.is_empty() {
            class.stream_id = Some(stream_id);
        }
    }

    /// Inserts a new class into the hierarchy.
    pub fn insert(&mut self,
                  urgency: u8,
                  incremental: bool,
                  weight: u32,
                  burst_loss_tolerance: u32,
                  protection_ratio: u32,
                  repair_delay_tolerance: u32,
                  parent: Option<u64>) -> u64 {

        let id = self.next_id();
        let class = HLSClass::new(id, parent, urgency, incremental, weight, burst_loss_tolerance, protection_ratio, repair_delay_tolerance);

        // If the class has a parent, update the parent's children
        if let Some(pid) = parent {
            if let Some(parent) = self.classes.get_mut(&pid) {
                parent.children.insert(id);
            }
        }

        self.classes.insert(id, class);
        id
    }

    pub(crate) fn children(&self, node_id: u64) -> HashSet<u64> {
        if let Some(node) = self.classes.get(&node_id) {
            node.children.clone()
        } else {
            HashSet::new()
        }
    }

    /// Returns the leaf descendants of a class.
    pub fn leaf_descendants(&self, node_id: u64) -> HashSet<u64> {
        let mut descendants: HashSet<u64> = HashSet::new();
        let mut queue: HashSet<u64> =
            self.children(node_id).into_iter().collect();

        while !queue.is_empty() {
            if let Some(current_id) = queue.iter().next().cloned() {
                queue.remove(&current_id);

                let children = self.children(current_id);
                if children.is_empty() {
                    descendants.insert(current_id);
                } else {
                    for child in children {
                        queue.insert(child); // Add new children to the queue
                    }
                }
            }
        }

        descendants
    }

    fn ancestors(&self, node_id: u64) -> HashSet<u64> {
        let mut ancestors: HashSet<u64> = HashSet::new();
        let mut current_id = node_id;

        while let Some(parent_id) = self.class(current_id).parent {
            ancestors.insert(parent_id);
            current_id = parent_id;
        }

        ancestors
    }

    pub(crate) fn parent(&self, node_id: u64) -> Option<u64> {
        if let Some(node) = self.classes.get(&node_id) {
            node.parent
        } else {
            None
        }
    }


    pub(crate) fn siblings(&self, node_id: u64) -> HashSet<u64> {
        // Return empty vector if the node does not exist
        if !self.classes.contains_key(&node_id) {
            return HashSet::new();
        }

        if let Some(parent_id) = self.parent(node_id) {
            if let Some(parent) = self.classes.get(&parent_id) {
                return parent.children.clone();
            }
        }

        // The node is a sibling of itself
        let mut reflexive = HashSet::new();
        reflexive.insert(node_id);

        reflexive
    }

    pub(crate) fn internal_nodes(&self, node_id: u64) -> HashSet<u64> {
        let mut internal_nodes: HashSet<u64> = HashSet::new();
        let mut queue: HashSet<u64> =
            self.children(node_id).into_iter().collect();

        while let Some(current_id) = queue.iter().next().cloned() {
            queue.remove(&current_id);

            let children = self.children(current_id);
            if !children.is_empty() {
                internal_nodes.insert(current_id);
                for child in children {
                    queue.insert(child);
                }
            }
        }

        internal_nodes
    }
}

impl fmt::Debug for HLSHierarchy {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        fn print_node(
            class: &HLSClass, classes: &HashMap<u64, HLSClass>,
            f: &mut Formatter<'_>, depth: u64, last: bool, prefix: String,
        ) -> fmt::Result {
            let current_prefix = if depth == 0 {
                "".to_string()
            } else if last {
                format!("{}└── ", prefix)
            } else {
                format!("{}├── ", prefix)
            };

            writeln!(
                f,
                "{}{}: g={}, w={}, u={}, i={}, r={}, b={}, a={}, blnc={}, {}res={}",
                current_prefix,
                match class.stream_id {
                    Some(stream_id) => format!("stream {} (class {})", stream_id, class.id),
                    None => class.id.to_string(),
                },
                class.guarantee,
                class.weight,
                class.urgency,
                class.incremental,
                class.protection_ratio,
                class.burst_loss_tolerance,
                class.repair_delay_tolerance,
                class.balance,
                if class.fair_quota.is_some() {
                    format!("fq: {}, ", class.fair_quota.unwrap())
                } else if !class.children.is_empty() {
                    "fq: None, ".to_string()
                } else {
                    "".to_string()
                },
                class.residual,
            )?;

            let child_prefix = if depth == 0 {
                "".to_string()
            } else if last {
                format!("{}    ", &prefix)
            } else {
                format!("{}│   ", &prefix)
            };

            if !class.children.is_empty() {
                for (i, child_id) in class.children.iter().enumerate() {
                    if let Some(child) = classes.get(child_id) {
                        // Determine if the current child is the last one.
                        let is_last_child = i == class.children.len() - 1;
                        print_node(
                            child,
                            classes,
                            f,
                            depth + 1,
                            is_last_child,
                            child_prefix.clone(),
                        )?;
                    }
                }
            }

            Ok(())
        }

        writeln!(f)?;

        if let Some(root) = self.classes.get(&self.root) {
            print_node(root, &self.classes, f, 0, true, "".to_string())?;
        }

        Ok(())
    }
}

/// The attributes of the HLS scheduler.
pub struct HLSScheduler {
    /// Set of active leaves.
    pub(crate) l_ac: HashSet<u64>,

    /// Set of active internal classes.
    i_ac: HashSet<u64>,

    /// The class hierarchy.
    pub hierarchy: HLSHierarchy,

    /// Whether the scheduler is in a main round.
    main_round: bool,

    /// Whether the scheduler is in a surplus round.
    surplus_round: bool,

    /// Class IDs (not stream IDs!) that have yet to be visited by the round-robin round.
    pub(crate) pending_leaves: VecDeque<u64>,
}

impl HLSScheduler {
    /// Creates a new HLS scheduler.
    pub fn new(hierarchy: HLSHierarchy) -> HLSScheduler {
        HLSScheduler {
            hierarchy,
            i_ac: HashSet::new(),
            l_ac: HashSet::new(),
            main_round: false,
            surplus_round: false,
            pending_leaves: VecDeque::new(),
        }
    }

    /// Computes the quota of an active class when a leaf is visited in the main round.
    /// The main round is finished once all active classes are visited in an arbitrary order.
    pub(crate) fn tick(&mut self, class_id: u64) {
        let parent_id = self.hierarchy.class(class_id).parent.unwrap();
        let parent_fair_quota = self.request_fair_quota(parent_id);
        let leaf_weight = self.hierarchy.class(class_id).guarantee;

        let leaf_quota = leaf_weight * parent_fair_quota;

        // Update the leaf's balance and quota as per formula (7).
        {
            let leaf = self.hierarchy.mut_class(class_id);
            leaf.balance += leaf_quota;
            leaf.ticked = true;
        }

        // Update the parent's balance.
        {
            let parent = self.hierarchy.mut_class(parent_id);
            parent.balance -= leaf_quota;
        }

        trace!("Ticked hierarchy is: {:?}", &self.hierarchy);
    }

    /// Returns the fair quota of the requested class.
    /// If necessary, balance updates are made for ancestor classes.
    fn request_fair_quota(&mut self, class_id: u64) -> i64 {
        if let Some(fq) = self.hierarchy.class(class_id).fair_quota {
            // If the parent's fair share has already been calculated in this round-robin round,
            // let the child use it to determine its balance.
            return fq;
        }

        // Parent has not computed its quota in the current round. Request it from its parent.
        let parent_id = self.hierarchy.class(class_id).parent;

        if let Some(pid) = parent_id {
            // Internal class reached.
            let parent_fair_quota = self.request_fair_quota(pid);

            //  Update balance and residual as per formula (6).
            let class_weight = self.hierarchy.class(class_id).guarantee;
            let class_quota = class_weight * parent_fair_quota;

            {
                let node = self.hierarchy.mut_class(class_id);

                // Update the class' balance.
                node.balance += node.residual + class_quota;

                // Update the class' residual
                node.residual = 0;
            }

            {
                // Update the parent's balance. There's a parent since this is an internal class.
                let parent = self.hierarchy.mut_class(parent_id.unwrap());
                parent.balance -= class_quota;
            }
        }

        // Calculate the fair share of the node as per (4).
        let children = self.hierarchy.children(class_id);
        let active_leaves = self.l_ac.clone();
        let active_internal = self.i_ac.clone();

        let active_leaves_or_internal_set =
            active_leaves.union(&active_internal).cloned().collect();
        let ks = children.intersection(&active_leaves_or_internal_set);

        // Sum of the guarantee of each class k in ks
        let sum_weights =
            ks.map(|k| self.hierarchy.class(*k).guarantee).sum::<i64>();

        let node = self.hierarchy.mut_class(class_id);

        // Calculate fair quota as in formula (4).
        let fair_quota = node.balance / sum_weights;

        // Set the class' fair quota
        node.fair_quota = Some(fair_quota);

        // Return the fair quota that was just calculated
        fair_quota
    }

    /// Prepares the round-robin scheduler for a new main round.
    /// `backlogged_streams`: Vector of flushable stream IDs
    pub(crate) fn init_round(&mut self, backlogged_streams: Vec<u64>) {
        let root_id = self.hierarchy.root;
        // Active classes are determined at the start of every round.
        let prev_active_leaves = self.l_ac.clone();

        let internal_classes: HashSet<u64> =
            self.hierarchy.internal_nodes(root_id);

        let leaf_classes: HashSet<u64> = self.hierarchy.leaf_descendants(root_id);

        let mut i_ac: HashSet<u64> = HashSet::new();
        let mut l_ac: HashSet<u64> = HashSet::new();

        let mut pending_leaves: VecDeque<u64> = VecDeque::new();

        // Initialize the scheduler before there are any active classes.
        if !self.main_round && !self.surplus_round {
            let root_id = self.hierarchy.root;
            let root = self.hierarchy.mut_class(root_id);
            root.balance = root.guarantee;
        }

        // Determine which leaf classes are active. Recomputed every round, i.e., main or surplus.
        for leaf in leaf_classes.clone() {
            let stream_id = self.hierarchy.class(leaf).stream_id.unwrap();
            let became_idle = self.hierarchy.class(leaf).idle;
            let leaf_weight = self.hierarchy.class(leaf).guarantee;

            // Mark leaves as not being yet ticked.
            self.hierarchy.mut_class(leaf).ticked = false;

            if backlogged_streams.contains(&stream_id) {
                let became_active = !prev_active_leaves.contains(&leaf);

                {
                    let stream_class = self.hierarchy.mut_class(leaf);
                    stream_class.emitted = 0;
                    l_ac.insert(stream_class.id);
                }

                // Modify root's residual do dynamically adjust Q*.
                if became_active {
                    {
                        trace!("Stream {} became active. Adding {} to root's residual.", stream_id, leaf_weight);
                        let root = self.hierarchy.mut_class(root_id);
                        root.residual += leaf_weight;
                    }
                }
            }

            if became_idle {
                {
                    trace!("Stream {} became idle in the past round. Subtracting {} from root's residual.", stream_id, leaf_weight);
                    let root = self.hierarchy.mut_class(root_id);
                    root.residual -= leaf_weight;
                }
            }

            {
                let stream_class = self.hierarchy.mut_class(leaf);

                // Reset idle indicator
                stream_class.idle = false;
            }
        }

        // The HLS paper states the round-robin happens at an arbitrary order.
        for sid in backlogged_streams.clone() {
            // Get the leaf in the hierarchy with a matching stream id
            if let Some(leaf) = leaf_classes
                .iter()
                .find(|id| self.hierarchy.class(**id).stream_id.unwrap() == sid)
            {
                pending_leaves.push_back(*leaf);
            }
        }

        // Reset the fair quotas of all internal classes.
        for id in internal_classes.clone() {
            let node = self.hierarchy.mut_class(id);
            node.fair_quota = None;
        }

        // Ancestors of an active leaf are active internal classes (excluding the root).
        for id in l_ac.clone() {
            let ancestors = self.hierarchy.ancestors(id);
            i_ac.extend(ancestors);
        }

        // The root is not an internal node.
        i_ac.remove(&root_id);

        // Determine whether to start a main or surplus round.
        let mut do_surplus = false;

        for i in i_ac.clone() {
            let balance = self.hierarchy.class(i).balance;
            let residual = self.hierarchy.class(i).residual;

            // Sum up the weight of each child k in ks.
            let child_weights_sum =
                self.weight_active_children(i, i_ac.clone(), l_ac.clone());

            // Whether to perform a surplus round.
            if balance + residual >= child_weights_sum {
                trace!("Performing surplus round: class {i} has b={balance} + r={residual} > w_ac={child_weights_sum}.");
                do_surplus = true;
                break;
            }
        }

        // Set the fair quota of the root.
        {
            let root = self.hierarchy.mut_class(root_id);

            // Update the root's balance and residual as per formula (5).
            // By adding the residual to the balance only at the start
            // of a new round, we ensure that all descendants can obtain a
            // portion of the unused balance.
            root.balance += root.residual;
            root.residual = 0;

            if do_surplus || root.balance < 0 {
                trace!(
                    "Set root's fair quota to 0 with balance {}).",
                    root.balance
                );
                root.fair_quota = Some(0);
            } else {
                root.fair_quota = None;
            }
        }

        // After initialization, the scheduler is always either in a main or surplus round.
        self.surplus_round = do_surplus;
        self.main_round = !do_surplus;

        // Reset which stream was visited last, as all active streams will have to tick at least once.
        self.i_ac = i_ac;
        self.l_ac = l_ac;
        self.pending_leaves = pending_leaves;

        trace!(
            "Streams {:?} backlogged. Starting {} round: {:?}",
            &self.pending_leaves,
            if !do_surplus { "main" } else { "surplus" },
            &self.hierarchy
        );
    }

    /// Sets the HLS hierarchy used by the scheduler in each round.
    pub fn set_hierarchy(&mut self, hierarchy: HLSHierarchy) {
        self.hierarchy = hierarchy;
    }

    /// Returns the sum of the weights of the active children of a class.
    fn weight_active_children(
        &self, i: u64, i_ac: HashSet<u64>, l_ac: HashSet<u64>,
    ) -> i64 {
        let children: HashSet<u64> = self.hierarchy.children(i);
        let union: HashSet<u64> = i_ac.union(&l_ac).copied().collect();

        let ks = children.intersection(&union);

        // Sum up the guarantee (= the global weight) of each child k in ks.
        ks.map(|k| self.hierarchy.class(*k).guarantee).sum::<i64>()
    }

    pub(crate) fn return_balance_to_parent(&mut self, class_id: u64) {
        // The root is always active and does not return the balance upstream.
        if let Some(pid) = self.hierarchy.class(class_id).parent {
            let class_balance = self.hierarchy.class(class_id).balance;

            // The class is now satisfied.
            {
                let class = self.hierarchy.mut_class(class_id);

                // Mark class as idle.
                class.idle = true;

                // Set its balance to 0, as per formula (9).
                class.balance = 0;
            }

            // Remove the class from the set of active classes. Since it is either in the internal or
            // in the active leaf set, removing it from both should be idempotent for the set it's not in.
            self.l_ac.remove(&class_id);
            self.i_ac.remove(&class_id);

            // Perform the update from formula (9) to return the balance to the parent.
            {
                let parent = self.hierarchy.mut_class(pid);
                parent.residual += class_balance;

                // If the parent itself is now idle, recursively return the balance.
                let children: HashSet<u64> = self.hierarchy.children(pid);
                let active_classes: HashSet<u64> =
                    self.l_ac.union(&self.i_ac).copied().collect();

                trace!(
                    "Returned residual of {class_balance} to the parent.{:?}",
                    &self.hierarchy
                );

                if children.intersection(&active_classes).next().is_none() {
                    self.return_balance_to_parent(pid)
                }
            }
        }
    }

    /// Uses BFS to determine the set of active streams for the current scheduling round.
    /// Returns the stream id.
    pub(crate) fn backlogged_classes_from_hierarchy(&mut self, flushable: Vec<u64>) -> Vec<u64> {
        let mut bfs_frontier: VecDeque<u64> = VecDeque::new();
        let hierarchy = &self.hierarchy;
        let root = hierarchy.root;

        let mut active_streams: Vec<u64> = Vec::new();

        // Start exploring from the root.
        bfs_frontier.push_back(root);

        // While the frontier is not empty, explore the hierarchy layer by layer.
        while !bfs_frontier.is_empty() {
            // Dequeue the first element from the frontier.
            if let Some(node) = bfs_frontier.pop_front() {
                let hierarchy = &self.hierarchy;
                let children = hierarchy.children(node);

                // Sort the children by their urgency and priority values.
                let mut priority_siblings: Vec<&HLSClass> = children
                    .into_iter()
                    .map(|c| hierarchy.class(c))
                    .collect();

                // Sort by class ID first to avoid fuzzy tests, as sets have no defined order.
                // HLS IDs should reflect the order of the requests as a tiebraker.
                priority_siblings.sort_by_key(|sibling| sibling.id);

                // Now, sort the layer by EPS priority.
                priority_siblings.sort();

                // Now, iterate over the children to prune the search tree.
                // We only include nodes at the same (and highest) urgency level.

                // The layer is already sorted. So the highest urgency is at the front.
                // But if the first node is a leaf, it needs to be in the flushable set.
                let mut max_urgency: Option<u8> = None;

                for sib in priority_siblings.clone() {
                    // Only leaf nodes have stream IDs
                    if let Some(stream_id) = sib.stream_id {
                        if flushable.contains(&stream_id) {
                            max_urgency = Some(sib.urgency);
                            break;
                        }
                    } else {
                        // Internal node. Found the highest priority.
                        max_urgency = Some(sib.urgency);
                        break;
                    }
                }

                if let Some(urgency) = max_urgency {
                    // Now, we filter the layer to only keep nodes at that urgency level.
                    let backlogged_classes: Vec<&&HLSClass> = priority_siblings
                        .iter()
                        .filter(|c| c.urgency == urgency)
                        .collect();

                    // Enqueue newly found classes in priority order.
                    for ps in backlogged_classes.iter() {
                        let id = ps.id;

                        // If the class is a leaf, mark it as an active leaf.
                        if let Some(stream_id) = ps.stream_id {
                            if flushable.contains(&stream_id) {
                                active_streams.push(stream_id);
                                // Break if the class is not incremental.
                                // This is to avoid non-incremental classes
                                // sharing bandwidth.
                                if !ps.incremental {
                                    break;
                                }
                            }
                        } else {
                            // Continue the search starting from this internal node.
                            bfs_frontier.push_back(id);
                        }
                    }
                }
            }
        }

        active_streams
    }
}
