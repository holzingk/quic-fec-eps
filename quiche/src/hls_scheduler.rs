use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::fmt::Formatter;
/// Implementation of the Hierarchical Link Sharing (HLS) Scheduling algorithm.
#[derive(Clone, Debug)]
pub(crate) struct HLSClass {
    /// Unique identifier of the class.
    id: u64,

    /// Parent class of the class.
    parent: Option<u64>,

    /// Set of children of the class.
    children: HashSet<u64>,

    /// The class' weight.
    pub(crate) weight: i64,

    /// Number of bytes the class is allowed to transmit.
    pub(crate) balance: i64,

    /// Permits for the transmission of bytes. Collected from descendants.
    pub(crate) residual: i64,

    /// Number of bytes that a child class with weight set to one can transmit.
    /// `None` if not yet computed.
    fair_quota: Option<i64>,

    /// The stream id of this class. `None` for classes that are not leaves.
    pub(crate) stream_id: Option<u64>,

    /// Whether the class became idle during a round.
    idle: bool,

    /// Whether the class' balance has already been updated.
    pub ticked: bool,

    /// How many bytes this stream has emitted in the current round.
    pub(crate) emitted: i64,

    /// SA-ECF: indicates whether the stream is waiting for a faster path to become available.
    pub waiting: u128,
}

/// Represents a class hierarchy in the context of the HLS paper.
pub struct HLSHierarchy {
    /// Map of class identifiers to class objects.
    pub(crate) classes: HashMap<u64, HLSClass>,

    /// Identifier of the root class.
    pub(crate) root: u64,

    /// Next class identifier to be assigned.
    next_id: u64,
}

impl HLSClass {
    pub fn new(id: u64, parent: Option<u64>, weight: i64) -> HLSClass {
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
            waiting: 0,
            ticked: false,
        }
    }
}

impl HLSHierarchy {
    /// Creates a hierarchy consisting only of the root node.
    pub fn new() -> HLSHierarchy {
        let root = 0;
        HLSHierarchy {
            classes: HashMap::new(),
            root,
            next_id: root,
        }
    }

    /// Returns a reference to the class with the given identifier.
    pub(crate) fn class(&self, class_id: u64) -> &HLSClass {
        self.classes.get(&class_id).unwrap()
    }

    /// Returns a mutable reference to the class with the given identifier.
    pub(crate) fn mut_class(&mut self, class_id: u64) -> &mut HLSClass {
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
    pub fn insert(&mut self, weight: i64, parent: Option<u64>) -> u64 {
        let id = self.next_id();
        let guarantee = weight;
        let class = HLSClass::new(id, parent, guarantee);

        // If the class has a parent, update the parent's children
        if let Some(pid) = parent {
            if let Some(parent) = self.classes.get_mut(&pid) {
                parent.children.insert(id);
            }
        }

        self.classes.insert(id, class);
        id
    }

    fn children(&self, node_id: u64) -> HashSet<u64> {
        if let Some(node) = self.classes.get(&node_id) {
            node.children.clone()
        } else {
            HashSet::new()
        }
    }

    pub(crate) fn leaf_descendants(&self, node_id: u64) -> HashSet<u64> {
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

    pub(crate) fn ancestors(&self, node_id: u64) -> HashSet<u64> {
        let mut ancestors: HashSet<u64> = HashSet::new();
        let mut current_id = node_id;

        while let Some(parent_id) = self.class(current_id).parent {
            ancestors.insert(parent_id);
            current_id = parent_id;
        }

        ancestors
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

impl Default for HLSHierarchy {
    fn default() -> Self {
        let mut hierarchy = HLSHierarchy::new();
        let max_streams: u64 = 50;
        let root = hierarchy.insert((max_streams * 1_000) as i64, None);

        // This default range is due to Quiche's tests using stream ids between 0-50.
        for sid in 0..max_streams {
            let class = hierarchy.insert(1_000, Some(root));
            hierarchy.set_stream_id(class, sid);
        }

        hierarchy
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
                "{}{}: balance: {}, weight: {}, {}residual: {}",
                current_prefix,
                match class.stream_id {
                    Some(stream_id) => format!("stream {}", stream_id),
                    None => class.id.to_string(),
                },
                class.balance,
                class.weight,
                if class.fair_quota.is_some() {
                    format!("fair quota: {}, ", class.fair_quota.unwrap())
                } else if !class.children.is_empty() {
                    "fair quota: None, ".to_string()
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
    l_ac: HashSet<u64>,

    /// Set of active internal classes.
    i_ac: HashSet<u64>,

    /// Total number of bytes from all classes that can be transmitted in a round.
    pub(crate) q: i64,

    /// The class hierarchy.
    pub(crate) hierarchy: HLSHierarchy,

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
            q: 0,
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
        let leaf_weight = self.hierarchy.class(class_id).weight;

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
            let class_weight = self.hierarchy.class(class_id).weight;
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

        // Sum of the weights of each class k in ks
        let sum_weights =
            ks.map(|k| self.hierarchy.class(*k).weight).sum::<i64>();

        let node = self.hierarchy.mut_class(class_id);

        // Calculate fair quota as in formula (4).
        let fair_quota = node.balance / sum_weights;

        // Set the class' fair quota
        node.fair_quota = Some(fair_quota);

        // Return the fair quota that was just calculated
        fair_quota
    }

    // "The total weights of all non-root active classes and the total maximum packet sizes of
    // active leaf classes determine the amount of quota allocated to each class,
    // which dictates the round size."
    fn calculate_q(
        &self, leaf_classes: HashSet<u64>, active_leaves: HashSet<u64>,
        internal_classes: HashSet<u64>,
    ) -> i64 {
        // Sum of the weights of the leaf classes
        let sum_leaf_weights = leaf_classes
            .iter()
            .map(|id| self.hierarchy.class(*id).weight)
            .sum::<i64>();

        // Sum of the weights of the internal classes
        let sum_internal_weights = internal_classes
            .iter()
            .map(|id| self.hierarchy.class(*id).weight)
            .sum::<i64>();

        let sum_active_leaf_weights = active_leaves
            .iter()
            .map(|id| self.hierarchy.class(*id).weight)
            .sum::<i64>();

        let constant = sum_internal_weights + sum_leaf_weights;
        let dynamic = sum_active_leaf_weights;

        let q = constant + dynamic;

        trace!(
            r#"{{"q":{},"constant":{}, "dynamic":{}}}"#,
            q,
            constant,
            dynamic
        );

        q
    }

    /// Prepares the round-robin scheduler for a new main round.
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
            let q = self.calculate_q(
                leaf_classes.clone(),
                HashSet::new(),
                internal_classes.clone(),
            );

            {
                let root_id = self.hierarchy.root;
                let root = self.hierarchy.mut_class(root_id);
                root.balance = q;
            }
        }

        // Determine which leaf classes are active. Recomputed every round, i.e., main or surplus.
        for leaf in leaf_classes.clone() {
            let stream_id = self.hierarchy.class(leaf).stream_id.unwrap();
            let became_idle = self.hierarchy.class(leaf).idle;
            let leaf_weight = self.hierarchy.class(leaf).weight;

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
        // Set the round-robin order to match Quiche's priorities instead.
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

        // Dynamically adjust Q* at the start of the round now that we know which leaves are active.
        let q = self.calculate_q(
            leaf_classes.clone(),
            l_ac.clone(),
            internal_classes.clone(),
        );
        self.q = q;

        // After initialization, the scheduler is always either in a main or surplus round.
        self.surplus_round = do_surplus;
        self.main_round = !do_surplus;

        // Reset which stream was visited last, as all active streams will have to tick at least once.
        self.i_ac = i_ac;
        self.l_ac = l_ac;
        self.pending_leaves = pending_leaves;

        trace!(
            "Streams {:?} backlogged. Starting {} round: {:?}",
            &backlogged_streams,
            if !do_surplus { "main" } else { "surplus" },
            &self.hierarchy
        );
    }

    /// Sets the HLS hierarchy used by the scheduler in each round.
    pub fn set_hierarchy(&mut self, hierarchy: HLSHierarchy) {
        self.hierarchy = hierarchy;
    }

    /// Ensures that the updated balance counters follow the invariant specified in the HLS paper.
    /// This check is performed after applying the updates outlined by formulas (5) through (9).
    pub(crate) fn hls_invariant_holds(&self) -> bool {
        let sum_balances = self
            .hierarchy
            .classes
            .values()
            .map(|c| c.balance)
            .sum::<i64>();

        let root_id = self.hierarchy.root;
        let root_residual = self.hierarchy.class(root_id).residual;
        let internal_classes = self.hierarchy.internal_nodes(root_id);

        // Sum up the residual of each internal class
        let sum_residuals = internal_classes
            .iter()
            .map(|c| self.hierarchy.class(*c).residual)
            .sum::<i64>();

        let invariant = sum_balances + sum_residuals + root_residual;

        trace!(
            "Sum of balances and residuals of {} should match Q*={}",
            invariant,
            self.q
        );
        invariant == self.q
    }

    /// Returns the sum of the weights of the active children of a class.
    fn weight_active_children(
        &self, i: u64, i_ac: HashSet<u64>, l_ac: HashSet<u64>,
    ) -> i64 {
        let children: HashSet<u64> = self.hierarchy.children(i);
        let union: HashSet<u64> = i_ac.union(&l_ac).copied().collect();

        let ks = children.intersection(&union);

        // Sum up the weight of each child k in ks.
        ks.map(|k| self.hierarchy.class(*k).weight).sum::<i64>()
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
}
