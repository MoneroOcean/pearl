#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::slice;

use plonky2_maybe_rayon::*;
use serde::{Deserialize, Serialize};

use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::merkle_proofs::MerkleProof;
use crate::iop::challenger::Challenger;
use crate::plonk::config::{GenericHashOut, Hasher};
use crate::util::log2_strict;

/// The Merkle cap of height `h` of a Merkle tree is the `h`-th layer (from the root) of the tree.
/// It can be used in place of the root to verify Merkle paths, which are `h` elements shorter.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(bound = "")]
// TODO: Change H to GenericHashOut<F>, since this only cares about the hash, not the hasher.
pub struct MerkleCap<F: RichField, H: Hasher<F>>(pub Vec<H::Hash>);

impl<F: RichField, H: Hasher<F>> Default for MerkleCap<F, H> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<F: RichField, H: Hasher<F>> MerkleCap<F, H> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn height(&self) -> usize {
        log2_strict(self.len())
    }

    pub fn flatten(&self) -> Vec<F> {
        self.0.iter().flat_map(|&h| h.to_vec()).collect()
    }

    pub fn digest(&self) -> HashOut<F> {
        let mut challenger = Challenger::<F, H>::new();
        challenger.observe_element(F::from_canonical_usize(self.len()));
        challenger.observe_cap(self);
        challenger.get_hash()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MerkleTree<F: RichField, H: Hasher<F>> {
    /// The data in the leaves of the Merkle tree.
    pub leaves: Vec<Vec<F>>,

    /// The digests in the tree. Consists of `cap.len()` sub-trees, each corresponding to one
    /// element in `cap`. Each subtree is contiguous and located at
    /// `digests[digests.len() / cap.len() * i..digests.len() / cap.len() * (i + 1)]`.
    /// Within each subtree, siblings are stored next to each other. The layout is,
    /// left_child_subtree || left_child_digest || right_child_digest || right_child_subtree, where
    /// left_child_digest and right_child_digest are H::Hash and left_child_subtree and
    /// right_child_subtree recurse. Observe that the digest of a node is stored by its _parent_.
    /// Consequently, the digests of the roots are not stored here (they can be found in `cap`).
    pub digests: Vec<H::Hash>,

    /// The Merkle cap.
    pub cap: MerkleCap<F, H>,
}

impl<F: RichField, H: Hasher<F>> Default for MerkleTree<F, H> {
    fn default() -> Self {
        Self {
            leaves: Vec::new(),
            digests: Vec::new(),
            cap: MerkleCap::default(),
        }
    }
}

pub(crate) fn capacity_up_to_mut<T>(v: &mut Vec<T>, len: usize) -> &mut [MaybeUninit<T>] {
    assert!(v.capacity() >= len);
    let v_ptr = v.as_mut_ptr().cast::<MaybeUninit<T>>();
    unsafe {
        // SAFETY: `v_ptr` is a valid pointer to a buffer of length at least `len`. Upon return, the
        // lifetime will be bound to that of `v`. The underlying memory will not be deallocated as
        // we hold the sole mutable reference to `v`. The contents of the slice may be
        // uninitialized, but the `MaybeUninit` makes it safe.
        slice::from_raw_parts_mut(v_ptr, len)
    }
}

pub(crate) fn fill_subtree<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
) -> H::Hash {
    assert_eq!(leaves.len(), digests_buf.len() / 2 + 1);
    if digests_buf.is_empty() {
        // Base case: single leaf
        H::hash_or_noop(&leaves[0])
    } else if H::hash_batch_size() >= 8 && leaves.len() <= BATCHED_SUBTREE_LEAVES {
        fill_batched_subtree::<F, H>(digests_buf, leaves)
    } else {
        // Layout is: left recursive output || left child digest
        //             || right child digest || right recursive output.
        // Split `digests_buf` into the two recursive outputs (slices) and two child digests
        // (references).
        let (left_digests_buf, right_digests_buf) = digests_buf.split_at_mut(digests_buf.len() / 2);
        let (left_digest_mem, left_digests_buf) = left_digests_buf.split_last_mut().unwrap();
        let (right_digest_mem, right_digests_buf) = right_digests_buf.split_first_mut().unwrap();
        // Split `leaves` between both children.
        let (left_leaves, right_leaves) = leaves.split_at(leaves.len() / 2);

        let (left_digest, right_digest) = plonky2_maybe_rayon::join(
            || fill_subtree::<F, H>(left_digests_buf, left_leaves),
            || fill_subtree::<F, H>(right_digests_buf, right_leaves),
        );

        left_digest_mem.write(left_digest);
        right_digest_mem.write(right_digest);
        H::two_to_one(left_digest, right_digest)
    }
}

// Keep scratch bounded independently of the full tree, and retain parallel recursion
// above this size. Each subtree can batch both leaf and internal-node permutations.
const BATCHED_SUBTREE_LEAVES: usize = 256;

/// Build a small subtree bottom-up, writing the original interleaved layout.
fn fill_batched_subtree<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
) -> H::Hash {
    assert!(leaves.len().is_power_of_two() && leaves.len() <= BATCHED_SUBTREE_LEAVES);
    assert_eq!(digests_buf.len(), 2 * (leaves.len() - 1));
    let mut inputs: [&[F]; BATCHED_SUBTREE_LEAVES] = [&[]; BATCHED_SUBTREE_LEAVES];
    for (input, leaf) in inputs.iter_mut().zip(leaves) {
        *input = leaf;
    }
    let empty_hash = H::hash_or_noop(&[]);
    let mut hashes = [empty_hash; BATCHED_SUBTREE_LEAVES];
    let mut left = [empty_hash; BATCHED_SUBTREE_LEAVES / 2];
    let mut right = [empty_hash; BATCHED_SUBTREE_LEAVES / 2];
    H::hash_or_noop_batch(&inputs[..leaves.len()], &mut hashes[..leaves.len()]);

    let mut width = leaves.len();
    let mut level = 0;
    while width > 1 {
        let pairs = width / 2;
        for pair in 0..pairs {
            left[pair] = hashes[2 * pair];
            right[pair] = hashes[2 * pair + 1];
            let offset = 2 * ((pair << (level + 1)) + (1 << level) - 1);
            digests_buf[offset].write(left[pair]);
            digests_buf[offset + 1].write(right[pair]);
        }
        H::two_to_one_batch(&left[..pairs], &right[..pairs], &mut hashes[..pairs]);
        width = pairs;
        level += 1;
    }
    hashes[0]
}

pub(crate) fn fill_digests_buf<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
    cap_height: usize,
) {
    // Special case of a tree that's all cap. The usual case will panic because we'll try to split
    // an empty slice into chunks of `0`. (We would not need this if there was a way to split into
    // `blah` chunks as opposed to chunks _of_ `blah`.)
    if digests_buf.is_empty() {
        debug_assert_eq!(cap_buf.len(), leaves.len());
        cap_buf
            .par_iter_mut()
            .zip(leaves)
            .for_each(|(cap_buf, leaf)| {
                cap_buf.write(H::hash_or_noop(leaf));
            });
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = leaves.len() >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    let leaves_chunks = leaves.par_chunks_exact(subtree_leaves_len);
    assert_eq!(digests_chunks.len(), cap_buf.len());
    assert_eq!(digests_chunks.len(), leaves_chunks.len());
    digests_chunks.zip(cap_buf).zip(leaves_chunks).for_each(
        |((subtree_digests, subtree_cap), subtree_leaves)| {
            // We have `1 << cap_height` sub-trees, one for each entry in `cap`. They are totally
            // independent, so we schedule one task for each. `digests_buf` and `leaves` are split
            // into `1 << cap_height` slices, one for each sub-tree.
            subtree_cap.write(fill_subtree::<F, H>(subtree_digests, subtree_leaves));
        },
    );
}

pub(crate) fn merkle_tree_prove<F: RichField, H: Hasher<F>>(
    leaf_index: usize,
    leaves_len: usize,
    cap_height: usize,
    digests: &[H::Hash],
) -> Vec<H::Hash> {
    let num_layers = log2_strict(leaves_len) - cap_height;
    debug_assert_eq!(leaf_index >> (cap_height + num_layers), 0);

    let digest_len = 2 * (leaves_len - (1 << cap_height));
    assert_eq!(digest_len, digests.len());

    let digest_tree: &[H::Hash] = {
        let tree_index = leaf_index >> num_layers;
        let tree_len = digest_len >> cap_height;
        &digests[tree_len * tree_index..tree_len * (tree_index + 1)]
    };

    // Mask out high bits to get the index within the sub-tree.
    let mut pair_index = leaf_index & ((1 << num_layers) - 1);
    (0..num_layers)
        .map(|i| {
            let parity = pair_index & 1;
            pair_index >>= 1;

            // The layers' data is interleaved as follows:
            // [layer 0, layer 1, layer 0, layer 2, layer 0, layer 1, layer 0, layer 3, ...].
            // Each of the above is a pair of siblings.
            // `pair_index` is the index of the pair within layer `i`.
            // The index of that the pair within `digests` is
            // `pair_index * 2 ** (i + 1) + (2 ** i - 1)`.
            let siblings_index = (pair_index << (i + 1)) + (1 << i) - 1;
            // We have an index for the _pair_, but we want the index of the _sibling_.
            // Double the pair index to get the index of the left sibling. Conditionally add `1`
            // if we are to retrieve the right sibling.
            let sibling_index = 2 * siblings_index + (1 - parity);
            digest_tree[sibling_index]
        })
        .collect()
}

impl<F: RichField, H: Hasher<F>> MerkleTree<F, H> {
    pub fn new(leaves: Vec<Vec<F>>, cap_height: usize) -> Self {
        let log2_leaves_len = log2_strict(leaves.len());
        assert!(
            cap_height <= log2_leaves_len,
            "cap_height={cap_height} should be at most log2(leaves.len())={log2_leaves_len}"
        );

        let num_digests = 2 * (leaves.len() - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);

        let len_cap = 1 << cap_height;
        let mut cap = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        fill_digests_buf::<F, H>(digests_buf, cap_buf, &leaves[..], cap_height);

        unsafe {
            // SAFETY: `fill_digests_buf` and `cap` initialized the spare capacity up to
            // `num_digests` and `len_cap`, resp.
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }

        Self {
            leaves,
            digests,
            cap: MerkleCap(cap),
        }
    }

    pub fn get(&self, i: usize) -> &[F] {
        &self.leaves[i]
    }

    /// Create a Merkle proof from a leaf index.
    pub fn prove(&self, leaf_index: usize) -> MerkleProof<F, H> {
        let cap_height = log2_strict(self.cap.len());
        let siblings =
            merkle_tree_prove::<F, H>(leaf_index, self.leaves.len(), cap_height, &self.digests);

        MerkleProof { siblings }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use anyhow::Result;

    use super::*;
    use crate::field::extension::Extendable;
    use crate::hash::merkle_proofs::{verify_merkle_proof_to_cap, MerkleProof};
    use crate::hash::poseidon::PoseidonHash;
    use crate::plonk::config::{GenericConfig, Hasher, PoseidonGoldilocksConfig};

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct ScalarPoseidonHash;

    impl<F: RichField> Hasher<F> for ScalarPoseidonHash {
        const HASH_SIZE: usize = <PoseidonHash as Hasher<F>>::HASH_SIZE;
        type Hash = HashOut<F>;
        type Permutation = <PoseidonHash as Hasher<F>>::Permutation;

        fn hash_no_pad(input: &[F]) -> Self::Hash {
            <PoseidonHash as Hasher<F>>::hash_no_pad(input)
        }

        fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
            <PoseidonHash as Hasher<F>>::two_to_one(left, right)
        }
    }

    fn deterministic_leaves<F, L>(num_leaves: usize, leaf_len: L) -> Vec<Vec<F>>
    where
        F: RichField,
        L: Fn(usize) -> usize,
    {
        (0..num_leaves)
            .map(|leaf_idx| {
                (0..leaf_len(leaf_idx))
                    .map(|elt_idx| {
                        F::from_canonical_u64((leaf_idx * 1_009 + elt_idx * 17 + 3) as u64)
                    })
                    .collect()
            })
            .collect()
    }

    fn compare_batched_tree_with_scalar<F: RichField>(
        leaves: Vec<Vec<F>>,
        cap_height: usize,
    ) -> Result<()> {
        let batched_tree = MerkleTree::<F, PoseidonHash>::new(leaves.clone(), cap_height);
        let scalar_tree = MerkleTree::<F, ScalarPoseidonHash>::new(leaves.clone(), cap_height);

        assert_eq!(batched_tree.digests, scalar_tree.digests);
        assert_eq!(batched_tree.cap.0, scalar_tree.cap.0);
        assert_eq!(batched_tree.cap.digest(), scalar_tree.cap.digest());

        let verify_index = leaves.len() / 2;
        for (leaf_index, leaf) in leaves.into_iter().enumerate() {
            let batched_proof: MerkleProof<F, PoseidonHash> = batched_tree.prove(leaf_index);
            let scalar_proof: MerkleProof<F, ScalarPoseidonHash> = scalar_tree.prove(leaf_index);
            assert_eq!(batched_proof.siblings, scalar_proof.siblings);

            if leaf_index == verify_index {
                // This verifier hashes the leaf through the scalar `hash_or_noop` path.
                verify_merkle_proof_to_cap::<F, PoseidonHash>(
                    leaf,
                    leaf_index,
                    &batched_tree.cap,
                    &batched_proof,
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn random_data<F: RichField>(n: usize, k: usize) -> Vec<Vec<F>> {
        (0..n).map(|_| F::rand_vec(k)).collect()
    }

    fn verify_all_leaves<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        leaves: Vec<Vec<F>>,
        cap_height: usize,
    ) -> Result<()> {
        let tree = MerkleTree::<F, C::Hasher>::new(leaves.clone(), cap_height);
        for (i, leaf) in leaves.into_iter().enumerate() {
            let proof = tree.prove(i);
            verify_merkle_proof_to_cap(leaf, i, &tree.cap, &proof)?;
        }
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_cap_height_too_big() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let cap_height = log_n + 1; // Should panic if `cap_height > len_n`.

        let leaves = random_data::<F>(1 << log_n, 7);
        let _ = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new(leaves, cap_height);
    }

    #[test]
    fn test_cap_height_eq_log2_len() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, log_n)?;

        Ok(())
    }

    #[test]
    fn test_merkle_trees() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, 1)?;

        Ok(())
    }

    #[test]
    fn batched_poseidon_merkle_tree_matches_scalar_hashing() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        const LEAF_LENGTHS: [usize; 9] = [0, 1, 2, 3, 4, 7, 8, 9, 64];

        for log_num_leaves in 0..=6 {
            let num_leaves = 1 << log_num_leaves;
            for leaf_len in LEAF_LENGTHS {
                let leaves = deterministic_leaves::<F, _>(num_leaves, |_| leaf_len);
                for cap_height in 0..=log_num_leaves {
                    compare_batched_tree_with_scalar(leaves.clone(), cap_height)?;
                }
            }

            let leaves = deterministic_leaves::<F, _>(num_leaves, |leaf_idx| 5 + leaf_idx % 5);
            for cap_height in 0..=log_num_leaves {
                compare_batched_tree_with_scalar(leaves.clone(), cap_height)?;
            }
        }
        Ok(())
    }

    #[test]
    fn bottom_up_subtree_matches_recursive_layout() {
        use crate::field::goldilocks_field::GoldilocksField as F;
        use crate::field::types::Field;

        // Exercise the bottom-up layout even on targets where SIMD is disabled.
        for log_num_leaves in 0..=8 {
            for leaf_len in [0, 4, 9, 65] {
                let mut leaves = deterministic_leaves::<F, _>(1 << log_num_leaves, |_| leaf_len);
                for (idx, leaf) in leaves.iter_mut().enumerate() {
                    if let Some(first) = leaf.first_mut() {
                        *first = F::from_noncanonical_u64(u64::MAX - idx as u64);
                    }
                }
                let expected = MerkleTree::<F, ScalarPoseidonHash>::new(leaves.clone(), 0);
                let mut output = vec![MaybeUninit::uninit(); 2 * (leaves.len() - 1)];
                let root = fill_batched_subtree::<F, PoseidonHash>(&mut output, &leaves);
                assert_eq!(root, expected.cap.0[0]);
                for (actual, expected) in output.into_iter().zip(expected.digests) {
                    // SAFETY: the helper writes every non-root digest exactly once.
                    assert_eq!(unsafe { actual.assume_init() }, expected);
                }
            }
        }
    }

    #[test]
    fn batched_merkle_subtree_boundaries_match_scalar() -> Result<()> {
        use crate::field::goldilocks_field::GoldilocksField as F;
        use crate::field::types::Field;

        for log_num_leaves in [7, 8, 9, 10] {
            let mut leaves = deterministic_leaves::<F, _>(1 << log_num_leaves, |_| 9);
            for (idx, leaf) in leaves.iter_mut().enumerate() {
                leaf[0] = F::from_noncanonical_u64(u64::MAX - idx as u64);
            }
            for cap_height in [0, 1, 2, log_num_leaves - 1, log_num_leaves] {
                compare_batched_tree_with_scalar(leaves.clone(), cap_height)?;
            }
        }
        Ok(())
    }
}
