#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::hash::hash_types::{BytesHash, RichField};
use crate::hash::hashing::PlonkyPermutation;
use crate::plonk::config::Hasher;
use crate::util::serialization::Write;

pub const SPONGE_RATE: usize = 8;
pub const SPONGE_CAPACITY: usize = 4;
pub const SPONGE_WIDTH: usize = SPONGE_RATE + SPONGE_CAPACITY;

/// Blake3 pseudo-permutation used in the challenger.
/// Hashes the state using Blake3 XOF mode and fills output via rejection sampling.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct Blake3Permutation<F: RichField> {
    state: [F; SPONGE_WIDTH],
}

impl<F: RichField> AsRef<[F]> for Blake3Permutation<F> {
    fn as_ref(&self) -> &[F] {
        &self.state
    }
}

/// Similar to KeccakPermutation, but without the property that squeeze() determines the entire state.
impl<F: RichField> PlonkyPermutation<F> for Blake3Permutation<F> {
    const RATE: usize = SPONGE_RATE;
    const WIDTH: usize = SPONGE_WIDTH;

    fn new<I: IntoIterator<Item = F>>(elts: I) -> Self {
        let mut perm = Self {
            state: [F::default(); SPONGE_WIDTH],
        };
        perm.set_from_iter(elts, 0);
        perm
    }

    fn set_elt(&mut self, elt: F, idx: usize) {
        self.state[idx] = elt;
    }

    fn set_from_slice(&mut self, elts: &[F], start_idx: usize) {
        self.state[start_idx..start_idx + elts.len()].copy_from_slice(elts);
    }

    fn set_from_iter<I: IntoIterator<Item = F>>(&mut self, elts: I, start_idx: usize) {
        for (s, e) in self.state[start_idx..].iter_mut().zip(elts) {
            *s = e;
        }
    }

    fn permute(&mut self) {
        self.permute_n::<{ SPONGE_WIDTH }>();
    }

    fn permute_n<const N: usize>(&mut self) {
        debug_assert_eq!(F::BITS, 64);
        // Serialize state to bytes
        let mut state_bytes = [0u8; SPONGE_WIDTH * 8];
        for (chunk, field) in state_bytes.chunks_exact_mut(8).zip(&self.state) {
            chunk.copy_from_slice(&field.to_canonical_u64().to_le_bytes());
        }

        let mut reader = blake3::Hasher::new().update(&state_bytes).finalize_xof();
        let mut idx = 0;
        let mut buf = [0u8; 64];
        while idx < N {
            reader.fill(&mut buf);
            for chunk in buf.chunks_exact(8) {
                let word = u64::from_le_bytes(chunk.try_into().unwrap());
                if word < F::ORDER {
                    self.state[idx] = F::from_canonical_u64(word);
                    idx += 1;
                    if idx == N {
                        return;
                    }
                }
            }
        }
    }

    fn squeeze(&self) -> &[F] {
        &self.state[..Self::RATE]
    }

    fn find_pow_witness(
        &self,
        candidates: core::ops::Range<u64>,
        witness_input_pos: usize,
        min_leading_zeros: u32,
    ) -> Option<u64> {
        use crate::hash::hashing::find_pow_witness_scalar;

        assert!(witness_input_pos < SPONGE_WIDTH);
        assert!(candidates.start <= candidates.end && candidates.end <= F::ORDER);
        #[allow(unused_mut)]
        let mut start = candidates.start;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            while candidates.end - start >= 16 {
                let responses = crate::hash::blake3_batch_avx512::pow_responses(
                    &self.state,
                    witness_input_pos,
                    start,
                );
                for (lane, response) in responses.into_iter().enumerate() {
                    let candidate = start + lane as u64;
                    match response {
                        Some(value) if value.leading_zeros() >= min_leading_zeros => {
                            return Some(candidate);
                        }
                        None => {
                            if find_pow_witness_scalar(
                                self,
                                candidate..candidate + 1,
                                witness_input_pos,
                                min_leading_zeros,
                            )
                            .is_some()
                            {
                                return Some(candidate);
                            }
                        }
                        _ => {}
                    }
                }
                start += 16;
            }
        }
        find_pow_witness_scalar(
            self,
            start..candidates.end,
            witness_input_pos,
            min_leading_zeros,
        )
    }
}

/// Blake3 hash function.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Blake3Hash<const N: usize>;

impl<F: RichField, const N: usize> Hasher<F> for Blake3Hash<N> {
    const HASH_SIZE: usize = N;
    type Hash = BytesHash<N>;
    type Permutation = Blake3Permutation<F>;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        let mut buffer = Vec::with_capacity(input.len() * F::BITS.div_ceil(8));
        buffer.write_field_vec(input).unwrap();
        BytesHash(blake3::hash(&buffer).as_bytes()[..N].try_into().unwrap())
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&left.0);
        hasher.update(&right.0);
        BytesHash(hasher.finalize().as_bytes()[..N].try_into().unwrap())
    }

    fn hash_batch_size() -> usize {
        if cfg!(all(target_arch = "x86_64", target_feature = "avx512f")) && N <= 32 {
            16
        } else {
            1
        }
    }

    fn hash_or_noop_batch(inputs: &[&[F]], outputs: &mut [Self::Hash]) {
        assert_eq!(inputs.len(), outputs.len());
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            if N <= 32 {
                crate::hash::blake3_batch_avx512::hash_or_noop_batch(inputs, outputs);
                return;
            }
        }
        for (input, output) in inputs.iter().zip(outputs) {
            *output = Self::hash_or_noop(input);
        }
    }

    fn two_to_one_batch(left: &[Self::Hash], right: &[Self::Hash], outputs: &mut [Self::Hash]) {
        assert_eq!(left.len(), right.len());
        assert_eq!(left.len(), outputs.len());
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            if N <= 32 {
                crate::hash::blake3_batch_avx512::two_to_one_batch(left, right, outputs);
                return;
            }
        }
        for ((&left, &right), output) in left.iter().zip(right).zip(outputs) {
            *output = <Self as Hasher<F>>::two_to_one(left, right);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField as F;
    use crate::field::types::{Field, PrimeField64};
    use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
    use crate::hash::merkle_tree::MerkleTree;

    fn leaves(count: usize, len: usize) -> Vec<Vec<F>> {
        (0..count)
            .map(|lane| {
                (0..len)
                    .map(|i| {
                        let value = if (i + lane) % 7 == 0 {
                            u64::MAX - (i + lane) as u64
                        } else {
                            (i as u64)
                                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                                .rotate_left(lane as u32)
                                ^ (lane as u64).wrapping_mul(0xd1b5_4a32_d192_ed03)
                        };
                        F::from_noncanonical_u64(value)
                    })
                    .collect()
            })
            .collect()
    }

    fn oracle<const N: usize>(input: &[F]) -> BytesHash<N> {
        let bytes: Vec<u8> = input
            .iter()
            .flat_map(|f| f.to_canonical_u64().to_le_bytes())
            .collect();
        let mut out = [0u8; N];
        if bytes.len() <= N {
            out[..bytes.len()].copy_from_slice(&bytes);
        } else {
            out.copy_from_slice(&blake3::hash(&bytes).as_bytes()[..N]);
        }
        BytesHash(out)
    }

    fn assert_leaf_batches<const N: usize>() {
        for count in [0, 1, 15, 16, 17, 32] {
            for len in [
                0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 127, 128, 129, 255, 256, 257,
            ] {
                let storage = leaves(count, len);
                let inputs: Vec<&[F]> = storage.iter().map(Vec::as_slice).collect();
                let mut outputs = vec![BytesHash([0u8; N]); count];
                <Blake3Hash<N> as Hasher<F>>::hash_or_noop_batch(&inputs, &mut outputs);
                for idx in 0..count {
                    assert_eq!(
                        outputs[idx],
                        oracle::<N>(inputs[idx]),
                        "N={N},count={count},len={len},lane={idx}"
                    );
                }
            }
        }
    }

    #[test]
    fn blake3_leaf_batches_match_crate_at_block_and_chunk_boundaries() {
        assert_leaf_batches::<27>();
        assert_leaf_batches::<32>();
        assert_leaf_batches::<1>();
        assert_leaf_batches::<0>();
    }

    #[test]
    fn blake3_unequal_leaf_batches_match_crate() {
        let storage: Vec<Vec<F>> = (0..32).map(|i| leaves(1, 120 + i).pop().unwrap()).collect();
        let inputs: Vec<&[F]> = storage.iter().map(Vec::as_slice).collect();
        let mut outputs = vec![BytesHash([0u8; 27]); inputs.len()];
        <Blake3Hash<27> as Hasher<F>>::hash_or_noop_batch(&inputs, &mut outputs);
        for (input, actual) in inputs.iter().zip(outputs) {
            assert_eq!(actual, oracle::<27>(input));
        }
    }

    fn assert_pair_batches<const N: usize>() {
        for count in [0, 1, 15, 16, 17, 32] {
            let left: Vec<_> = (0..count)
                .map(|lane| {
                    BytesHash(core::array::from_fn(|i| {
                        (lane as u8).wrapping_mul(31).wrapping_add(i as u8)
                    }))
                })
                .collect();
            let right: Vec<_> = (0..count)
                .map(|lane| {
                    BytesHash(core::array::from_fn(|i| {
                        (lane as u8)
                            .wrapping_mul(19)
                            .wrapping_sub((i as u8).wrapping_mul(11))
                    }))
                })
                .collect();
            let mut outputs = vec![BytesHash([0u8; N]); count];
            <Blake3Hash<N> as Hasher<F>>::two_to_one_batch(&left, &right, &mut outputs);
            for lane in 0..count {
                let mut bytes = Vec::with_capacity(2 * N);
                bytes.extend_from_slice(&left[lane].0);
                bytes.extend_from_slice(&right[lane].0);
                assert_eq!(
                    &outputs[lane].0,
                    &blake3::hash(&bytes).as_bytes()[..N],
                    "N={N},count={count},lane={lane}"
                );
            }
        }
    }

    #[test]
    fn blake3_compression_batches_match_crate_for_full_and_truncated_hashes() {
        assert_pair_batches::<0>();
        assert_pair_batches::<1>();
        assert_pair_batches::<27>();
        assert_pair_batches::<31>();
        assert_pair_batches::<32>();
    }

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct ScalarBlake3;

    impl Hasher<F> for ScalarBlake3 {
        const HASH_SIZE: usize = 27;
        type Hash = BytesHash<27>;
        type Permutation = Blake3Permutation<F>;

        fn hash_no_pad(input: &[F]) -> Self::Hash {
            <Blake3Hash<27> as Hasher<F>>::hash_no_pad(input)
        }

        fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
            <Blake3Hash<27> as Hasher<F>>::two_to_one(left, right)
        }
    }

    #[test]
    fn blake3_batched_merkle_roots_layout_and_proofs_match_scalar() {
        for len in [3, 9, 129, 257] {
            let storage = leaves(512, len);
            for cap_height in [0, 1, 8, 9] {
                let batched = MerkleTree::<F, Blake3Hash<27>>::new(storage.clone(), cap_height);
                let scalar = MerkleTree::<F, ScalarBlake3>::new(storage.clone(), cap_height);
                assert_eq!(batched.cap.0, scalar.cap.0);
                assert_eq!(batched.digests, scalar.digests);
                for idx in 0..storage.len() {
                    assert_eq!(batched.prove(idx).siblings, scalar.prove(idx).siblings);
                }
                let idx = 257;
                verify_merkle_proof_to_cap::<F, Blake3Hash<27>>(
                    storage[idx].clone(),
                    idx,
                    &batched.cap,
                    &batched.prove(idx),
                )
                .unwrap();
            }
        }
    }
}
