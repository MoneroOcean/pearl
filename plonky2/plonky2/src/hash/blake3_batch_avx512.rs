//! Sixteen independent ordinary BLAKE3 hashes, using one message per SIMD lane.
//! The compression schedule and flags follow the BLAKE3 specification. Inputs
//! retain the canonical field serialization used by `Blake3Hash`.

use core::arch::x86_64::*;

use crate::hash::blake3_perm::Blake3Hash;
use crate::hash::hash_types::{BytesHash, RichField};
use crate::plonk::config::Hasher;

const LANES: usize = 16;
const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const PARENT: u32 = 4;
const ROOT: u32 = 8;
const SCHEDULE: [[usize; 16]; 7] = {
    let permutation = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
    let mut rounds = [[0; 16]; 7];
    let mut i = 0;
    while i < 16 {
        rounds[0][i] = i;
        i += 1;
    }
    let mut round = 1;
    while round < 7 {
        i = 0;
        while i < 16 {
            rounds[round][i] = rounds[round - 1][permutation[i]];
            i += 1;
        }
        round += 1;
    }
    rounds
};

#[inline(always)]
unsafe fn mix<const A: usize, const B: usize, const C: usize, const D: usize>(
    v: &mut [__m512i; 16],
    x: __m512i,
    y: __m512i,
) {
    v[A] = _mm512_add_epi32(_mm512_add_epi32(v[A], v[B]), x);
    v[D] = _mm512_ror_epi32::<16>(_mm512_xor_si512(v[D], v[A]));
    v[C] = _mm512_add_epi32(v[C], v[D]);
    v[B] = _mm512_ror_epi32::<12>(_mm512_xor_si512(v[B], v[C]));
    v[A] = _mm512_add_epi32(_mm512_add_epi32(v[A], v[B]), y);
    v[D] = _mm512_ror_epi32::<8>(_mm512_xor_si512(v[D], v[A]));
    v[C] = _mm512_add_epi32(v[C], v[D]);
    v[B] = _mm512_ror_epi32::<7>(_mm512_xor_si512(v[B], v[C]));
}

#[inline(always)]
#[unroll::unroll_for_loops]
unsafe fn compress_xof(
    cv: [__m512i; 8],
    message: [__m512i; 16],
    counter: u32,
    len: u32,
    flags: u32,
) -> [__m512i; 16] {
    let mut v = [_mm512_setzero_si512(); 16];
    v[..8].copy_from_slice(&cv);
    for i in 0..4 {
        v[i + 8] = _mm512_set1_epi32(IV[i] as i32);
    }
    v[12] = _mm512_set1_epi32(counter as i32);
    v[13] = _mm512_setzero_si512();
    v[14] = _mm512_set1_epi32(len as i32);
    v[15] = _mm512_set1_epi32(flags as i32);
    for r in 0..7 {
        let s = SCHEDULE[r];
        mix::<0, 4, 8, 12>(&mut v, message[s[0]], message[s[1]]);
        mix::<1, 5, 9, 13>(&mut v, message[s[2]], message[s[3]]);
        mix::<2, 6, 10, 14>(&mut v, message[s[4]], message[s[5]]);
        mix::<3, 7, 11, 15>(&mut v, message[s[6]], message[s[7]]);
        mix::<0, 5, 10, 15>(&mut v, message[s[8]], message[s[9]]);
        mix::<1, 6, 11, 12>(&mut v, message[s[10]], message[s[11]]);
        mix::<2, 7, 8, 13>(&mut v, message[s[12]], message[s[13]]);
        mix::<3, 4, 9, 14>(&mut v, message[s[14]], message[s[15]]);
    }
    core::array::from_fn(|i| {
        if i < 8 {
            _mm512_xor_si512(v[i], v[i + 8])
        } else {
            _mm512_xor_si512(v[i], cv[i - 8])
        }
    })
}

#[inline(always)]
unsafe fn compress(
    cv: [__m512i; 8],
    message: [__m512i; 16],
    counter: u32,
    len: u32,
    flags: u32,
) -> [__m512i; 8] {
    let output = compress_xof(cv, message, counter, len, flags);
    core::array::from_fn(|i| output[i])
}

#[inline(always)]
unsafe fn initial_cv() -> [__m512i; 8] {
    IV.map(|word| _mm512_set1_epi32(word as i32))
}

#[inline(always)]
unsafe fn load_words(words: &[[u32; LANES]; 16]) -> [__m512i; 16] {
    core::array::from_fn(|i| _mm512_loadu_si512(words[i].as_ptr().cast()))
}

/// A rejection in any of the first eight words changes which word becomes the
/// eighth accepted field element, so that lane must use the ordinary sampler.
fn accepted_pow_response<F: RichField>(words: [u64; 8]) -> Option<u64> {
    words
        .iter()
        .all(|&word| word < F::ORDER)
        .then_some(words[7])
}

/// First eight XOF words for sixteen candidate states. `None` means the caller
/// must repeat that lane through the scalar permutation's rejection sampler.
pub(crate) fn pow_responses<F: RichField>(
    input: &[F; 12],
    witness_input_pos: usize,
    start: u64,
) -> [Option<u64>; LANES] {
    assert!(witness_input_pos < 12);
    assert!(start <= F::ORDER - LANES as u64);
    let mut first = [[0u32; LANES]; 16];
    let mut last = [[0u32; LANES]; 16];
    for field in 0..12 {
        for lane in 0..LANES {
            let word = if field == witness_input_pos {
                start + lane as u64
            } else {
                input[field].to_canonical_u64()
            };
            let block = if field < 8 { &mut first } else { &mut last };
            let index = (field % 8) * 2;
            block[index][lane] = word as u32;
            block[index + 1][lane] = (word >> 32) as u32;
        }
    }
    let mut words = [[0u32; LANES]; 16];
    // SAFETY: sixteen canonical candidates and two fixed-length input blocks;
    // AVX-512 is a compile-time requirement of this module.
    unsafe {
        let cv = compress(initial_cv(), load_words(&first), 0, 64, CHUNK_START);
        let output = compress_xof(cv, load_words(&last), 0, 32, CHUNK_END | ROOT);
        for i in 0..16 {
            _mm512_storeu_si512(words[i].as_mut_ptr().cast(), output[i]);
        }
    }
    core::array::from_fn(|lane| {
        accepted_pow_response::<F>(core::array::from_fn(|i| {
            words[2 * i][lane] as u64 | ((words[2 * i + 1][lane] as u64) << 32)
        }))
    })
}

/// Hash a chunk's canonical field bytes, retaining its chaining value when it
/// is one of two chunks and producing root output when it is the only chunk.
unsafe fn field_chunk<F: RichField>(
    inputs: &[&[F]],
    field_start: usize,
    field_len: usize,
    chunk_counter: u32,
    root: bool,
) -> [__m512i; 8] {
    let mut cv = initial_cv();
    let blocks = field_len.div_ceil(8).max(1);
    for block in 0..blocks {
        let block_fields = (field_len - block * 8).min(8);
        let mut words = [[0u32; LANES]; 16];
        for lane in 0..LANES {
            for field in 0..block_fields {
                let word = inputs[lane][field_start + block * 8 + field].to_canonical_u64();
                words[field * 2][lane] = word as u32;
                words[field * 2 + 1][lane] = (word >> 32) as u32;
            }
        }
        let mut flags = if block == 0 { CHUNK_START } else { 0 };
        if block + 1 == blocks {
            flags |= CHUNK_END;
            if root {
                flags |= ROOT;
            }
        }
        cv = compress(
            cv,
            load_words(&words),
            chunk_counter,
            (block_fields * 8) as u32,
            flags,
        );
    }
    cv
}

unsafe fn store_hashes<const N: usize>(cv: [__m512i; 8], outputs: &mut [BytesHash<N>]) {
    let mut words = [[0u32; LANES]; 8];
    for i in 0..8 {
        _mm512_storeu_si512(words[i].as_mut_ptr().cast(), cv[i]);
    }
    for lane in 0..LANES {
        for byte in 0..N {
            outputs[lane].0[byte] = (words[byte / 4][lane] >> ((byte % 4) * 8)) as u8;
        }
    }
}

pub(crate) fn hash_or_noop_batch<F: RichField, const N: usize>(
    inputs: &[&[F]],
    outputs: &mut [BytesHash<N>],
) {
    assert_eq!(inputs.len(), outputs.len());
    assert!(N <= 32);
    let Some(first) = inputs.first() else { return };
    let len = first.len();
    let supported = len <= 256 && len * 8 > N && inputs.iter().all(|input| input.len() == len);
    let packed_len = if supported {
        inputs.len() / LANES * LANES
    } else {
        0
    };
    for start in (0..packed_len).step_by(LANES) {
        let group = &inputs[start..start + LANES];
        // SAFETY: this module is compiled only with AVX-512 enabled. Every
        // group has sixteen equal-length slices, and at most two chunks.
        unsafe {
            let cv = if len <= 128 {
                field_chunk(group, 0, len, 0, true)
            } else {
                let left = field_chunk(group, 0, 128, 0, false);
                let right = field_chunk(group, 128, len - 128, 1, false);
                let mut parent = [_mm512_setzero_si512(); 16];
                parent[..8].copy_from_slice(&left);
                parent[8..].copy_from_slice(&right);
                compress(initial_cv(), parent, 0, 64, PARENT | ROOT)
            };
            store_hashes(cv, &mut outputs[start..start + LANES]);
        }
    }
    for (input, output) in inputs[packed_len..].iter().zip(&mut outputs[packed_len..]) {
        *output = <Blake3Hash<N> as Hasher<F>>::hash_or_noop(input);
    }
}

pub(crate) fn two_to_one_batch<const N: usize>(
    left: &[BytesHash<N>],
    right: &[BytesHash<N>],
    outputs: &mut [BytesHash<N>],
) {
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), outputs.len());
    assert!(N <= 32);
    let packed_len = left.len() / LANES * LANES;
    for start in (0..packed_len).step_by(LANES) {
        let mut words = [[0u32; LANES]; 16];
        for lane in 0..LANES {
            for byte in 0..2 * N {
                let value = if byte < N {
                    left[start + lane].0[byte]
                } else {
                    right[start + lane].0[byte - N]
                };
                words[byte / 4][lane] |= (value as u32) << ((byte % 4) * 8);
            }
        }
        // Merkle children are simply concatenated bytes, not BLAKE3 subtree
        // chaining values. Their ordinary short-message hash uses chunk flags.
        unsafe {
            let cv = compress(
                initial_cv(),
                load_words(&words),
                0,
                (2 * N) as u32,
                CHUNK_START | CHUNK_END | ROOT,
            );
            store_hashes(cv, &mut outputs[start..start + LANES]);
        }
    }
    for idx in packed_len..left.len() {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&left[idx].0).update(&right[idx].0);
        outputs[idx]
            .0
            .copy_from_slice(&hasher.finalize().as_bytes()[..N]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField as F;
    use crate::field::types::{Field, Field64, PrimeField64};

    #[test]
    fn blake3_pow_xof_matches_crate_for_every_state_position() {
        for seed in 0..4u64 {
            let input: [F; 12] = core::array::from_fn(|i| {
                F::from_noncanonical_u64(u64::MAX.wrapping_sub((i as u64 + seed) * 977))
            });
            for position in 0..12 {
                for start in [seed * 19, F::ORDER - 16] {
                    let actual = pow_responses(&input, position, start);
                    for lane in 0..16 {
                        let mut bytes = [0u8; 96];
                        for (i, field) in input.iter().enumerate() {
                            let word = if i == position {
                                start + lane as u64
                            } else {
                                field.to_canonical_u64()
                            };
                            bytes[8 * i..8 * i + 8].copy_from_slice(&word.to_le_bytes());
                        }
                        let mut xof = [0u8; 64];
                        blake3::Hasher::new()
                            .update(&bytes)
                            .finalize_xof()
                            .fill(&mut xof);
                        let words = core::array::from_fn(|i| {
                            u64::from_le_bytes(xof[8 * i..8 * i + 8].try_into().unwrap())
                        });
                        assert_eq!(
                            actual[lane],
                            accepted_pow_response::<F>(words),
                            "seed={seed}, position={position}, start={start}, lane={lane}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn blake3_pow_rejection_is_detected_in_every_output_position() {
        assert_eq!(
            accepted_pow_response::<F>([F::ORDER - 1; 8]),
            Some(F::ORDER - 1)
        );
        for position in 0..8 {
            for rejected in [F::ORDER, u64::MAX] {
                let mut words = [0u64; 8];
                words[position] = rejected;
                assert_eq!(accepted_pow_response::<F>(words), None);
            }
        }
    }
}
