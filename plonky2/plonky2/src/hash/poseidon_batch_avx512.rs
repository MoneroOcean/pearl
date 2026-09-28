//! Eight independent, protocol-identical Goldilocks Poseidon permutations.
//!
//! Instead of sparse partial-round matrices with arbitrary field coefficients,
//! use the original small-coefficient MDS in every round. Forward substitution
//! moves the linear coordinates' constants into one scalar per partial round
//! and a final residual vector. This strategy is also described in Plonky3's
//! `poseidon1/src/internal.rs::textbook_partial_permute_state`; the parameters
//! here are exclusively Pearl/Plonky2's, NOT Plonky3's different Poseidon hash.
//!
//! The MDS uses the same 3 x 4 FFT factorization as our scalar implementation,
//! but splits each lane into 32-bit limbs and delays reduction to the output.

use core::arch::x86_64::*;
use core::ops::{Add, Sub};

use plonky2_field::goldilocks_field::GoldilocksField as F;
use plonky2_field::ops::Square;
use plonky2_field::packable::Packable;
#[cfg(test)]
use plonky2_field::packed::PackedField;
use plonky2_field::types::Field;

use crate::hash::poseidon::{
    Poseidon, ALL_ROUND_CONSTANTS, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS, N_ROUNDS, SPONGE_WIDTH,
};

type P = <F as Packable>::Packing;
const MODULUS: u128 = 0xffff_ffff_0000_0001;
const EPSILON: i64 = 0xffff_ffff;

// The integer factorization below is specific to these exact coefficients.
const _: () = {
    let expected_circ = [17, 15, 41, 16, 2, 28, 13, 13, 39, 18, 34, 20];
    let mut i = 0;
    while i < SPONGE_WIDTH {
        assert!(F::MDS_MATRIX_CIRC[i] == expected_circ[i]);
        assert!(F::MDS_MATRIX_DIAG[i] == if i == 0 { 8 } else { 0 });
        i += 1;
    }
};

/// Maintain `original_state = transformed_state + offset` between rounds.
/// Only coordinate zero is nonlinear, so its offset is consumed before x^7;
/// all remaining offsets are carried through the linear MDS to the next round.
const fn partial_constants() -> ([u64; N_PARTIAL_ROUNDS], [u64; SPONGE_WIDTH]) {
    let mut scalar = [0; N_PARTIAL_ROUNDS];
    let mut offset = [0u64; SPONGE_WIDTH];
    let mut round = 0;
    while round < N_PARTIAL_ROUNDS {
        let mut remainder = [0u64; SPONGE_WIDTH];
        let mut i = 0;
        while i < SPONGE_WIDTH {
            let c = ALL_ROUND_CONSTANTS[(round + HALF_N_FULL_ROUNDS) * SPONGE_WIDTH + i];
            let sum = ((offset[i] as u128 + c as u128) % MODULUS) as u64;
            if i == 0 {
                scalar[round] = sum;
            } else {
                remainder[i] = sum;
            }
            i += 1;
        }
        let mut row = 0;
        while row < SPONGE_WIDTH {
            let mut sum = remainder[row] as u128 * F::MDS_MATRIX_DIAG[row] as u128;
            let mut col = 0;
            while col < SPONGE_WIDTH {
                sum +=
                    remainder[(row + col) % SPONGE_WIDTH] as u128 * F::MDS_MATRIX_CIRC[col] as u128;
                col += 1;
            }
            offset[row] = (sum % MODULUS) as u64;
            row += 1;
        }
        round += 1;
    }
    (scalar, offset)
}

const PARTIAL_CONSTANTS: ([u64; N_PARTIAL_ROUNDS], [u64; SPONGE_WIDTH]) = partial_constants();

#[derive(Copy, Clone)]
#[repr(transparent)]
struct Word(__m512i);

impl Add for Word {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // SAFETY: this entire module is compiled only for the native AVX-512 packing.
        Self(unsafe { _mm512_add_epi64(self.0, rhs.0) })
    }
}

impl Sub for Word {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_sub_epi64(self.0, rhs.0) })
    }
}

impl Word {
    #[inline(always)]
    fn shl<const N: u32>(self) -> Self {
        Self(unsafe { _mm512_slli_epi64::<N>(self.0) })
    }
}

#[inline(always)]
fn fft4(x: [Word; 4]) -> (Word, (Word, Word), Word) {
    let a = x[0] + x[2];
    let b = x[1] + x[3];
    (a + b, (x[0] - x[2], x[3] - x[1]), a - b)
}

#[inline(always)]
fn ifft4(y: (Word, (Word, Word), Word)) -> [Word; 4] {
    let a = y.0 + y.2;
    let b = y.0 - y.2;
    [a + y.1 .0, b - y.1 .1, a - y.1 .0, b + y.1 .1]
}

#[inline(always)]
fn mds_limb(s: [Word; SPONGE_WIDTH]) -> [Word; SPONGE_WIDTH] {
    let (a0, (b0r, b0i), c0) = fft4([s[0], s[3], s[6], s[9]]);
    let (a1, (b1r, b1i), c1) = fft4([s[1], s[4], s[7], s[10]]);
    let (a2, (b2r, b2i), c2) = fft4([s[2], s[5], s[8], s[11]]);
    let sum = a0 + a1 + a2;
    let a0_out = (sum + a2).shl::<4>();
    let a1_out = (sum + a0).shl::<4>();
    let a2_out = (sum + a1).shl::<4>();

    let b0r_out = b0r.shl::<1>() + b0i + b1r + b1i.shl::<4>() + b2r - b2i.shl::<2>();
    let b0i_out = b0i.shl::<1>() - b0r - b1r.shl::<4>() + b1i + b2r.shl::<2>() + b2i;
    let b1r_out = b1r.shl::<1>() + b1i - b0r.shl::<2>() - b0i + b2r + b2i.shl::<4>();
    let b1i_out = b0r - b0i.shl::<2>() - b1r + b1i.shl::<1>() - b2r.shl::<4>() + b2i;
    let b2r_out = b0r.shl::<4>() - b0i - b1r.shl::<2>() - b1i + b2r.shl::<1>() + b2i;
    let b2i_out = b0r + b0i.shl::<4>() + b1r - b1i.shl::<2>() - b2r + b2i.shl::<1>();

    let zero = Word(unsafe { _mm512_setzero_si512() });
    let c0_out = c2.shl::<3>() - c0 - c1.shl::<1>();
    let c1_out = zero - c0.shl::<3>() - c1 - c2.shl::<1>();
    let c2_out = c0.shl::<1>() - c1.shl::<3>() - c2;

    let [r0, r3, r6, r9] = ifft4((a0_out, (b0r_out, b0i_out), c0_out));
    let [r1, r4, r7, r10] = ifft4((a1_out, (b1r_out, b1i_out), c1_out));
    let [r2, r5, r8, r11] = ifft4((a2_out, (b2r_out, b2i_out), c2_out));
    [
        r0 + s[0].shl::<3>(),
        r1,
        r2,
        r3,
        r4,
        r5,
        r6,
        r7,
        r8,
        r9,
        r10,
        r11,
    ]
}

#[inline(always)]
unsafe fn reduce_limbs(low: Word, high: Word) -> __m512i {
    // Both limbs may grow through three linear layers, each with row sum at
    // most 264. Thus each is at most 264^3*(2^32-1) < 2^57. The high word of
    // low + 2^32*high remains below 2^25, so one bounded Goldilocks fold fits.
    let mask = _mm512_set1_epi64(EPSILON);
    let lo = _mm512_add_epi64(low.0, _mm512_slli_epi64::<32>(high.0));
    let carry = _mm512_cmplt_epu64_mask(lo, low.0);
    let hi = _mm512_add_epi64(
        _mm512_srli_epi64::<32>(high.0),
        _mm512_maskz_mov_epi64(carry, _mm512_set1_epi64(1)),
    );
    let correction = _mm512_mul_epu32(hi, mask);
    let sum = _mm512_add_epi64(lo, correction);
    let overflow = _mm512_cmplt_epu64_mask(sum, lo);
    _mm512_mask_add_epi64(sum, overflow, sum, mask)
}

#[inline(always)]
fn mds(state: [P; SPONGE_WIDTH]) -> [P; SPONGE_WIDTH] {
    // SAFETY: P is the repr(transparent) eight-Goldilocks-lane packing selected
    // by this module's cfg. Every u64 is a valid (possibly noncanonical) field
    // representation. Transmute by value avoids any reference/alignment casts.
    unsafe {
        let raw: [__m512i; SPONGE_WIDTH] = core::mem::transmute(state);
        let mask = _mm512_set1_epi64(EPSILON);
        let low = mds_limb(raw.map(|x| Word(_mm512_and_si512(x, mask))));
        let high = mds_limb(raw.map(|x| Word(_mm512_srli_epi64::<32>(x))));
        let result: [__m512i; SPONGE_WIDTH] =
            core::array::from_fn(|i| reduce_limbs(low[i], high[i]));
        core::mem::transmute(result)
    }
}

#[inline(always)]
fn sbox(x: P) -> P {
    let x2 = x.square();
    let x3 = x2 * x;
    x3 * x2.square()
}

#[inline]
fn partial_rounds(mut state: [P; SPONGE_WIDTH]) -> [P; SPONGE_WIDTH] {
    // Only coordinate zero is nonlinear. Carry the other eleven coordinates
    // as unreduced integer limbs through three linear rounds. This preserves
    // the same field values while avoiding eleven reductions per inner round.
    for constants in PARTIAL_CONSTANTS.0.chunks(3) {
        unsafe {
            let raw: [__m512i; SPONGE_WIDTH] = core::mem::transmute(state);
            let mask = _mm512_set1_epi64(EPSILON);
            let mut low = raw.map(|x| Word(_mm512_and_si512(x, mask)));
            let mut high = raw.map(|x| Word(_mm512_srli_epi64::<32>(x)));
            for &constant in constants {
                let coordinate: P = core::mem::transmute(reduce_limbs(low[0], high[0]));
                let nonlinear: __m512i = core::mem::transmute(
                    sbox(coordinate + F::from_canonical_u64(constant)));
                low[0] = Word(_mm512_and_si512(nonlinear, mask));
                high[0] = Word(_mm512_srli_epi64::<32>(nonlinear));
                low = mds_limb(low);
                high = mds_limb(high);
            }
            let result: [__m512i; SPONGE_WIDTH] =
                core::array::from_fn(|i| reduce_limbs(low[i], high[i]));
            state = core::mem::transmute(result);
        }
    }
    for (value, constant) in state.iter_mut().zip(PARTIAL_CONSTANTS.1) {
        *value += F::from_canonical_u64(constant);
    }
    state
}

#[cfg(test)]
#[inline]
fn permute_sequential(mut state: [P; SPONGE_WIDTH]) -> [P; SPONGE_WIDTH] {
    for round in 0..HALF_N_FULL_ROUNDS {
        for i in 0..SPONGE_WIDTH {
            state[i] = sbox(
                state[i] + F::from_canonical_u64(ALL_ROUND_CONSTANTS[round * SPONGE_WIDTH + i]),
            );
        }
        state = mds(state);
    }
    state = partial_rounds(state);
    for round in HALF_N_FULL_ROUNDS + N_PARTIAL_ROUNDS..N_ROUNDS {
        for i in 0..SPONGE_WIDTH {
            state[i] = sbox(
                state[i] + F::from_canonical_u64(ALL_ROUND_CONSTANTS[round * SPONGE_WIDTH + i]),
            );
        }
        state = mds(state);
    }
    state
}

#[inline(always)]
fn full_sboxes_interleaved<const N: usize>(state: &mut [P; SPONGE_WIDTH], round: usize) {
    for start in (0..SPONGE_WIDTH).step_by(N) {
        let x: [P; N] = core::array::from_fn(|i| {
            state[start + i]
                + F::from_canonical_u64(ALL_ROUND_CONSTANTS[round * SPONGE_WIDTH + start + i])
        });
        let x2 = x.map(|v| v.square());
        let x4 = x2.map(|v| v.square());
        for i in 0..N {
            state[start + i] = (x2[i] * x[i]) * x4[i];
        }
    }
}

#[inline]
fn permute_interleaved<const N: usize>(mut state: [P; SPONGE_WIDTH]) -> [P; SPONGE_WIDTH] {
    for round in 0..HALF_N_FULL_ROUNDS {
        full_sboxes_interleaved::<N>(&mut state, round);
        state = mds(state);
    }
    state = partial_rounds(state);
    for round in HALF_N_FULL_ROUNDS + N_PARTIAL_ROUNDS..N_ROUNDS {
        full_sboxes_interleaved::<N>(&mut state, round);
        state = mds(state);
    }
    state
}

#[inline]
pub(crate) fn permute(state: [P; SPONGE_WIDTH]) -> [P; SPONGE_WIDTH] {
    // Interleave independent S-box dependency chains without keeping every
    // intermediate for all twelve coordinates live simultaneously.
    permute_interleaved::<3>(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use plonky2_field::types::PrimeField64;

    #[test]
    fn interleaved_sboxes_match_existing_permutation() {
        let mut seed = 0xa409_3822_299f_31d0;
        for case in 0..128 {
            let input = state(&mut seed, case);
            let expected = permute_sequential(input);
            for actual in [
                permute_interleaved::<2>(input),
                permute_interleaved::<3>(input),
                permute_interleaved::<4>(input),
                permute_interleaved::<6>(input),
                permute_interleaved::<12>(input),
            ] {
                assert_eq!(
                    actual.map(|p| p.as_slice().to_vec()),
                    expected.map(|p| p.as_slice().to_vec())
                );
            }
        }
    }

    #[test]
    #[ignore = "manual permutation scheduling benchmark"]
    fn benchmark_permutation_schedules() {
        use std::hint::black_box;
        let variants: [(&str, fn([P; SPONGE_WIDTH]) -> [P; SPONGE_WIDTH]); 7] = [
            ("sequential", permute_sequential),
            ("groups2", permute_interleaved::<2>),
            ("groups3", permute_interleaved::<3>),
            ("groups4", permute_interleaved::<4>),
            ("groups6", permute_interleaved::<6>),
            ("groups12", permute_interleaved::<12>),
            (
                "sparse",
                crate::hash::poseidon_batch::packed_poseidon_permutation::<F, P>,
            ),
        ];
        for _ in 0..3 {
            for (name, kernel) in variants {
                let mut seed = 0x517c_c1b7_2722_0a95;
                let mut input = state(&mut seed, 8);
                let start = std::time::Instant::now();
                for _ in 0..100000 {
                    input = kernel(black_box(input));
                }
                eprintln!(
                    "{name}: {:?}, checksum {:?}",
                    start.elapsed(),
                    black_box(input[0])
                );
            }
        }
    }

    fn state(seed: &mut u64, case: usize) -> [P; SPONGE_WIDTH] {
        let mut result = [P::ZEROS; SPONGE_WIDTH];
        for value in &mut result {
            for lane in value.as_slice_mut() {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                *lane = match case {
                    0 => F::ZERO,
                    1 => F::ONE,
                    2 => F::NEG_ONE,
                    3 => F::from_noncanonical_u64(u64::MAX),
                    4 => F::from_noncanonical_u64(MODULUS as u64),
                    _ => F::from_noncanonical_u64(*seed),
                };
            }
        }
        result
    }

    #[test]
    fn delayed_reduction_mds_matches_scalar() {
        let mut seed = 0x3c6e_f372_fe94_f82b;
        for case in 0..512 {
            let input = state(&mut seed, case);
            let actual = mds(input);
            for lane in 0..P::WIDTH {
                let scalar = core::array::from_fn(|i| input[i].as_slice()[lane]);
                let expected = F::mds_layer(&scalar);
                for i in 0..SPONGE_WIDTH {
                    assert_eq!(
                        actual[i].as_slice()[lane],
                        expected[i],
                        "case={case},lane={lane},i={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn reduce_limbs_matches_u128_modulo_at_max_growth() {
        let max_limb = 264u128.pow(3) * ((1u128 << 32) - 1);
        assert!(max_limb < (1u128 << 57));
        let max_limb = max_limb as u64;
        let epsilon = EPSILON as u64;
        let cases = [
            (
                [
                    0,
                    1,
                    epsilon,
                    max_limb,
                    max_limb - 1,
                    1u64 << 56,
                    (1u64 << 25) - 1,
                    (1u64 << 32) - 1,
                ],
                [
                    0,
                    epsilon,
                    epsilon,
                    max_limb,
                    max_limb - 1,
                    1u64 << 56,
                    (1u64 << 25) - 1,
                    1u64 << 32,
                ],
            ),
            (
                [
                    epsilon,
                    max_limb,
                    max_limb - 1,
                    1u64 << 32,
                    0,
                    1,
                    1u64 << 56,
                    max_limb,
                ],
                [
                    1u64 << 32,
                    (1u64 << 33) - 1,
                    (1u64 << 56) + epsilon,
                    max_limb,
                    max_limb - 1,
                    (1u64 << 25) - 1,
                    1,
                    0,
                ],
            ),
        ];

        for (case, (low_lanes, high_lanes)) in cases.into_iter().enumerate() {
            let low = Word(unsafe { core::mem::transmute(low_lanes) });
            let high = Word(unsafe { core::mem::transmute(high_lanes) });
            let actual: [u64; 8] = unsafe { core::mem::transmute(reduce_limbs(low, high)) };
            for lane in 0..8 {
                let expected =
                    (low_lanes[lane] as u128 + ((high_lanes[lane] as u128) << 32)) % MODULUS;
                assert_eq!(
                    (actual[lane] as u128) % MODULUS,
                    expected,
                    "case={case},lane={lane}"
                );
            }
        }
    }

    #[test]
    fn three_raw_mds_limb_rounds_match_scalar_mds() {
        let mut seed = 0x6a09_e667_f3bc_c909;
        let mask = EPSILON as u64;
        for case in 0..16 {
            let input = state(&mut seed, case);
            let mut low: [Word; SPONGE_WIDTH] = core::array::from_fn(|i| {
                let lanes: [u64; 8] = core::array::from_fn(|lane| {
                    input[i].as_slice()[lane].to_noncanonical_u64() & mask
                });
                Word(unsafe { core::mem::transmute(lanes) })
            });
            let mut high: [Word; SPONGE_WIDTH] = core::array::from_fn(|i| {
                let lanes: [u64; 8] = core::array::from_fn(|lane| {
                    input[i].as_slice()[lane].to_noncanonical_u64() >> 32
                });
                Word(unsafe { core::mem::transmute(lanes) })
            });

            for _ in 0..3 {
                low = mds_limb(low);
                high = mds_limb(high);
            }
            let reduced: [__m512i; SPONGE_WIDTH] =
                core::array::from_fn(|i| unsafe { reduce_limbs(low[i], high[i]) });
            let actual: [P; SPONGE_WIDTH] = unsafe { core::mem::transmute(reduced) };

            for lane in 0..P::WIDTH {
                let mut expected = core::array::from_fn(|i| input[i].as_slice()[lane]);
                for _ in 0..3 {
                    expected = F::mds_layer(&expected);
                }
                for i in 0..SPONGE_WIDTH {
                    assert_eq!(
                        actual[i].as_slice()[lane],
                        expected[i],
                        "case={case},lane={lane},i={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn substituted_partial_rounds_match_naive() {
        let mut seed = 0xbb67_ae85_84ca_a73b;
        for case in 0..256 {
            let input = state(&mut seed, case);
            let actual = partial_rounds(input);
            for lane in 0..P::WIDTH {
                let mut expected = core::array::from_fn(|i| input[i].as_slice()[lane]);
                let mut round = HALF_N_FULL_ROUNDS;
                F::partial_rounds_naive(&mut expected, &mut round);
                for i in 0..SPONGE_WIDTH {
                    assert_eq!(
                        actual[i].as_slice()[lane],
                        expected[i],
                        "case={case},lane={lane},i={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn full_permutation_matches_scalar_and_naive() {
        let mut seed = 0xa54f_f53a_5f1d_36f1;
        for case in 0..256 {
            let input = state(&mut seed, case);
            let actual = permute(input);
            for lane in 0..P::WIDTH {
                let scalar = core::array::from_fn(|i| input[i].as_slice()[lane]);
                let expected = F::poseidon(scalar);
                assert_eq!(expected, F::poseidon_naive(scalar));
                for i in 0..SPONGE_WIDTH {
                    assert_eq!(
                        actual[i].as_slice()[lane],
                        expected[i],
                        "case={case},lane={lane},i={i}"
                    );
                }
            }
        }
    }
}
