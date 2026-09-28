//! SIMD batching for independent Poseidon `hash_or_noop` calls.
//!
//! Eight-lane packing is profitable on the measured AVX-512 CPUs. Four-lane
//! AVX2 packing was slower than the optimized scalar permutation, so that
//! target deliberately keeps the scalar path.

use plonky2_field::packable::Packable;
use plonky2_field::packed::PackedField;

use crate::hash::hash_types::{HashOut, RichField, NUM_HASH_OUT_ELTS};
use crate::hash::poseidon::{
    Poseidon, PoseidonHash, ALL_ROUND_CONSTANTS, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS, SPONGE_RATE,
    SPONGE_WIDTH,
};
use crate::plonk::config::Hasher;

/// Hashes independent inputs with `PoseidonHash::hash_or_noop` semantics.
///
/// Equal-length inputs are processed in packed groups of the field's native
/// packing width. Inputs of length at most four, unequal-length batches, and
/// any trailing inputs that do not fill a packed group use the scalar hasher.
/// `outputs` must have the same length as `inputs`.
pub(crate) fn hash_or_noop_batch<F: RichField>(inputs: &[&[F]], outputs: &mut [HashOut<F>]) {
    assert_eq!(inputs.len(), outputs.len());

    let Some(first_input) = inputs.first() else {
        return;
    };
    if first_input.len() <= NUM_HASH_OUT_ELTS
        || inputs.iter().any(|input| input.len() != first_input.len())
    {
        hash_scalar_batch(inputs, outputs);
        return;
    }

    let packed_width = <<F as Packable>::Packing as PackedField>::WIDTH;
    if packed_width < 8 {
        hash_scalar_batch(inputs, outputs);
        return;
    }

    let packed_len = inputs.len() / packed_width * packed_width;
    for group_start in (0..packed_len).step_by(packed_width) {
        let group_end = group_start + packed_width;
        hash_equal_len_packed::<F, <F as Packable>::Packing>(
            &inputs[group_start..group_end],
            &mut outputs[group_start..group_end],
        );
    }
    hash_scalar_batch(&inputs[packed_len..], &mut outputs[packed_len..]);
}

#[inline]
fn hash_scalar_batch<F: RichField>(inputs: &[&[F]], outputs: &mut [HashOut<F>]) {
    for (input, output) in inputs.iter().zip(outputs) {
        *output = <PoseidonHash as Hasher<F>>::hash_or_noop(input);
    }
}

/// Compress independent Merkle-node pairs with one permutation per pair.
pub(crate) fn two_to_one_batch<F: RichField>(
    left: &[HashOut<F>],
    right: &[HashOut<F>],
    outputs: &mut [HashOut<F>],
) {
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), outputs.len());
    let packed_width = <<F as Packable>::Packing as PackedField>::WIDTH;
    let packed_len = if packed_width >= 8 {
        left.len() / packed_width * packed_width
    } else {
        0
    };
    for group_start in (0..packed_len).step_by(packed_width) {
        let group_end = group_start + packed_width;
        compress_packed::<F, <F as Packable>::Packing>(
            &left[group_start..group_end],
            &right[group_start..group_end],
            &mut outputs[group_start..group_end],
        );
    }
    for idx in packed_len..left.len() {
        outputs[idx] = <PoseidonHash as Hasher<F>>::two_to_one(left[idx], right[idx]);
    }
}

fn compress_packed<F, P>(left: &[HashOut<F>], right: &[HashOut<F>], outputs: &mut [HashOut<F>])
where
    F: RichField + Packable<Packing = P>,
    P: PackedField<Scalar = F>,
{
    debug_assert_eq!(left.len(), P::WIDTH);
    debug_assert_eq!(right.len(), P::WIDTH);
    debug_assert_eq!(outputs.len(), P::WIDTH);
    let mut state = [P::ZEROS; SPONGE_WIDTH];
    for idx in 0..NUM_HASH_OUT_ELTS {
        for lane in 0..P::WIDTH {
            state[idx].as_slice_mut()[lane] = left[lane].elements[idx];
            state[NUM_HASH_OUT_ELTS + idx].as_slice_mut()[lane] = right[lane].elements[idx];
        }
    }
    state = F::poseidon_native_packed(state);
    for idx in 0..NUM_HASH_OUT_ELTS {
        for lane in 0..P::WIDTH {
            outputs[lane].elements[idx] = state[idx].as_slice()[lane];
        }
    }
}

fn hash_equal_len_packed<F, P>(inputs: &[&[F]], outputs: &mut [HashOut<F>])
where
    F: RichField + Packable<Packing = P>,
    P: PackedField<Scalar = F>,
{
    debug_assert_eq!(inputs.len(), P::WIDTH);
    debug_assert_eq!(outputs.len(), P::WIDTH);
    debug_assert!(inputs.iter().all(|input| input.len() > NUM_HASH_OUT_ELTS));
    debug_assert!(inputs.iter().all(|input| input.len() == inputs[0].len()));

    let mut state = [P::ZEROS; SPONGE_WIDTH];
    for input_start in (0..inputs[0].len()).step_by(SPONGE_RATE) {
        let chunk_len = (inputs[0].len() - input_start).min(SPONGE_RATE);
        for state_idx in 0..chunk_len {
            let packed_state = &mut state[state_idx];
            let lanes = packed_state.as_slice_mut();
            for lane in 0..P::WIDTH {
                lanes[lane] = inputs[lane][input_start + state_idx];
            }
        }
        state = F::poseidon_native_packed(state);
    }

    for output_idx in 0..NUM_HASH_OUT_ELTS {
        let lanes = state[output_idx].as_slice();
        for lane in 0..P::WIDTH {
            outputs[lane].elements[output_idx] = lanes[lane];
        }
    }
}

#[inline]
fn packed_sbox<P: PackedField>(x: P) -> P {
    let x2 = x.square();
    let x3 = x2 * x;
    let x4 = x2.square();
    x3 * x4
}

#[inline(always)]
pub(crate) fn packed_mds_layer<F, P>(state: &[P; SPONGE_WIDTH]) -> [P; SPONGE_WIDTH]
where
    F: Poseidon,
    P: PackedField<Scalar = F>,
{
    // The frequency-domain factorization used by the scalar Goldilocks MDS
    // also works over packed field elements. Keep the generic fallback so a
    // different Poseidon parameter set cannot silently use these constants.
    if F::MDS_MATRIX_CIRC == [17, 15, 41, 16, 2, 28, 13, 13, 39, 18, 34, 20]
        && F::MDS_MATRIX_DIAG == [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    {
        return packed_goldilocks_mds(state);
    }
    let mut result = [P::ZEROS; SPONGE_WIDTH];
    for row in 0..SPONGE_WIDTH {
        let mut sum = P::ZEROS;
        for col in 0..SPONGE_WIDTH {
            let coeff = F::from_canonical_u64(F::MDS_MATRIX_CIRC[col]);
            sum += state[(col + row) % SPONGE_WIDTH] * coeff;
        }
        let diagonal = F::from_canonical_u64(F::MDS_MATRIX_DIAG[row]);
        sum += state[row] * diagonal;
        result[row] = sum;
    }
    result
}

#[inline(always)]
fn times_two<P: PackedField>(value: P) -> P {
    value + value
}

#[inline(always)]
fn times_four<P: PackedField>(value: P) -> P {
    times_two(times_two(value))
}

#[inline(always)]
fn times_eight<P: PackedField>(value: P) -> P {
    times_two(times_four(value))
}

#[inline(always)]
fn times_sixteen<P: PackedField>(value: P) -> P {
    times_four(times_four(value))
}

#[inline(always)]
fn fft4<P: PackedField>(x: [P; 4]) -> (P, (P, P), P) {
    let z0 = x[0] + x[2];
    let z1 = x[1] + x[3];
    (z0 + z1, (x[0] - x[2], x[3] - x[1]), z0 - z1)
}

#[inline(always)]
fn ifft4<P: PackedField>(y: (P, (P, P), P)) -> [P; 4] {
    let z0 = y.0 + y.2;
    let z1 = y.0 - y.2;
    [z0 + y.1 .0, z1 - y.1 .1, z0 - y.1 .0, z1 + y.1 .1]
}

/// The same 3 x 4 factorization and scaling as `poseidon12_mds`.
/// Constants are small powers of two, implemented without full field multiplies.
#[inline(always)]
fn packed_goldilocks_mds<P: PackedField>(s: &[P; SPONGE_WIDTH]) -> [P; SPONGE_WIDTH] {
    let (a0, (b0r, b0i), c0) = fft4([s[0], s[3], s[6], s[9]]);
    let (a1, (b1r, b1i), c1) = fft4([s[1], s[4], s[7], s[10]]);
    let (a2, (b2r, b2i), c2) = fft4([s[2], s[5], s[8], s[11]]);

    // Frequency block one: [16, 32, 16].
    let sum = a0 + a1 + a2;
    let a0_out = times_sixteen(sum + a2);
    let a1_out = times_sixteen(sum + a0);
    let a2_out = times_sixteen(sum + a1);

    // Frequency block two: [(2, -1), (-4, 1), (16, 1)].
    let b0r_out = times_two(b0r) + b0i + b1r + times_sixteen(b1i) + b2r - times_four(b2i);
    let b0i_out = times_two(b0i) - b0r - times_sixteen(b1r) + b1i + times_four(b2r) + b2i;
    let b1r_out = times_two(b1r) + b1i - times_four(b0r) - b0i + b2r + times_sixteen(b2i);
    let b1i_out = b0r - times_four(b0i) - b1r + times_two(b1i) - times_sixteen(b2r) + b2i;
    let b2r_out = times_sixteen(b0r) - b0i - times_four(b1r) - b1i + times_two(b2r) + b2i;
    let b2i_out = b0r + times_sixteen(b0i) + b1r - times_four(b1i) - b2r + times_two(b2i);

    // Frequency block three: [-1, -8, 2].
    let c0_out = times_eight(c2) - c0 - times_two(c1);
    let c1_out = P::ZEROS - times_eight(c0) - c1 - times_two(c2);
    let c2_out = times_two(c0) - times_eight(c1) - c2;

    let [r0, r3, r6, r9] = ifft4((a0_out, (b0r_out, b0i_out), c0_out));
    let [r1, r4, r7, r10] = ifft4((a1_out, (b1r_out, b1i_out), c1_out));
    let [r2, r5, r8, r11] = ifft4((a2_out, (b2r_out, b2i_out), c2_out));
    [
        r0 + times_eight(s[0]),
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
#[unroll::unroll_for_loops]
fn packed_partial_mds_init<F, P>(state: &[P; SPONGE_WIDTH]) -> [P; SPONGE_WIDTH]
where
    F: Poseidon,
    P: PackedField<Scalar = F>,
{
    let mut result = [P::ZEROS; SPONGE_WIDTH];
    result[0] = state[0];
    for row in 1..12 {
        for col in 1..12 {
            let coeff =
                F::from_canonical_u64(F::FAST_PARTIAL_ROUND_INITIAL_MATRIX[row - 1][col - 1]);
            result[col] += state[row] * coeff;
        }
    }
    result
}

#[inline(always)]
#[unroll::unroll_for_loops]
fn packed_partial_mds_fast<F, P>(state: &[P; SPONGE_WIDTH], round: usize) -> [P; SPONGE_WIDTH]
where
    F: Poseidon,
    P: PackedField<Scalar = F>,
{
    let mut first = state[0] * F::from_canonical_u64(F::MDS_MATRIX_CIRC[0] + F::MDS_MATRIX_DIAG[0]);
    for idx in 1..12 {
        let coeff = F::from_canonical_u64(F::FAST_PARTIAL_ROUND_W_HATS[round][idx - 1]);
        first += state[idx] * coeff;
    }

    let mut result = [P::ZEROS; SPONGE_WIDTH];
    result[0] = first;
    for idx in 1..12 {
        let coeff = F::from_canonical_u64(F::FAST_PARTIAL_ROUND_VS[round][idx - 1]);
        result[idx] = state[0] * coeff + state[idx];
    }
    result
}

pub(crate) fn packed_poseidon_permutation<F, P>(mut state: [P; SPONGE_WIDTH]) -> [P; SPONGE_WIDTH]
where
    F: Poseidon,
    P: PackedField<Scalar = F>,
{
    let mut round = 0;
    for _ in 0..HALF_N_FULL_ROUNDS {
        for idx in 0..SPONGE_WIDTH {
            state[idx] += F::from_canonical_u64(ALL_ROUND_CONSTANTS[idx + SPONGE_WIDTH * round]);
            state[idx] = packed_sbox(state[idx]);
        }
        state = packed_mds_layer::<F, P>(&state);
        round += 1;
    }

    for idx in 0..SPONGE_WIDTH {
        state[idx] += F::from_canonical_u64(F::FAST_PARTIAL_FIRST_ROUND_CONSTANT[idx]);
    }
    state = packed_partial_mds_init::<F, P>(&state);
    for partial_round in 0..N_PARTIAL_ROUNDS {
        state[0] = packed_sbox(state[0]);
        state[0] += F::from_canonical_u64(F::FAST_PARTIAL_ROUND_CONSTANTS[partial_round]);
        state = packed_partial_mds_fast::<F, P>(&state, partial_round);
    }
    round += N_PARTIAL_ROUNDS;

    for _ in 0..HALF_N_FULL_ROUNDS {
        for idx in 0..SPONGE_WIDTH {
            state[idx] += F::from_canonical_u64(ALL_ROUND_CONSTANTS[idx + SPONGE_WIDTH * round]);
            state[idx] = packed_sbox(state[idx]);
        }
        state = packed_mds_layer::<F, P>(&state);
        round += 1;
    }
    debug_assert_eq!(round, crate::hash::poseidon::N_ROUNDS);
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::{Field, PrimeField64};
    use crate::hash::poseidon::Poseidon;

    type F = GoldilocksField;
    type P = <F as Packable>::Packing;

    fn next_u64(state: &mut u64) -> u64 {
        // Deterministic xorshift generator so the test needs no RNG dependency.
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn next_field(state: &mut u64, index: usize) -> F {
        // Include values above the Goldilocks modulus as well as pseudorandom values.
        let value = if index % 7 == 0 {
            u64::MAX.wrapping_sub(index as u64)
        } else {
            next_u64(state)
        };
        F::from_noncanonical_u64(value)
    }

    fn assert_batch_matches_scalar(input_count: usize, input_len: usize) {
        let mut seed = 0x8f3d_9a72_64c1_05e7 ^ input_count as u64 ^ ((input_len as u64) << 32);
        let input_storage: Vec<Vec<F>> = (0..input_count)
            .map(|lane| {
                (0..input_len)
                    .map(|idx| next_field(&mut seed, lane * input_len + idx))
                    .collect()
            })
            .collect();
        let inputs: Vec<&[F]> = input_storage.iter().map(Vec::as_slice).collect();
        let mut actual = vec![HashOut::ZERO; input_count];
        hash_or_noop_batch(&inputs, &mut actual);

        for (input, actual_hash) in inputs.iter().zip(actual) {
            let expected = <PoseidonHash as Hasher<F>>::hash_or_noop(input);
            assert_eq!(
                actual_hash, expected,
                "count={input_count}, len={input_len}"
            );
        }
    }

    #[test]
    fn batched_compression_matches_scalar_with_noncanonical_inputs() {
        let mut seed = 0xd131_0ba6_98df_b5ac;
        for count in [0, 1, 2, 7, 8, 9, 15, 16, 17, 128] {
            for case in 0..8 {
                let mut next_hash = || HashOut {
                    elements: core::array::from_fn(|idx| match case {
                        0 => F::ZERO,
                        1 => F::ONE,
                        2 => F::NEG_ONE,
                        3 => F::from_noncanonical_u64(u64::MAX),
                        _ => next_field(&mut seed, idx + case * 7),
                    }),
                };
                let left: Vec<_> = (0..count).map(|_| next_hash()).collect();
                let right: Vec<_> = (0..count).map(|_| next_hash()).collect();
                let mut actual = vec![HashOut::ZERO; count];
                two_to_one_batch(&left, &right, &mut actual);
                for idx in 0..count {
                    assert_eq!(
                        actual[idx],
                        <PoseidonHash as Hasher<F>>::two_to_one(left[idx], right[idx]),
                        "count={count}, case={case}, idx={idx}",
                    );
                }
            }
        }
    }

    #[test]
    fn batched_hash_matches_scalar_for_small_and_long_inputs() {
        let packed_width = P::WIDTH;
        let mut counts = vec![
            0,
            1,
            packed_width.saturating_sub(1),
            packed_width,
            packed_width + 1,
            2 * packed_width + 3,
        ];
        counts.sort_unstable();
        counts.dedup();

        for input_len in 0..=20 {
            for &input_count in &counts {
                assert_batch_matches_scalar(input_count, input_len);
            }
        }
        for input_len in [64, 1024] {
            for &input_count in &[packed_width, packed_width + 1, 2 * packed_width + 3] {
                assert_batch_matches_scalar(input_count, input_len);
            }
        }
    }

    #[test]
    fn unequal_lengths_use_scalar_hashing() {
        let input_count = 2 * P::WIDTH + 3;
        let mut seed = 0x3141_5926_5358_9793;
        let input_storage: Vec<Vec<F>> = (0..input_count)
            .map(|lane| {
                (0..5 + lane % (P::WIDTH + 1))
                    .map(|idx| next_field(&mut seed, lane * 31 + idx))
                    .collect()
            })
            .collect();
        let inputs: Vec<&[F]> = input_storage.iter().map(Vec::as_slice).collect();
        let mut actual = vec![HashOut::ZERO; input_count];
        hash_or_noop_batch(&inputs, &mut actual);

        for (input, actual_hash) in inputs.iter().zip(actual) {
            assert_eq!(
                actual_hash,
                <PoseidonHash as Hasher<F>>::hash_or_noop(input)
            );
        }
    }

    #[test]
    fn packed_mds_matches_scalar_on_edge_cases_and_random_states() {
        let mut seed = 0x6a09_e667_f3bc_c909;
        for case in 0..512 {
            let mut state = [P::ZEROS; SPONGE_WIDTH];
            for idx in 0..SPONGE_WIDTH {
                for lane in 0..P::WIDTH {
                    state[idx].as_slice_mut()[lane] = match case {
                        0 => F::ZERO,
                        1 => F::ONE,
                        2 => F::NEG_ONE,
                        3 => F::from_noncanonical_u64(u64::MAX),
                        _ => next_field(&mut seed, case + lane * SPONGE_WIDTH + idx),
                    };
                }
            }
            let actual = packed_mds_layer::<F, P>(&state);
            for lane in 0..P::WIDTH {
                let scalar = core::array::from_fn(|idx| state[idx].as_slice()[lane]);
                let expected = F::mds_layer(&scalar);
                for idx in 0..SPONGE_WIDTH {
                    assert_eq!(
                        actual[idx].as_slice()[lane],
                        expected[idx],
                        "case={case}, lane={lane}, idx={idx}"
                    );
                }
            }
        }
    }

    #[test]
    fn packed_permutation_matches_scalar_per_lane() {
        let mut seed = 0x2718_2818_2845_9045;
        let mut scalar_states = vec![[F::ZERO; SPONGE_WIDTH]; P::WIDTH];
        let mut packed_state = [P::ZEROS; SPONGE_WIDTH];

        for (lane, scalar_state) in scalar_states.iter_mut().enumerate() {
            for idx in 0..SPONGE_WIDTH {
                let value = next_field(&mut seed, lane * SPONGE_WIDTH + idx);
                scalar_state[idx] = value;
                packed_state[idx].as_slice_mut()[lane] = value;
            }
        }

        let packed_result = packed_poseidon_permutation::<F, P>(packed_state);
        for (lane, scalar_state) in scalar_states.into_iter().enumerate() {
            let expected = F::poseidon(scalar_state);
            for idx in 0..SPONGE_WIDTH {
                assert_eq!(
                    packed_result[idx].as_slice()[lane],
                    expected[idx],
                    "lane={lane}, idx={idx}"
                );
            }
        }
    }

    #[test]
    #[ignore = "manual SIMD throughput comparison; run with --ignored --nocapture"]
    fn benchmark_scalar_vs_batched_hash_or_noop() {
        if P::WIDTH < 4 {
            eprintln!("skipping: this target has no 4-lane or 8-lane field packing");
            return;
        }

        const INPUT_LEN: usize = 512;
        const ITERATIONS: usize = 20;
        let mut seed = 0xa409_3822_299f_31d0;
        let input_storage: Vec<Vec<F>> = (0..P::WIDTH)
            .map(|lane| {
                (0..INPUT_LEN)
                    .map(|idx| next_field(&mut seed, lane * INPUT_LEN + idx))
                    .collect()
            })
            .collect();
        let inputs: Vec<&[F]> = input_storage.iter().map(Vec::as_slice).collect();
        let mut outputs = vec![HashOut::ZERO; P::WIDTH];

        let scalar_start = std::time::Instant::now();
        let mut scalar_checksum = 0u64;
        for _ in 0..ITERATIONS {
            for input in &inputs {
                let hash = <PoseidonHash as Hasher<F>>::hash_or_noop(input);
                scalar_checksum = scalar_checksum
                    .wrapping_add(std::hint::black_box(hash.elements[0].to_canonical_u64()));
            }
        }
        let scalar_elapsed = scalar_start.elapsed();

        let batch_start = std::time::Instant::now();
        let mut batch_checksum = 0u64;
        for _ in 0..ITERATIONS {
            hash_or_noop_batch(&inputs, &mut outputs);
            for hash in &outputs {
                batch_checksum = batch_checksum
                    .wrapping_add(std::hint::black_box(hash.elements[0].to_canonical_u64()));
            }
            std::hint::black_box(&outputs);
        }
        let batch_elapsed = batch_start.elapsed();

        assert_eq!(scalar_checksum, batch_checksum);
        eprintln!(
            "Poseidon batch width={} input_len={} iterations={} scalar_elapsed={:?} batch_elapsed={:?} checksum={}",
            P::WIDTH,
            INPUT_LEN,
            ITERATIONS,
            scalar_elapsed,
            batch_elapsed,
            std::hint::black_box(batch_checksum),
        );
    }
}
