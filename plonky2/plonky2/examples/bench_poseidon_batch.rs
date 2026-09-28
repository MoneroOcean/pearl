//! Single-threaded scalar/batched Poseidon comparison (no proof generation).
use std::hint::black_box;
use std::time::Instant;

use plonky2::field::goldilocks_field::GoldilocksField as F;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hash_types::HashOut;
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::plonk::config::Hasher;

fn main() {
    let width = <PoseidonHash as Hasher<F>>::hash_batch_size();
    let iterations = 512;
    for length in [8, 135, 512, 1594] {
        let storage: Vec<Vec<F>> = (0..width)
            .map(|lane| {
                (0..length)
                    .map(|i| {
                        F::from_noncanonical_u64(
                            (i as u64 * 1009 + lane as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                        )
                    })
                    .collect()
            })
            .collect();
        let inputs: Vec<&[F]> = storage.iter().map(Vec::as_slice).collect();
        let mut outputs = vec![HashOut::<F>::ZERO; width];
        <PoseidonHash as Hasher<F>>::hash_or_noop_batch(&inputs, &mut outputs);
        for (input, output) in inputs.iter().zip(&outputs) {
            assert_eq!(<PoseidonHash as Hasher<F>>::hash_or_noop(input), *output);
        }
        for sample in 0..3 {
            let scalar_start = Instant::now();
            let mut scalar_checksum = 0u64;
            for _ in 0..iterations {
                for input in &inputs {
                    let hash = <PoseidonHash as Hasher<F>>::hash_or_noop(black_box(input));
                    scalar_checksum = scalar_checksum
                        .wrapping_add(black_box(hash.elements[0].to_canonical_u64()));
                }
            }
            let scalar = scalar_start.elapsed().as_secs_f64();
            let batch_start = Instant::now();
            let mut batch_checksum = 0u64;
            for _ in 0..iterations {
                <PoseidonHash as Hasher<F>>::hash_or_noop_batch(black_box(&inputs), &mut outputs);
                for hash in &outputs {
                    batch_checksum =
                        batch_checksum.wrapping_add(black_box(hash.elements[0].to_canonical_u64()));
                }
            }
            let batch = batch_start.elapsed().as_secs_f64();
            assert_eq!(scalar_checksum, batch_checksum);
            println!("width={width} fields={length} sample={sample} scalar_seconds={scalar:.6} batch_seconds={batch:.6} speedup={:.3}", scalar / batch);
        }
    }
}
