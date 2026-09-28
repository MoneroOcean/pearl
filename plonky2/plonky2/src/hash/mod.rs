//! plonky2 hashing logic for in-circuit hashing and Merkle proof verification
//! as well as specific hash functions implementation.

mod arch;
pub mod batch_merkle_tree;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod blake3_batch_avx512;
pub mod blake3_perm;
pub mod hash_types;
pub mod hashing;
pub mod keccak;
pub mod merkle_proofs;
pub mod merkle_tree;
pub mod path_compression;
pub mod poseidon;
pub(crate) mod poseidon_batch;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bw",
    target_feature = "avx512cd",
    target_feature = "avx512dq",
    target_feature = "avx512f",
    target_feature = "avx512vl"
))]
mod poseidon_batch_avx512;
pub mod poseidon_goldilocks;
