//! Utility module providing some helper functions.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;

/// A helper function to transpose a row-wise trace and put it in the format that `prove` expects.
pub fn trace_rows_to_poly_values<F: Field, const COLUMNS: usize>(
    trace_rows: Vec<[F; COLUMNS]>,
) -> Vec<PolynomialValues<F>> {
    plonky2::util::transpose_rows(&trace_rows)
        .into_iter()
        .map(PolynomialValues::new)
        .collect()
}
