use ff::{FromUniformBytes, PrimeField};
use super::primitives::{generate_constants, Spec};

pub const R_F: usize = 8;
pub const R_P: usize = 56;
pub const SECURE_MDS: usize = 0;


#[derive(Debug, Clone, Copy)]
pub struct PoseidonSpec;

/// The type used to hold the MDS matrix and its inverse.
pub type Mds<F, const T: usize> = [[F; T]; T];


impl<F: PrimeField, const T: usize, const R: usize> Spec<F, T, R> for PoseidonSpec 
where
F: FromUniformBytes<64> + Ord,
{
    fn full_rounds() -> usize {
        R_F
    }

    fn partial_rounds() -> usize {
        R_P
    }

    fn sbox(val: F) -> F {
        let val_sq = val * val;
        val_sq * val_sq * val
    }

    fn secure_mds() -> usize {
        SECURE_MDS
    }

    fn constants() -> (Vec<[F; T]>, Mds<F, T>, Mds<F, T>, [F; T]) {
        generate_constants::<F, PoseidonSpec, T, R>()
    }
}