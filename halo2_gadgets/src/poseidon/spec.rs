use ff::{Field, FromUniformBytes, PrimeField};
use super::primitives::{generate_constants, Spec};

pub const R_F: usize = 8;
pub const R_P: usize = 56;
pub const SECURE_MDS: usize = 0;


#[derive(Debug, Clone)]
pub struct PoseidonSpec<F, const T: usize, const RATE: usize>{
    pub round_constants: Vec<[F; T]>,
    pub mds_matrix: Mds<F, T>,
    pub mds_inverse: Mds<F, T>,
    pub mds_diagonal: [F; T],
}

impl<F: FromUniformBytes<64> + Ord, const T: usize, const RATE: usize> PoseidonSpec<F, T, RATE>{
    pub fn new() -> Self{
        let (round_constants, mds_matrix, mds_inverse, mds_diagonal) = generate_constants::<F, PoseidonSpec<F, T, RATE>, T, RATE>();
        Self {
            round_constants, mds_matrix, mds_inverse, mds_diagonal
        }
    }

    pub fn from_spec(spec: PoseidonSpec<F, T, RATE>) -> Self{
        spec
    }
}

/// The type used to hold the MDS matrix and its inverse.
pub type Mds<F, const T: usize> = [[F; T]; T];

impl<F: Field, const T: usize, const RATE: usize> Spec<F, T, RATE> for PoseidonSpec<F, T, RATE>
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
        let (round_constants, mds_matrix, mds_inverse, mds_diagonal) = generate_constants::<F, Self, T, RATE>();
        (round_constants, mds_matrix, mds_inverse, mds_diagonal)    }
}