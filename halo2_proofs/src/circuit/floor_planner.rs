//! Implementations of common circuit floor planners.

pub(super) mod single_pass;
use std::collections::HashMap;
use crate::circuit::floor_planner::folding::strategy::Allocations;

#[derive(Debug, Clone)]
pub struct FloorPlannerData {
    pub column_allocations: HashMap<RegionColumn, Allocations>,
    pub regions: Vec<RegionStart>,
    pub first_unassigned_row: usize,
}

mod v1;
mod folding;

pub use v1::{V1Pass, V1};
pub use folding::{Folding, FoldingPass, FoldingPlan, MeasurementPass};

use super::{layouter::RegionColumn, RegionStart};
