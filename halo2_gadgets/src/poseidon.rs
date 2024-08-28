//! The Poseidon algebraic hash function.

use std::{convert::TryInto, mem};
use std::fmt;
use std::marker::PhantomData;
use std::time::Instant;
use ff::{FromUniformBytes, PrimeField};
use group::ff::Field;
use halo2_proofs::circuit::{SimpleFloorPlanner, Value};
use halo2_proofs::dev::MockProver;
use halo2_proofs::plonk::{Circuit, ConstraintSystem};
use halo2_proofs::{
    circuit::{AssignedCell, Chip, Layouter},
    plonk::Error,
};
use primitives::{self as poseidonInline, generate_constants, permute};

mod pow5;
use halo2curves::bn256::Fr as Fp;
pub use pow5::{Pow5Chip, Pow5Config, StateWord};

pub mod primitives;
pub mod spec;
use primitives::{Absorbing, ConstantLength, Domain, Spec, SpongeMode, Squeezing, State};
use rand::rngs::OsRng;

use self::primitives::VariableLength;
use self::spec::{PoseidonSpec, R_F, R_P, SECURE_MDS};

/// A word from the padded input to a Poseidon sponge.
#[derive(Clone, Debug)]
pub enum PaddedWord<F: Field> {
    /// A message word provided by the prover.
    Message(AssignedCell<F, F>),
    /// A padding word, that will be fixed in the circuit parameters.
    Padding(F),
}

/// The set of circuit instructions required to use the Poseidon permutation.
pub trait PoseidonInstructions<F: Field, S: Spec<F, T, RATE>, const T: usize, const RATE: usize>:
    Chip<F>
{
    /// Variable representing the word over which the Poseidon permutation operates.
    type Word: Clone + fmt::Debug + From<AssignedCell<F, F>> + Into<AssignedCell<F, F>>;

    /// Applies the Poseidon permutation to the given state.
    fn permute(
        &self,
        layouter: &mut impl Layouter<F>,
        initial_state: &State<Self::Word, T>,
    ) -> Result<State<Self::Word, T>, Error>;
}

/// The set of circuit instructions required to use the [`Sponge`] and [`Hash`] gadgets.
///
/// [`Hash`]: self::Hash
pub trait PoseidonSpongeInstructions<
    F: Field,
    S: Spec<F, T, RATE>,
    D: Domain<F, RATE>,
    const T: usize,
    const RATE: usize,
>: PoseidonInstructions<F, S, T, RATE>
{
    /// Returns the initial empty state for the given domain.
    fn initial_state(&self, layouter: &mut impl Layouter<F>)
        -> Result<State<Self::Word, T>, Error>;

    /// Adds the given input to the state.
    fn add_input(
        &self,
        layouter: &mut impl Layouter<F>,
        initial_state: &State<Self::Word, T>,
        input: &Absorbing<PaddedWord<F>, RATE>,
    ) -> Result<State<Self::Word, T>, Error>;

    /// Extracts sponge output from the given state.
    fn get_output(state: &State<Self::Word, T>) -> Squeezing<Self::Word, RATE>;
}

/// A word over which the Poseidon permutation operates.
#[derive(Debug)]
pub struct Word<
    F: Field,
    PoseidonChip: PoseidonInstructions<F, S, T, RATE>,
    S: Spec<F, T, RATE>,
    const T: usize,
    const RATE: usize,
> {
    inner: PoseidonChip::Word,
}

impl<
        F: Field,
        PoseidonChip: PoseidonInstructions<F, S, T, RATE>,
        S: Spec<F, T, RATE>,
        const T: usize,
        const RATE: usize,
    > Word<F, PoseidonChip, S, T, RATE>
{
    /// The word contained in this gadget.
    pub fn inner(&self) -> PoseidonChip::Word {
        self.inner.clone()
    }

    /// Construct a [`Word`] gadget from the inner word.
    pub fn from_inner(inner: PoseidonChip::Word) -> Self {
        Self { inner }
    }
}

fn poseidon_sponge<
    F: Field,
    PoseidonChip: PoseidonSpongeInstructions<F, S, D, T, RATE>,
    S: Spec<F, T, RATE>,
    D: Domain<F, RATE>,
    const T: usize,
    const RATE: usize,
>(
    chip: &PoseidonChip,
    mut layouter: impl Layouter<F>,
    state: &mut State<PoseidonChip::Word, T>,
    input: Option<&Absorbing<PaddedWord<F>, RATE>>,
) -> Result<Squeezing<PoseidonChip::Word, RATE>, Error> {
    if let Some(input) = input {
        *state = chip.add_input(&mut layouter, state, input)?;
    }
    *state = chip.permute(&mut layouter, state)?;
    Ok(PoseidonChip::get_output(state))
}

/// A Poseidon sponge.
#[derive(Debug, Clone)]
pub struct Sponge<
    F: Field,
    PoseidonChip: PoseidonSpongeInstructions<F, S, D, T, RATE> + Clone,
    S: Spec<F, T, RATE>,
    M: SpongeMode,
    D: Domain<F, RATE>,
    const T: usize,
    const RATE: usize,
> {
    chip: PoseidonChip,
    mode: M,
    state: State<PoseidonChip::Word, T>,
    _marker: PhantomData<D>,
}

impl<
        F: Field,
        PoseidonChip: PoseidonSpongeInstructions<F, S, D, T, RATE> + Clone,
        S: Spec<F, T, RATE>,
        D: Domain<F, RATE>,
        const T: usize,
        const RATE: usize,
    > Sponge<F, PoseidonChip, S, Absorbing<PaddedWord<F>, RATE>, D, T, RATE>
{
    /// Constructs a new duplex sponge for the given Poseidon specification.
    pub fn new(chip: PoseidonChip, mut layouter: impl Layouter<F>) -> Result<Self, Error> {
        chip.initial_state(&mut layouter).map(|state| Sponge {
            chip,
            mode: Absorbing(
                (0..RATE)
                    .map(|_| None)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            ),
            state,
            _marker: PhantomData::default(),
        })
    }

    /// Absorbs an element into the sponge.
    pub fn absorb(
        &mut self,
        mut layouter: impl Layouter<F>,
        value: PaddedWord<F>,
    ) -> Result<(), Error> {
        for entry in self.mode.0.iter_mut() {
            if entry.is_none() {
                *entry = Some(value);
                return Ok(());
            }
        }

        // We've already absorbed as many elements as we can
        let _ = poseidon_sponge(
            &self.chip,
            layouter.namespace(|| "PoseidonSponge"),
            &mut self.state,
            Some(&self.mode),
        )?;
        self.mode = Absorbing::init_with(value);

        Ok(())
    }

    /// Transitions the sponge into its squeezing state.
    #[allow(clippy::type_complexity)]
    pub fn finish_absorbing(
        mut self,
        mut layouter: impl Layouter<F>,
    ) -> Result<Sponge<F, PoseidonChip, S, Squeezing<PoseidonChip::Word, RATE>, D, T, RATE>, Error>
    {   
        let mode = poseidon_sponge(
            &self.chip,
            layouter.namespace(|| "PoseidonSponge"),
            &mut self.state,
            Some(&self.mode),
        )?;
        Ok(Sponge {
            chip: self.chip,
            mode,
            state: self.state,
            _marker: PhantomData::default(),
        })
    }
}

impl<
        F: Field,
        PoseidonChip: PoseidonSpongeInstructions<F, S, D, T, RATE> + Clone,
        S: Spec<F, T, RATE>,
        D: Domain<F, RATE>,
        const T: usize,
        const RATE: usize,
    > Sponge<F, PoseidonChip, S, Squeezing<PoseidonChip::Word, RATE>, D, T, RATE>
{
    /// Squeezes an element from the sponge.
    pub fn squeeze(&mut self, mut layouter: impl Layouter<F>) -> Result<AssignedCell<F, F>, Error> {
        loop {
            for entry in self.mode.0.iter_mut() {
                if let Some(inner) = entry.take() {
                    return Ok(inner.into());
                }
            }
            // We've already squeezed out all available elements
            self.mode = poseidon_sponge(
                &self.chip,
                layouter.namespace(|| "PoseidonSponge"),
                &mut self.state,
                None,
            )?;
        }
    }

    /// Transitions the sponge into its squeezing state.
    #[allow(clippy::type_complexity)]
    pub fn finish_squeezing(
        mut self,
        squeezed_cell: AssignedCell<F, F>,
        mut layouter: impl Layouter<F>,
    ) -> Result<Sponge<F, PoseidonChip, S, Absorbing<PaddedWord<F>, RATE>, D, T, RATE>, Error>
    {   
        let mode = Absorbing::init_with(PaddedWord::Message(squeezed_cell));
        Ok(Sponge {
            chip: self.chip,
            mode,
            state: self.state,
            _marker: PhantomData::default(),
        })
    }
}

/// A Poseidon hash function, built around a sponge.
#[derive(Debug)]
pub struct Hash<
    F: Field,
    PoseidonChip: PoseidonSpongeInstructions<F, S, D, T, RATE> + Clone,
    S: Spec<F, T, RATE>,
    D: Domain<F, RATE>,
    const T: usize,
    const RATE: usize,
> {
    sponge: Sponge<F, PoseidonChip, S, Absorbing<PaddedWord<F>, RATE>, D, T, RATE>,
}

impl<
        F: Field,
        PoseidonChip: PoseidonSpongeInstructions<F, S, D, T, RATE> + Clone,
        S: Spec<F, T, RATE>,
        D: Domain<F, RATE>,
        const T: usize,
        const RATE: usize,
    > Hash<F, PoseidonChip, S, D, T, RATE>
{
    /// Initializes a new hasher.
    pub fn init(chip: PoseidonChip, layouter: impl Layouter<F>) -> Result<Self, Error> {
        Sponge::new(chip, layouter).map(|sponge| Hash { sponge })
    }
}

impl<
        F: PrimeField,
        PoseidonChip: PoseidonSpongeInstructions<F, S, ConstantLength<L>, T, RATE> + Clone,
        S: Spec<F, T, RATE>,
        const T: usize,
        const RATE: usize,
        const L: usize,
    > Hash<F, PoseidonChip, S, ConstantLength<L>, T, RATE>
{
    /// Hashes the given input.
    pub fn hash(
        mut self,
        mut layouter: impl Layouter<F>,
        message: [AssignedCell<F, F>; L],
    ) -> Result<AssignedCell<F, F>, Error> {
        let time = Instant::now();
        for (i, value) in message
            .into_iter()
            .map(PaddedWord::Message)
            .chain(<ConstantLength<L> as Domain<F, RATE>>::padding(L).map(PaddedWord::Padding))
            .enumerate()
        {   
            self.sponge
                .absorb(layouter.namespace(|| format!("absorb_{}", i)), value)?;
        }
        println!("Time taken to absorb: {:?}", time.elapsed());
        let time = Instant::now();
        let hash = self.sponge
            .finish_absorbing(layouter.namespace(|| "finish absorbing"))?
            .squeeze(layouter.namespace(|| "squeeze"));
        println!("Time taken to squeeze: {:?}", time.elapsed());
        hash
    }
}

/// Statefull Poseidon hasher.
#[derive(Clone, Debug)]
pub struct PoseidonHash<F: PrimeField, const T: usize, const RATE: usize> {
    init_state: State<F, T>,
    state: State<F, T>,
    spec: PoseidonSpec,
    absorbing: Vec<F>,
}

impl<F: PrimeField, const T: usize, const RATE: usize> PoseidonHash<F, T, RATE> 
where
    F: FromUniformBytes<64> + Ord,
{

    /// Create new Poseidon hasher.
    pub fn new<const R_F: usize, const R_P: usize, const SECURE_MDS: usize>() -> Self {
        let init_state = [F::ZERO; T];
        Self {
            spec: PoseidonSpec,
            state: init_state,
            init_state,
            absorbing: Vec::new(),
        }
    }

    /// Initialize a poseidon hasher from an existing spec.
    pub fn from_spec(spec: PoseidonSpec) -> Self {
        let init_state = [F::ZERO; T];
        Self { spec, state: init_state, init_state, absorbing: Vec::new() }
    }

    /// Reset state to default and clear the buffer.
    pub fn clear(&mut self) {
        self.state = self.init_state;
        self.absorbing.clear();
    }

    pub fn permutation(&mut self) {
        let (round_constants, mds, _mds_inv, _) 
            = generate_constants::<F, PoseidonSpec, T, RATE>();
        permute::<F, PoseidonSpec, T, RATE>(&mut self.state, &mds, &round_constants);
    }

    /// Appends elements to the absorption line updates state while `RATE` is
    /// full
    pub fn update(&mut self, elements: &[F]) {
        let mut input_elements = self.absorbing.clone();
        input_elements.extend_from_slice(elements);

        for chunk in input_elements.chunks(RATE) {
            if chunk.len() < RATE {
                // Must be the last iteration of this update. Feed unpermutaed inputs to the
                // absorbation line
                self.absorbing = chunk.to_vec();
            } else {
                // Add new chunk of inputs for the next permutation cycle.
                for (input_element, state) in chunk.iter().zip(self.state.iter_mut().skip(1)) {
                    state.add_assign(input_element);
                }
                // Perform intermediate permutation
                self.permutation();
                // Flush the absorption line
                self.absorbing.clear();
            }
        }
    }

    /// Results a single element by absorbing already added inputs
    pub fn squeeze(&mut self) -> F {
        let mut last_chunk = self.absorbing.clone();
        {
            // Expect padding offset to be in [0, RATE)
            debug_assert!(last_chunk.len() < RATE);
        }
        // Add the finishing sign of the variable length hashing. Note that this mut
        // also apply when absorbing line is empty
        last_chunk.push(F::ONE);
        // Add the last chunk of inputs to the state for the final permutation cycle

        for (input_element, state) in last_chunk.iter().zip(self.state.iter_mut().skip(1)) {
            state.add_assign(input_element);
        }

        // Perform final permutation
        self.permutation();
        // Flush the absorption line
        self.absorbing.clear();
        // Returns the challenge while preserving internal state
        self.state[1]
    }
}


#[derive(Clone, Debug)]
pub(crate) struct PoseidonState<F: PrimeField, const T: usize, const RATE: usize> {
    pub(crate) s: [AssignedCell<F, F>; T],
}

// impl<F: PrimeField, const T: usize, const RATE: usize> PoseidonState<F, T, RATE> {
//     pub fn default() -> Self {
//         let mut default_state = [F::ZERO; T];
//         // from Section 4.2 of https://eprint.iacr.org/2019/458.pdf
//         // • Variable-Input-Length Hashing. The capacity value is 2^64 + (o−1) where o the output length.
//         // for our transcript use cases, o = 1
//         default_state[0] = F::from_u128(1u128 << 64);
//         Self { s: default_state.map(|f| ctx.load_constant(f)) }
//     }
// }

#[derive(Clone, Debug)]
/// Poseidon sponge. This is stateful.
pub struct PoseidonSpongeChip<F: PrimeField, const T: usize, const RATE: usize> {
    pub chip: Pow5Chip<F, T, RATE>,
    init_state: State<StateWord<F>, T>,
    state: State<StateWord<F>, T>,
    spec: PoseidonSpec,
    absorbing: Vec<AssignedCell<F, F>>,
}

impl<F: PrimeField, const T: usize, const RATE: usize> PoseidonSpongeChip<F, T, RATE> {
    /// Create new Poseidon hasher.
    pub fn new<const R_F: usize, const R_P: usize, const SECURE_MDS: usize>(
        chip: Pow5Chip<F,T, RATE >, layouter: impl Layouter<F>) -> Self {
        let init_state = chip.initial_state(layouter).unwrap();
        let state = init_state.clone();
        Self {
            chip,
            init_state,
            state,
            spec: PoseidonSpec,//::new::<R_F, R_P, SECURE_MDS>(),
            absorbing: Vec::new(),
        }
    }

    /// Initialize a poseidon hasher from an existing spec.
    pub fn from_spec(chip: Pow5Chip<F, T, RATE>, layouter: impl Layouter<F>, spec: PoseidonSpec) -> Self {
        let init_state = chip.initial_state(layouter).unwrap();
        let state = init_state.clone();
        Self {
            chip,
            init_state,
            state,
            spec,
            absorbing: Vec::new(),
        }
    }

    /// Reset state to default and clear the buffer.
    pub fn clear(&mut self) {
        self.state = self.init_state.clone();
        self.absorbing.clear();
    }

    /// Store given `elements` into buffer.
    pub fn update(&mut self, elements: &[AssignedCell<F, F>]) {
        self.absorbing.extend_from_slice(elements);
    }

    /// Consume buffer and perform permutation, then output second element of
    /// state.
    pub fn squeeze(
        &mut self,
        mut layouter: impl Layouter<F>,
    ) -> Result<AssignedCell<F, F>, Error> {
        let input_elements = mem::take(&mut self.absorbing);

        for (i, input_chunk) in input_elements.chunks(RATE).enumerate() {
            let chunk_len = input_chunk.len();
            if chunk_len < RATE {

                let mut padded_chunk = input_chunk.iter().cloned().map(|cell| PaddedWord::Message(cell.clone())).collect::<Vec<_>>();
                    padded_chunk.extend(<VariableLength<F, RATE> as Domain<F, RATE>>::padding(chunk_len).iter().cloned().map(PaddedWord::Padding));

                let padded_chunk_array: [PaddedWord<F>; RATE] = padded_chunk.try_into().unwrap();
                self.state = self.chip.add_input(layouter.namespace(|| format!("absorb_{i}")), &self.init_state, &padded_chunk_array)?;
                self.state = self.chip.permutation(layouter.namespace(|| format!("absorb_{i}")), &self.state)?;

            } else {

                let input_chunk: [PaddedWord<F>; RATE] = input_chunk.iter().cloned().map(|cell| PaddedWord::Message(cell.clone())).collect::<Vec<_>>().try_into().unwrap();
                self.state = self.chip.add_input(layouter.namespace(|| format!("absorb_{i}")), &self.init_state, &input_chunk)?;
                self.state = self.chip.permutation(layouter.namespace(|| format!("absorb_{i}")), &self.state)?;
            }
        }

        let hash = self.state[1].0.clone();
        self.update(&[hash.clone()]);
        Ok(hash)
    }
}
    struct HashCircuit<
        S: Spec<Fp, WIDTH, RATE>,
        const WIDTH: usize,
        const RATE: usize,
        const L: usize,
    > {
        message: Value<[Fp; L]>,
        // For the purpose of this test, witness the result.
        // TODO: Move this into an instance column.
        output: Value<Fp>,
        _spec: PhantomData<S>,
    }

    impl<S: Spec<Fp, WIDTH, RATE>, const WIDTH: usize, const RATE: usize, const L: usize>
        Circuit<Fp> for HashCircuit<S, WIDTH, RATE, L>
    {
        type Config = Pow5Config<Fp, WIDTH, RATE>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self {
                message: Value::unknown(),
                output: Value::unknown(),
                _spec: PhantomData,
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fp>) -> Pow5Config<Fp, WIDTH, RATE> {
            let state = (0..WIDTH).map(|_| meta.advice_column()).collect::<Vec<_>>();
            let partial_sbox = meta.advice_column();

            let rc_a = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();
            let rc_b = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();

            meta.enable_constant(rc_b[0]);

            Pow5Chip::configure::<S>(
                meta,
                state.try_into().unwrap(),
                partial_sbox,
                rc_a.try_into().unwrap(),
                rc_b.try_into().unwrap(),
            )
        }

        fn synthesize(
            &self,
            config: Pow5Config<Fp, WIDTH, RATE>,
            mut layouter: impl Layouter<Fp>,
        ) -> Result<(), Error> {
            let chip = Pow5Chip::construct(config.clone());

            let message = layouter.assign_region(
                || "load message",
                |mut region| {
                    let mut message_word = |i: usize, j: usize| {
                        let idx = j * RATE + i;
                        let value = self.message.map(|message_vals| message_vals[idx]);
                        region.assign_advice(
                            || format!("load message_{}", i),
                            config.state[i],
                            j,
                            || value,
                        )
                    };

                    let message: Result<Vec<_>, Error> = ((0..L).map(|i| i % RATE)).zip((0..L).map(|i| i / RATE)).map(|(i, j)| message_word(i, j)).collect();
                    Ok(message.unwrap())
                    // Ok(message?.try_into().unwrap())
                },
            )?;

            // let value = vec![Fp::ZERO; L];
            // let message = region.assign_advice(
            //     || "load message",
            //     config.state[0],
            //     0,
            //     || self.message,
            // )?;

            let mut transcript = PoseidonSpongeChip::new::<R_F, R_P, SECURE_MDS>(
                chip,
                layouter.namespace(|| "init"),
            );
            transcript.update(&message);
            let output = transcript.squeeze(layouter.namespace(|| "hash"));
            transcript.update(&message);
            let output2 = transcript.squeeze(layouter.namespace(|| "hash2"));
            println!("output: {:?}", output);
            println!("output2: {:?}", output2);
            // layouter.assign_region(
            //     || "constrain output",
            //     |mut region| {
            //         let expected_var = region.assign_advice(
            //             || "load output",
            //             config.state[0],
            //             0,
            //             || self.output,
            //         )?;
            //         region.constrain_equal(output.cell(), expected_var.cell())
            //     },
            // )
            Ok(())
        }
    }

    #[test]
    fn poseidon_hash() {
        // all three are diff wtf?
        let rng = OsRng;
        let message = [Fp::from(6), Fp::from(5), Fp::from(4), Fp::from(3), Fp::from(2)];
        let output =
        poseidonInline::Hash::<_, PoseidonSpec, VariableLength<Fp, 2>, 3, 2>::init().hash(&message);
        println!("outputInline: {:?}", output);

        let mut poseidonInline = PoseidonHash::<Fp, 3, 2>::new::<R_F, R_P, SECURE_MDS>();
        poseidonInline.update(&message);
        let output = poseidonInline.squeeze();
        println!("poseidonhash: {:?}", output);

        let k = 9;
        let circuit = HashCircuit::<PoseidonSpec, 3, 2, 5> {
            message: Value::known(message),
            output: Value::known(output),
            _spec: PhantomData,
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));

    }

    #[cfg(feature = "test-dev-graph")]
    #[test]
    fn poseidon_hash_graph() {
        use plotters::prelude::*;

        let root = BitMapBackend::new("PoseidonChip.png", (1024, 768)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let root = root
            .titled("Poseidon Chip Layout", ("sans-serif", 60))
            .unwrap();

        let rng = OsRng;
        let message = [Fp::from(6), Fp::from(5), Fp::from(4), Fp::from(3), Fp::from(2)];
        let output =
            poseidonInline::Hash::<_, PoseidonSpec, VariableLength<Fp, 2>, 3, 2>::init().hash(&message);
        println!("outputInline: {:?}", output);

        let k = 8;
        let circuit = HashCircuit::<PoseidonSpec, 3, 2, 5> {
            message: Value::known(message),
            output: Value::known(output),
            _spec: PhantomData,
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));

        halo2_proofs::dev::CircuitLayout::default()
        .render(8, &circuit, &root)
        .unwrap();
    }