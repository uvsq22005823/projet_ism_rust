use std::env;
use std::fs::File;
use std::io::{self, BufRead};
use rand::{Rng, SeedableRng};

// NOTE:
// This is double precision, change it if you change floating point precision!
const COMPUTER_PRECISION: f64 = 1E-16;

// NOTE: there's 1000 particles in total in our dataset
//    Since it is fix, we could use arrays instead of vectors
//    I still let vectors 'cause I want to be able to run on other inputs
const VECTOR_DEFAULT_ALLOCATION: usize = 1000;

// Constants
const R_ETOILE   : f64 = 3.0;
const R_ETOILE_SQ: f64 = R_ETOILE * R_ETOILE;
const EPS_ETOILE : f64 = 0.2;
const CONST_LJ_48: f64 = -48.0;
const CONST_LJ_MULT_EPS_ETOILE: f64 = CONST_LJ_48 * EPS_ETOILE;
const RAYON_COUPURE: f64 = 10.0;
const SQUARED_RAYON_COUPURE: f64 = RAYON_COUPURE * RAYON_COUPURE;
const R_ETOILE_ENERGY_PRECOMPUTED: f64 = EPS_ETOILE * 4.0 / 2.0;

// Box's dimensions (for periodical conditions)
//  L_X = L_Y = L_Z = L (because it's a box I guess?)
//  Note that the code will only use L
// const L_X: f64 = 42.0;
// const L_Y: f64 = 42.0;
// const L_Z: f64 = 42.0;
const L: f64 = 42.0;
const PERIODIC_IMAGES_AMOUNT: usize = 27;
const TRANSLATION_VECTORS: [[f64; 3]; PERIODIC_IMAGES_AMOUNT] =
[
  [0.0,   0.0,   0.0],
  [0.0,   0.0,    L ],
  [0.0,   0.0,   -L ],
  [0.0,    L ,   0.0],
  [0.0,    L ,    L ],
  [0.0,    L ,   -L ],
  [0.0,   -L ,   0.0],
  [0.0,   -L ,    L ],
  [0.0,   -L ,   -L ],
  [ L ,   0.0,   0.0],
  [ L ,   0.0,    L ],
  [ L ,   0.0,   -L ],
  [ L ,    L ,   0.0],
  [ L ,    L ,    L ],
  [ L ,    L ,   -L ],
  [ L ,   -L ,   0.0],
  [ L ,   -L ,    L ],
  [ L ,   -L ,   -L ],
  [-L ,   0.0,   0.0],
  [-L ,   0.0,    L ],
  [-L ,   0.0,   -L ],
  [-L ,    L ,   0.0],
  [-L ,    L ,    L ],
  [-L ,    L ,   -L ],
  [-L ,   -L ,   0.0],
  [-L ,   -L ,    L ],
  [-L ,   -L ,   -L ],
] ;

// Constants for Verlet algorithm
// const DT: f64 = 1.0;
const SIMULATION_TEMPERATURE: f64 = 300.0;
const CONVERSION_FORCE: f64 = -0.0001 * 4.186;
const PARTICULE_MASS: f64 = 18.0;
const CONSTANTE_R: f64 = 0.00199;
// Precomputing some stuff for temperature_computation function
const PRE_CONVERSION_FORCE: f64 = 1.0 / (2.0 * CONVERSION_FORCE);

// SoA
struct Particles<'part>
{
  x_dim: &'part mut Vec<f64>,
  y_dim: &'part mut Vec<f64>,
  z_dim: &'part mut Vec<f64>
}


// Equivalent of the SIGN function in Fortran
fn signe(a: f64, b: f64) -> f64
{
  match b >= 0.0
  {
    true  =>  a.abs(),
    false => -a.abs(),
  }
}


// Computes system's kinetic energy
// Returns temperature if return_temp is true and kinetic_energy if it is false
fn temperature_computation(cin_mov: &mut Particles, taille_vect: usize,
                           pre_temperature: f64, return_temp: bool) -> f64
{
  let mut kinetic_energy: f64 = 0.0;

  // Compute system's kinetic energy
  for i in 0..taille_vect
  {
    let mut partial_sum: f64 = 0.0;
    partial_sum += cin_mov.x_dim[i] * cin_mov.x_dim[i];
    partial_sum += cin_mov.y_dim[i] * cin_mov.y_dim[i];
    partial_sum += cin_mov.z_dim[i] * cin_mov.z_dim[i];
    partial_sum /= PARTICULE_MASS;  // TODO: test * 1/PARTICULE_MASS
    kinetic_energy += partial_sum;
  }
  kinetic_energy  *= PRE_CONVERSION_FORCE;

  match return_temp
  {
    true => pre_temperature * kinetic_energy,
    false => kinetic_energy
  }
}


// Returns system's energy
// Forces are updated during the function
fn energy_computation(dims: &Particles,
                      forces: &mut Particles, taille_vect: usize) -> f64
{
  let mut energy: f64 = 0.0;  // Lennard Jones term accumulator
  for translation_vector in TRANSLATION_VECTORS
  {
    for i in 0..taille_vect
    {
      // Fetching particule's position
      let x_i = dims.x_dim[i];
      let y_i = dims.y_dim[i];
      let z_i = dims.z_dim[i];

      for j in 0..taille_vect
      {
        // Fetching other particule's position
        // TRANSLATION_VECTORS[i][k], k = {x, y, z}
        let x_j = dims.x_dim[j] + translation_vector[0];
        let y_j = dims.y_dim[j] + translation_vector[1];
        let z_j = dims.z_dim[j] + translation_vector[2];

        // Precalculing some stuff that is reused later
        let x_ij = x_i - x_j;
        let y_ij = y_i - y_j;
        let z_ij = z_i - z_j;

        // Checking if it is necessary to compute this iteration
        let squared_r_ij: f64 = x_ij * x_ij + y_ij * y_ij + z_ij * z_ij;
        if squared_r_ij > SQUARED_RAYON_COUPURE
        {
          continue;
        }
        if squared_r_ij == 0.0
        {
          // panic!("DIVISION BY ZERO");
          // Let's just skip the iteration for now
          continue;
        }

        // Main computation
        let r_2 : f64 = R_ETOILE_SQ / squared_r_ij;
        let r_4 : f64 = r_2 * r_2;
        let r_6 : f64 = r_4 * r_2;
        let r_8 : f64 = r_4 * r_4;
        let r_12: f64 = r_8 * r_4;
        let r_14: f64 = r_12 * r_2;
        energy += r_12 - (r_6 + r_6);  // Computing Lennard Jones term


        // Updating forces
        let this_force: f64 = CONST_LJ_MULT_EPS_ETOILE * (r_14 - r_8);
        /*
        0 1 2 i 3 4 5

        accumulates in i forces with interaction (i, 3..5)
        accumulates in 3..5 forces with interaction (3..5, i)
        */

        // Some small precomputing, might not be useful
        let this_force_x = this_force * x_ij;
        let this_force_y = this_force * y_ij;
        let this_force_z = this_force * z_ij;

        // Accumulating forces for other elements with this one
        // forces.x_dim[j] -= this_force_x;
        // forces.y_dim[j] -= this_force_y;
        // forces.z_dim[j] -= this_force_z;

        // Accumulating forces for this element with the others
        forces.x_dim[i] += this_force_x;
        forces.y_dim[i] += this_force_y;
        forces.z_dim[i] += this_force_z;
      }
    }
  }

  energy * R_ETOILE_ENERGY_PRECOMPUTED
}


// Result should be < precision
fn compute_forces(forces: &Particles, taille: usize) -> f64
{
  let mut somme_forces: f64 = 0.0;
  for i in 0..taille
  {
    somme_forces += forces.x_dim[i];
    somme_forces += forces.y_dim[i];
    somme_forces += forces.z_dim[i];
  }
  somme_forces
}


// Naive (probably very slow) function to check duplicates in input
fn check_input(positions: &Particles, len: usize) -> Option<(usize, usize)>
{
  // Checking for each particle if there is one on the same coordinates
  //  Two particles cannot physically be at the same place
  for particule_i in 0..len
  {
    for particule_j in particule_i+1..len
    {
      if positions.x_dim[particule_i] == positions.x_dim[particule_j] &&
      positions.y_dim[particule_i] == positions.y_dim[particule_j] &&
      positions.z_dim[particule_i] == positions.z_dim[particule_j]
      {
        return Some((particule_i, particule_j));
      }
    }
  }
  None
}


fn main()
{
  // ------------------------------- Initialization-----------------------------

  // Checking program call
  let args: Vec<String> = env::args().collect();
  if args.len() != 2 && args.len() != 3
  {
    // It might be possible to factorize that and put everything in panic! call?
    eprintln!("ERROR: Wrong number of arguments!");
    eprintln!("\tUsage: cargo run --release -- file [seed]");
    eprintln!("\tUsage (without cargo): {} file [seed]", args[0]);
    panic!("Incorrect call to program.");
  }
  // Attempting to open input file
  let file_path = &args[1];
  let file = match File::open(file_path)
  {
    Ok(file) => file,
    Err(error) => match error.kind()
    {
      std::io::ErrorKind::NotFound => panic!("File not found!"),
      _ => panic!("Unknown error upon opening file."),
    },
  };
  let mut reader = io::BufReader::new(file);

  // Checking seed if one
  let mut seed: u64 = 0;  // Default seed is 0, might change that later on
  if args.len() == 3
  {
    match args[2].parse()
    {
      Ok(value) => seed = value,
      Err(_e) => eprintln!("Error: {} is not a valid seed, using default seed.",
                           args[2])
    }
  }
  println!("Starting program using seed = {}", seed);

  // Init PRNG
  let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
  // println!("Random f64: {}", rng.random::<f64>());

  // Skipping first line 'cause it's not important
  // WARNING: it might be better to really define what commented lines should be
  let mut my_string = String::new();
  reader.read_line(&mut my_string).expect("This shouldn't fail, right?");

  // Creating my vectors ; they have a default capacity that fits our data
  let mut x_vector: Vec<f64> = Vec::with_capacity(VECTOR_DEFAULT_ALLOCATION);
  let mut y_vector: Vec<f64> = Vec::with_capacity(VECTOR_DEFAULT_ALLOCATION);
  let mut z_vector: Vec<f64> = Vec::with_capacity(VECTOR_DEFAULT_ALLOCATION);

  let positions = Particles
  {
    x_dim: &mut x_vector,
    y_dim: &mut y_vector,
    z_dim: &mut z_vector,
  };

  // Reading each line and adding value to the correct vector
  for line in reader.lines()
  {
    let this_line = match line
    {
      Ok(line) => line,
      Err(_) => panic!("An error occured upon reading input file!"),
    };
    let mut iter = this_line.split_whitespace();

    // Format:
    //   Type X Y Z
    //      Type is disregarded for this exercise
    for i in 0..4
    {
      match iter.next()
      {
        Some(value) => match i
        {
          // WARNING: We might want to increase capacity!
          //      => Add it as an input parameter?
          //      => Get number of lines first?
          1 => &positions.x_dim.push(value.parse::<f64>().unwrap()),
          2 => &positions.y_dim.push(value.parse::<f64>().unwrap()),
          3 => &positions.z_dim.push(value.parse::<f64>().unwrap()),
          _ => &()
        },
        None => panic!("Values are not using the expected format!"),
      };
    }
  }

  // Checking input and calculating precision
  let taille: usize = positions.x_dim.len();
  if let Some((a, b)) = check_input(&positions, taille)
  {
    panic!("Particle number {} is a duplicate of number {} !", a, b);
  }
  let precision: f64 = COMPUTER_PRECISION * (taille as f64);

  // Creating forces vectors
  // MIGHT BE BETTER TO USE ARRAYS (TOCHECK)
  // Since we access it x y z, it might be better to go AoS
  let mut fx_vector: Vec<f64> = Vec::with_capacity(taille);
  let mut fy_vector: Vec<f64> = Vec::with_capacity(taille);
  let mut fz_vector: Vec<f64> = Vec::with_capacity(taille);

  // Initializing forces vectors
  for _ in 0..taille
  {
    // Initializing values so that there's something
    fx_vector.push(0.0);
    fy_vector.push(0.0);
    fz_vector.push(0.0);
  }

  let mut forces = Particles
  {
    x_dim: &mut fx_vector,
    y_dim: &mut fy_vector,
    z_dim: &mut fz_vector,
  };

  // Creating kinetic movement vectors
  let mut px_vector: Vec<f64> = Vec::with_capacity(taille);
  let mut py_vector: Vec<f64> = Vec::with_capacity(taille);
  let mut pz_vector: Vec<f64> = Vec::with_capacity(taille);

  // Initializing kinetic movement vectors
  for _ in 0..taille
  {
    // Initializing values with random number
    px_vector.push(signe(1.0, 0.5 - rng.random::<f64>())
                   * rng.random::<f64>());
    py_vector.push(signe(1.0, 0.5 - rng.random::<f64>())
                   * rng.random::<f64>());
    pz_vector.push(signe(1.0, 0.5 - rng.random::<f64>())
                   * rng.random::<f64>());
  }

  let mut cin_mov = Particles
  {
    x_dim: &mut px_vector,
    y_dim: &mut py_vector,
    z_dim: &mut pz_vector,
  };

  let freedom_degree : f64 = 3.0 * (taille as f64) - 3.0;
  let pre_temperature: f64 = 1.0 / (freedom_degree * CONSTANTE_R);
  // Ratio for updating cin_mov vectors
  let pre_ratio: f64 = freedom_degree * CONSTANTE_R * SIMULATION_TEMPERATURE;

  // Update kinetic movement
  let kinetic_energy = temperature_computation(&mut cin_mov, taille,
                                               pre_temperature, false);
  let rapport: f64 = pre_ratio / kinetic_energy;
  let mut sum_x: f64 = 0.0;
  let mut sum_y: f64 = 0.0;
  let mut sum_z: f64 = 0.0;
  for i in 0..taille
  {
    // Updating kinetic movement and computing their centers
    cin_mov.x_dim[i] *= rapport;
    sum_x += cin_mov.x_dim[i];
    cin_mov.y_dim[i] *= rapport;
    sum_y += cin_mov.y_dim[i];
    cin_mov.z_dim[i] *= rapport;
    sum_z += cin_mov.z_dim[i];
  }
  // Computing correction factors
  sum_x /= taille as f64;
  sum_y /= taille as f64;
  sum_z /= taille as f64;
  // Re-updating kinetic movement according to correction factor
  for i in 0..taille
  {
    cin_mov.x_dim[i] = (cin_mov.x_dim[i] - sum_x) * rapport;
    cin_mov.y_dim[i] = (cin_mov.y_dim[i] - sum_y) * rapport;
    cin_mov.z_dim[i] = (cin_mov.z_dim[i] - sum_z) * rapport;
  }

  // ---------------------------------------------------------------------------



  // -------------------------------- Computation ------------------------------

  let current_energy: f64 = energy_computation(&positions, &mut forces, taille);
  let somme_forces  : f64 = compute_forces(&forces, taille);
  print!("Number of elements : {} ; ", taille);
  print!("Computer precision : {:e} ; ", COMPUTER_PRECISION);
  println!("Current precision : {:e}", precision);
  if somme_forces.abs() < precision
  {
    println!("Sum of forces = {:e} (==0)", somme_forces);
  }
  else
  {
    println!("Sum of forces = {:e} (!=0)", somme_forces);
  }
  println!("System energy is {}", current_energy);

  // ---------------------------------------------------------------------------
}
