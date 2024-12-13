use std::env;
use std::fs::File;
use std::io::{self, BufRead};

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

// SoA
struct Particles<'part>
{
  x_dim: &'part mut Vec<f64>,
  y_dim: &'part mut Vec<f64>,
  z_dim: &'part mut Vec<f64>
}


// Returns system's energy
// Forces are updated during the function
fn energy_computation(dims: &Particles,
                      forces: &mut Particles, taille_vect: usize) -> f64
{
  let mut energy: f64 = 0.0;  // LJ
  // let mut forces_somme: f64 = 0.0;  // Forces accumalor
  for i_sym in 0..PERIODIC_IMAGES_AMOUNT
  {
    for i in 0..taille_vect
      {
        // Fetching particule's position
        // TRANSLATION_VECTORS[i][k], k = {x, y, z}
        let x_i = dims.x_dim[i];
        let y_i = dims.y_dim[i];
        let z_i = dims.z_dim[i];

        for j in 0..taille_vect
        {
          // Fetching other particule's position
          let x_j = dims.x_dim[j] + TRANSLATION_VECTORS[i_sym][0];
          let y_j = dims.y_dim[j] + TRANSLATION_VECTORS[i_sym][1];
          let z_j = dims.z_dim[j] + TRANSLATION_VECTORS[i_sym][2];

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
          forces.x_dim[j] -= this_force_x;
          forces.y_dim[j] -= this_force_y;
          forces.z_dim[j] -= this_force_z;

          // Accumulating forces for this element with the others
          forces.x_dim[i] += this_force_x;
          forces.y_dim[i] += this_force_y;
          forces.z_dim[i] += this_force_z;


          // Computing Lennard Jones term
          energy += r_12 - (r_6 + r_6);
        }
      }

  }

  (energy * EPS_ETOILE) * 4.0
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
  //  (And we need a division by distance, so it'd be embarassing)
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
  if args.len() != 2
  {
    // It might be possible to factorize that ?
    eprintln!("ERROR: Wrong number of arguments!");
    eprintln!("\tUsage: cargo run --release -- file");
    eprintln!("\tUsage (without cargo): {} file", args[0]);
    panic!("Incorrect call to program.");
  }
  // Attempting to open input file
  let file_path = &args[1];
  let file = match File::open(file_path)
  {
    Ok(file) => file,
    Err(error) => match error.kind()
    {
      std::io::ErrorKind::NotFound =>
      {
        panic!("File not found!");
      }
      _ => panic!("Unknown error upon opening file."),
    },
  };
  let mut reader = io::BufReader::new(file);

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
  // WARNING: Some check should be added in case multiple particules are identical!
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

  // ---------------------------------------------------------------------------



  // -------------------------------- Computation ------------------------------

  let current_energy: f64 = energy_computation(&positions, &mut forces, taille);
  let somme_forces = compute_forces(&forces, taille);
  print!("Number of elements : {} ; ", taille);
  print!("Computer precision : {:e} ; ", COMPUTER_PRECISION);
  println!("Current precision : {:e}", precision);
  if somme_forces.abs() < precision
  {
    println!("Shit is working! Sum of forces = {:e}", somme_forces);
  }
  else
  {
    println!("Shit ain't working >< Sum of forces = {:e}", somme_forces);
  }
  println!("System's energy is {}", current_energy);

  // ---------------------------------------------------------------------------
}
