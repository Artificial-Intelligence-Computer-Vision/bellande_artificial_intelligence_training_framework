use rand::prelude::*;
use std::cell::RefCell;

thread_local! {
    static GENERATOR: RefCell<StdRng> = RefCell::new(StdRng::from_entropy());
}

pub fn set_seed(seed: u64) {
    GENERATOR.with(|g| {
        *g.borrow_mut() = StdRng::seed_from_u64(seed);
    });
}

pub fn normal(mean: f32, std: f32, shape: &[usize]) -> Tensor {
    let normal = rand_distr::Normal::new(mean as f64, std as f64).unwrap();
    let size = shape.iter().product();

    let data: Vec<f32> = GENERATOR.with(|g| {
        (0..size)
            .map(|_| normal.sample(&mut *g.borrow_mut()) as f32)
            .collect()
    });

    Tensor::new(data, shape.to_vec(), false, Device::CPU, DataType::Float32)
}
