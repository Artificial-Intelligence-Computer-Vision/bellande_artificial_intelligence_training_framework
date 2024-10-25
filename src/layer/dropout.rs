// Copyright (C) 2024 Bellande Artificial Intelligence Computer Vision Research Innovation Center, Ronaldson Bellande

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

pub struct Dropout {
    p: f32,
    mask: Option<Vec<bool>>,
    training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(
            p >= 0.0 && p < 1.0,
            "Dropout probability must be between 0 and 1"
        );
        Dropout {
            p,
            mask: None,
            training: true,
        }
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl Layer for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if self.training {
            let mut rng = rand::thread_rng();
            let mask: Vec<bool> = (0..input.data.len())
                .map(|_| rng.gen::<f32>() > self.p)
                .collect();

            let scale = 1.0 / (1.0 - self.p);
            let output: Vec<f32> = input
                .data
                .iter()
                .zip(mask.iter())
                .map(|(&x, &m)| if m { x * scale } else { 0.0 })
                .collect();

            self.mask = Some(mask);
            Tensor::new(output, input.shape.clone(), input.requires_grad)
        } else {
            input.clone()
        }
    }

    fn backward(&mut self, grad: &Tensor) -> Tensor {
        if let Some(mask) = &self.mask {
            let scale = 1.0 / (1.0 - self.p);
            let dx: Vec<f32> = grad
                .data
                .iter()
                .zip(mask.iter())
                .map(|(&g, &m)| if m { g * scale } else { 0.0 })
                .collect();

            Tensor::new(dx, grad.shape.clone(), true)
        } else {
            grad.clone()
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}
