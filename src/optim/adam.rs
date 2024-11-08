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

use crate::core::{error::BellandeError, tensor::Tensor};
use std::collections::HashMap;

pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    m: HashMap<usize, Vec<f32>>,
    v: HashMap<usize, Vec<f32>>,
    step: usize,
}

impl Adam {
    pub fn new(
        params: Vec<Tensor>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        let mut m = HashMap::new();
        let mut v = HashMap::new();

        for (idx, param) in params.iter().enumerate() {
            m.insert(idx, vec![0.0; param.data.len()]);
            v.insert(idx, vec![0.0; param.data.len()]);
        }

        Adam {
            params,
            lr,
            betas,
            eps,
            weight_decay,
            m,
            v,
            step: 0,
        }
    }

    pub fn step(&mut self) -> Result<(), BellandeError> {
        self.step += 1;
        let bias_correction1 = 1.0 - self.betas.0.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.betas.1.powi(self.step as i32);

        for (idx, param) in self.params.iter_mut().enumerate() {
            if let Some(grad) = &param.grad {
                let m = self.m.get_mut(&idx).unwrap();
                let v = self.v.get_mut(&idx).unwrap();

                for ((p, g), (m, v)) in param
                    .data
                    .iter_mut()
                    .zip(grad.iter())
                    .zip(m.iter_mut().zip(v.iter_mut()))
                {
                    if self.weight_decay != 0.0 {
                        *g += self.weight_decay * *p;
                    }

                    // Update biased first moment estimate
                    *m = self.betas.0 * *m + (1.0 - self.betas.0) * g;

                    // Update biased second raw moment estimate
                    *v = self.betas.1 * *v + (1.0 - self.betas.1) * g * g;

                    // Compute bias-corrected moment estimates
                    let m_hat = *m / bias_correction1;
                    let v_hat = *v / bias_correction2;

                    // Update parameters
                    *p -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
                }
            }
        }

        Ok(())
    }

    pub fn zero_grad(&mut self) {
        for param in &mut self.params {
            if let Some(grad) = &mut param.grad {
                grad.iter_mut().for_each(|g| *g = 0.0);
            }
        }
    }

    pub fn get_lr(&self) -> f32 {
        self.lr
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}
