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

pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    nesterov: bool,
    velocity: HashMap<usize, Vec<f32>>,
}

impl SGD {
    pub fn new(
        params: Vec<Tensor>,
        lr: f32,
        momentum: f32,
        weight_decay: f32,
        nesterov: bool,
    ) -> Self {
        let mut velocity = HashMap::new();
        if momentum > 0.0 {
            for (idx, param) in params.iter().enumerate() {
                velocity.insert(idx, vec![0.0; param.data.len()]);
            }
        }

        SGD {
            params,
            lr,
            momentum,
            weight_decay,
            nesterov,
            velocity,
        }
    }

    pub fn step(&mut self) -> Result<(), BellandeError> {
        for (idx, param) in self.params.iter_mut().enumerate() {
            if let Some(grad) = &param.grad {
                let v = if self.momentum > 0.0 {
                    Some(self.velocity.get_mut(&idx).unwrap())
                } else {
                    None
                };

                for ((p, g), v_opt) in param.data.iter_mut().zip(grad.iter()).zip(v.into_iter()) {
                    let mut d_p = *g;

                    if self.weight_decay != 0.0 {
                        d_p += self.weight_decay * *p;
                    }

                    if let Some(v) = v_opt {
                        *v = self.momentum * *v + d_p;

                        if self.nesterov {
                            d_p += self.momentum * *v;
                        } else {
                            d_p = *v;
                        }
                    }

                    *p -= self.lr * d_p;
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
}
