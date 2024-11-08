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

pub struct RMSprop {
    params: Vec<Tensor>,
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
    v: HashMap<usize, Vec<f32>>,   // Square average
    g: HashMap<usize, Vec<f32>>,   // Gradient average (if centered)
    buf: HashMap<usize, Vec<f32>>, // Momentum buffer
}

impl RMSprop {
    pub fn new(
        params: Vec<Tensor>,
        lr: f32,
        alpha: f32,
        eps: f32,
        weight_decay: f32,
        momentum: f32,
        centered: bool,
    ) -> Self {
        let mut v = HashMap::new();
        let mut g = HashMap::new();
        let mut buf = HashMap::new();

        for (idx, param) in params.iter().enumerate() {
            v.insert(idx, vec![0.0; param.data.len()]);
            if centered {
                g.insert(idx, vec![0.0; param.data.len()]);
            }
            if momentum > 0.0 {
                buf.insert(idx, vec![0.0; param.data.len()]);
            }
        }

        RMSprop {
            params,
            lr,
            alpha,
            eps,
            weight_decay,
            momentum,
            centered,
            v,
            g,
            buf,
        }
    }

    pub fn step(&mut self) -> Result<(), BellandeError> {
        for (idx, param) in self.params.iter_mut().enumerate() {
            if let Some(grad) = &param.grad {
                let v = self.v.get_mut(&idx).unwrap();
                let g = if self.centered {
                    Some(self.g.get_mut(&idx).unwrap())
                } else {
                    None
                };
                let buf = if self.momentum > 0.0 {
                    Some(self.buf.get_mut(&idx).unwrap())
                } else {
                    None
                };

                for ((p, g_val), v_val) in param.data.iter_mut().zip(grad.iter()).zip(v.iter_mut())
                {
                    let mut grad = *g_val;

                    if self.weight_decay != 0.0 {
                        grad += self.weight_decay * *p;
                    }

                    *v_val = self.alpha * *v_val + (1.0 - self.alpha) * grad * grad;

                    if let Some(g_avg) = g {
                        *g_avg = self.alpha * *g_avg + (1.0 - self.alpha) * grad;
                        let denom = v_val.sqrt() - g_avg.powi(2) + self.eps;
                        grad *= 1.0 / denom;
                    } else {
                        grad *= 1.0 / (v_val.sqrt() + self.eps);
                    }

                    if let Some(buf_val) = buf {
                        *buf_val = self.momentum * *buf_val + grad;
                        *p -= self.lr * *buf_val;
                    } else {
                        *p -= self.lr * grad;
                    }
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
