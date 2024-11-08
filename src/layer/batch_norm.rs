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
use std::sync::Arc;

pub struct BatchNorm2d {
    num_features: usize,
    eps: f32,
    momentum: f32,
    running_mean: Arc<Tensor>,
    running_var: Arc<Tensor>,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    training: bool,
}

impl BatchNorm2d {
    pub fn new(num_features: usize, eps: f32, momentum: f32, affine: bool) -> Self {
        BatchNorm2d {
            num_features,
            eps,
            momentum,
            running_mean: Arc::new(Tensor::zeros(&[num_features])),
            running_var: Arc::new(Tensor::ones(&[num_features])),
            weight: if affine {
                Some(Tensor::ones(&[num_features]))
            } else {
                None
            },
            bias: if affine {
                Some(Tensor::zeros(&[num_features]))
            } else {
                None
            },
            training: true,
        }
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, BellandeError> {
        if input.shape.len() != 4 {
            return Err(BellandeError::InvalidShape);
        }

        let (batch_size, channels, height, width) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );

        if channels != self.num_features {
            return Err(BellandeError::DimensionMismatch);
        }

        let mut output = input.data.clone();

        if self.training {
            // Calculate mean and variance
            let mut mean = vec![0.0; channels];
            let mut var = vec![0.0; channels];
            let size = batch_size * height * width;

            for c in 0..channels {
                let mut sum = 0.0;
                let mut sq_sum = 0.0;

                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            let val = input.data[idx];
                            sum += val;
                            sq_sum += val * val;
                        }
                    }
                }

                mean[c] = sum / size as f32;
                var[c] = sq_sum / size as f32 - mean[c] * mean[c];
            }

            // Update running statistics
            for c in 0..channels {
                self.running_mean.data[c] =
                    self.momentum * self.running_mean.data[c] + (1.0 - self.momentum) * mean[c];
                self.running_var.data[c] =
                    self.momentum * self.running_var.data[c] + (1.0 - self.momentum) * var[c];
            }

            // Normalize
            for c in 0..channels {
                let std = (var[c] + self.eps).sqrt();
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            output[idx] = (output[idx] - mean[c]) / std;

                            if let Some(ref weight) = self.weight {
                                output[idx] *= weight.data[c];
                            }
                            if let Some(ref bias) = self.bias {
                                output[idx] += bias.data[c];
                            }
                        }
                    }
                }
            }
        } else {
            // Use running statistics
            for c in 0..channels {
                let std = (self.running_var.data[c] + self.eps).sqrt();
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            output[idx] = (output[idx] - self.running_mean.data[c]) / std;

                            if let Some(ref weight) = self.weight {
                                output[idx] *= weight.data[c];
                            }
                            if let Some(ref bias) = self.bias {
                                output[idx] += bias.data[c];
                            }
                        }
                    }
                }
            }
        }

        Ok(Tensor::new(
            output,
            input.shape.clone(),
            input.requires_grad,
            input.device.clone(),
            input.dtype,
        ))
    }
}
