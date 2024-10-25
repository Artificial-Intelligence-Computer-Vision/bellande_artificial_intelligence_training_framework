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

use std::collections::HashMap;

pub struct BatchNorm2d {
    num_features: usize,
    eps: f32,
    momentum: f32,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
    weight: Tensor, // gamma
    bias: Tensor,   // beta
    cache: Option<BatchNormCache>,
    training: bool,
}

struct BatchNormCache {
    input: Tensor,
    normalized: Tensor,
    std: Vec<f32>,
    centered: Vec<f32>,
}

impl BatchNorm2d {
    pub fn new(num_features: usize, eps: f32, momentum: f32) -> Self {
        BatchNorm2d {
            num_features,
            eps,
            momentum,
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
            weight: Tensor::new(vec![1.0; num_features], vec![num_features], true),
            bias: Tensor::new(vec![0.0; num_features], vec![num_features], true),
            cache: None,
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

impl Layer for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let batch_size = input.shape[0];
        let channels = input.shape[1];
        let height = input.shape[2];
        let width = input.shape[3];

        let mut output = vec![0.0; input.data.len()];
        let spatial_size = height * width;

        if self.training {
            let mut mean = vec![0.0; channels];
            let mut var = vec![0.0; channels];
            let mut normalized = vec![0.0; input.data.len()];
            let mut centered = vec![0.0; input.data.len()];

            // Calculate mean and variance for each channel
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

                mean[c] = sum / (batch_size * spatial_size) as f32;
                var[c] = (sq_sum / (batch_size * spatial_size) as f32) - mean[c] * mean[c];

                // Update running statistics
                self.running_mean[c] =
                    self.momentum * self.running_mean[c] + (1.0 - self.momentum) * mean[c];
                self.running_var[c] =
                    self.momentum * self.running_var[c] + (1.0 - self.momentum) * var[c];
            }

            // Normalize and apply affine transform
            let std: Vec<f32> = var.iter().map(|&v| (v + self.eps).sqrt()).collect();

            for c in 0..channels {
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            centered[idx] = input.data[idx] - mean[c];
                            normalized[idx] = centered[idx] / std[c];
                            output[idx] = self.weight.data[c] * normalized[idx] + self.bias.data[c];
                        }
                    }
                }
            }

            self.cache = Some(BatchNormCache {
                input: input.clone(),
                normalized: Tensor::new(normalized, input.shape.clone(), true),
                std,
                centered,
            });
        } else {
            // Inference mode
            for c in 0..channels {
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            let normalized = (input.data[idx] - self.running_mean[c])
                                / (self.running_var[c] + self.eps).sqrt();
                            output[idx] = self.weight.data[c] * normalized + self.bias.data[c];
                        }
                    }
                }
            }
        }

        Tensor::new(output, input.shape.clone(), input.requires_grad)
    }

    fn backward(&mut self, grad: &Tensor) -> Tensor {
        if let Some(cache) = &self.cache {
            let batch_size = grad.shape[0];
            let channels = grad.shape[1];
            let height = grad.shape[2];
            let width = grad.shape[3];
            let spatial_size = height * width;

            let mut dx = vec![0.0; grad.data.len()];
            let mut dgamma = vec![0.0; channels];
            let mut dbeta = vec![0.0; channels];

            // Calculate gradients for gamma and beta
            for c in 0..channels {
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;
                            dgamma[c] += grad.data[idx] * cache.normalized.data[idx];
                            dbeta[c] += grad.data[idx];
                        }
                    }
                }
            }

            self.weight.grad = Some(dgamma);
            self.bias.grad = Some(dbeta);

            // Calculate gradient with respect to input
            let m = (batch_size * spatial_size) as f32;

            for c in 0..channels {
                let std_inv = 1.0 / cache.std[c];

                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * channels + c) * height + h) * width + w;

                            dx[idx] = grad.data[idx] * self.weight.data[c] * std_inv
                                - cache.centered[idx] * std_inv * std_inv / m;
                        }
                    }
                }
            }

            Tensor::new(dx, grad.shape.clone(), true)
        } else {
            panic!("Backward called before forward!");
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
