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

pub struct MaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    indices: Option<Vec<usize>>,
}

impl MaxPool2d {
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        MaxPool2d {
            kernel_size,
            stride,
            indices: None,
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        if input.shape.len() != 4 {
            return Err(BellandeError::InvalidShape);
        }

        let (batch_size, channels, height, width) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );

        let output_height = (height - self.kernel_size.0) / self.stride.0 + 1;
        let output_width = (width - self.kernel_size.1) / self.stride.1 + 1;

        let mut output = vec![0.0; batch_size * channels * output_height * output_width];
        let mut indices = vec![0; batch_size * channels * output_height * output_width];

        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..output_height {
                    for w in 0..output_width {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;

                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let in_h = h * self.stride.0 + kh;
                                let in_w = w * self.stride.1 + kw;
                                let idx = ((b * channels + c) * height + in_h) * width + in_w;
                                let val = input.data[idx];

                                if val > max_val {
                                    max_val = val;
                                    max_idx = idx;
                                }
                            }
                        }

                        let out_idx = ((b * channels + c) * output_height + h) * output_width + w;
                        output[out_idx] = max_val;
                        indices[out_idx] = max_idx;
                    }
                }
            }
        }

        self.indices = Some(indices);

        Ok(Tensor::new(
            output,
            vec![batch_size, channels, output_height, output_width],
            input.requires_grad,
            input.device.clone(),
            input.dtype,
        ))
    }

    pub fn backward(&self, grad_output: &Tensor) -> Result<Tensor, BellandeError> {
        if let Some(ref indices) = self.indices {
            let mut grad_input = vec![0.0; indices.len()];

            for (out_idx, &in_idx) in indices.iter().enumerate() {
                grad_input[in_idx] += grad_output.data[out_idx];
            }

            Ok(Tensor::new(
                grad_input,
                grad_output.shape.clone(),
                true,
                grad_output.device.clone(),
                grad_output.dtype,
            ))
        } else {
            Err(BellandeError::RuntimeError(
                "Forward pass not called".into(),
            ))
        }
    }
}
