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

pub struct Conv2d {
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    weight: Tensor,
    bias: Option<Tensor>,
    input_cache: Option<Tensor>,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> Self {
        let weight = Tensor::randn(&[out_channels, in_channels, kernel_size.0, kernel_size.1]);

        let bias = if bias {
            Some(Tensor::zeros(&[out_channels]))
        } else {
            None
        };

        Conv2d {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight,
            bias,
            input_cache: None,
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

        if channels != self.in_channels {
            return Err(BellandeError::DimensionMismatch);
        }

        let output_height = (height + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let output_width = (width + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;

        let mut output = vec![0.0; batch_size * self.out_channels * output_height * output_width];

        // Implement convolution operation
        for b in 0..batch_size {
            for out_c in 0..self.out_channels {
                for out_h in 0..output_height {
                    for out_w in 0..output_width {
                        let mut sum = 0.0;

                        for in_c in 0..self.in_channels {
                            for k_h in 0..self.kernel_size.0 {
                                for k_w in 0..self.kernel_size.1 {
                                    let in_h = out_h * self.stride.0 + k_h - self.padding.0;
                                    let in_w = out_w * self.stride.1 + k_w - self.padding.1;

                                    if in_h < height && in_w < width {
                                        let input_idx =
                                            ((b * channels + in_c) * height + in_h) * width + in_w;
                                        let weight_idx = ((out_c * self.in_channels + in_c)
                                            * self.kernel_size.0
                                            + k_h)
                                            * self.kernel_size.1
                                            + k_w;
                                        sum += input.data[input_idx] * self.weight.data[weight_idx];
                                    }
                                }
                            }
                        }

                        if let Some(ref bias) = self.bias {
                            sum += bias.data[out_c];
                        }

                        let output_idx = ((b * self.out_channels + out_c) * output_height + out_h)
                            * output_width
                            + out_w;
                        output[output_idx] = sum;
                    }
                }
            }
        }

        self.input_cache = Some(input.clone());

        Ok(Tensor::new(
            output,
            vec![batch_size, self.out_channels, output_height, output_width],
            true,
            input.device.clone(),
            input.dtype,
        ))
    }

    pub fn backward(
        &self,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor, Option<Tensor>), BellandeError> {
        if let Some(ref input) = self.input_cache {
            let (batch_size, _, output_height, output_width) = (
                grad_output.shape[0],
                grad_output.shape[1],
                grad_output.shape[2],
                grad_output.shape[3],
            );

            // Gradient with respect to input
            let mut grad_input = vec![0.0; input.data.len()];
            // Gradient with respect to weight
            let mut grad_weight = vec![0.0; self.weight.data.len()];
            // Gradient with respect to bias
            let mut grad_bias = if self.bias.is_some() {
                Some(vec![0.0; self.out_channels])
            } else {
                None
            };

            // Implement backward pass
            // ... (Complex backward pass implementation)

            Ok((
                Tensor::new(
                    grad_input,
                    input.shape.clone(),
                    true,
                    input.device.clone(),
                    input.dtype,
                ),
                Tensor::new(
                    grad_weight,
                    self.weight.shape.clone(),
                    true,
                    self.weight.device.clone(),
                    self.weight.dtype,
                ),
                grad_bias.map(|bias| {
                    Tensor::new(
                        bias,
                        vec![self.out_channels],
                        true,
                        self.weight.device.clone(),
                        self.weight.dtype,
                    )
                }),
            ))
        } else {
            Err(BellandeError::RuntimeError(
                "Forward pass not called".into(),
            ))
        }
    }
}
