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

use crate::core::{
    autograd::AutogradFunction, device::Device, dtype::DataType, error::BellandeError,
};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
    pub grad: Option<Vec<f32>>,
    pub grad_fn: Option<Arc<dyn AutogradFunction>>,
    pub device: Device,
    pub dtype: DataType,
}

impl Tensor {
    pub fn new(
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
        device: Device,
        dtype: DataType,
    ) -> Self {
        let size = shape.iter().product();
        assert_eq!(data.len(), size, "Data size does not match shape");

        Tensor {
            data,
            shape,
            requires_grad,
            grad: if requires_grad {
                Some(vec![0.0; size])
            } else {
                None
            },
            grad_fn: None,
            device,
            dtype,
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Tensor::new(
            vec![0.0; size],
            shape.to_vec(),
            false,
            Device::default(),
            DataType::default(),
        )
    }

    pub fn ones(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Tensor::new(
            vec![1.0; size],
            shape.to_vec(),
            false,
            Device::default(),
            DataType::default(),
        )
    }

    pub fn randn(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Tensor::new(
            crate::core::random::normal(0.0, 1.0, size),
            shape.to_vec(),
            false,
            Device::default(),
            DataType::default(),
        )
    }

    pub fn backward(&mut self) -> Result<(), BellandeError> {
        if !self.requires_grad {
            return Err(BellandeError::NoGradients);
        }

        if self.grad.is_none() {
            self.grad = Some(vec![1.0; self.data.len()]);
        }

        if let Some(ref grad_fn) = self.grad_fn {
            if let Some(ref grad) = self.grad {
                grad_fn.backward(&Tensor::new(
                    grad.clone(),
                    self.shape.clone(),
                    false,
                    self.device.clone(),
                    self.dtype,
                ))?;
            }
        }

        Ok(())
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, BellandeError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(BellandeError::InvalidShape);
        }

        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);

        if k != k2 {
            return Err(BellandeError::DimensionMismatch);
        }

        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..k {
                    sum += self.data[i * k + k] * other.data[k * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Ok(Tensor::new(
            result,
            vec![m, n],
            self.requires_grad || other.requires_grad,
            self.device.clone(),
            self.dtype,
        ))
    }
}
