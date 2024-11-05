use crate::core::device::Device;
use crate::core::dtype::DataType;
use crate::core::error::BellandeError;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub requires_grad: bool,
    pub grad: Option<Vec<f32>>,
    pub grad_fn: Option<Box<dyn AutogradFunction>>,
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
        let stride = calculate_stride(&shape);
        Tensor {
            data,
            shape,
            stride,
            requires_grad,
            grad: if requires_grad {
                Some(vec![0.0; data.len()])
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
            Device::CPU,
            DataType::Float32,
        )
    }

    pub fn randn(shape: &[usize]) -> Self {
        use rand::distributions::{Distribution, Normal};
        let normal = Normal::new(0.0, 1.0);
        let mut rng = rand::thread_rng();

        let size = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng) as f32).collect();

        Tensor::new(data, shape.to_vec(), false, Device::CPU, DataType::Float32)
    }

    pub fn backward(&mut self) -> Result<(), BellandeError> {
        if !self.requires_grad {
            return Err(BellandeError::NoGradients);
        }

        if self.data.len() != 1 {
            return Err(BellandeError::InvalidBackward);
        }

        if let Some(grad) = &mut self.grad {
            grad[0] = 1.0;
        }

        if let Some(grad_fn) = &self.grad_fn {
            grad_fn.backward(self.grad.as_ref().unwrap())?;
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
