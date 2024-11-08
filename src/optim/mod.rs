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

pub mod adam;
pub mod rmsprop;
pub mod scheduler;
pub mod sgd;

/// The Optimizer trait defines the interface for optimization algorithms used in training neural networks.
pub trait Optimizer: Send + Sync {
    fn step(&mut self) -> Result<(), BellandeError>;
    fn zero_grad(&mut self);

    /// Gets the current learning rate
    fn get_learning_rate(&self) -> f32;

    /// Sets a new learning rate
    fn set_learning_rate(&mut self, lr: f32);

    /// Gets the name of the optimizer
    fn name(&self) -> &str {
        "GenericOptimizer"
    }

    /// Gets the current parameter groups
    fn get_param_groups(&self) -> &[ParameterGroup];

    /// Gets the current parameter groups mutably
    fn get_param_groups_mut(&mut self) -> &mut [ParameterGroup];

    /// Adds a parameter group to the optimizer
    fn add_param_group(&mut self, group: ParameterGroup);

    /// Gets the current state of the optimizer
    fn state(&self) -> &OptimizerState;

    /// Gets the current state of the optimizer mutably
    fn state_mut(&mut self) -> &mut OptimizerState;
}

#[derive(Clone)]
pub struct ParameterGroup {
    pub params: Vec<Tensor>,
    pub lr: f32,
    pub weight_decay: f32,
    pub momentum: Option<f32>,
    pub betas: Option<(f32, f32)>,
    pub eps: f32,
}

impl ParameterGroup {
    pub fn new(params: Vec<Tensor>) -> Self {
        Self {
            params,
            lr: 0.001,
            weight_decay: 0.0,
            momentum: None,
            betas: None,
            eps: 1e-8,
        }
    }

    pub fn with_lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = Some(momentum);
        self
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.betas = Some((beta1, beta2));
        self
    }

    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

/// Represents the internal state of an optimizer
#[derive(Default)]
pub struct OptimizerState {
    /// Step count for the optimizer
    pub step: usize,
    /// State dictionary for storing optimizer-specific values
    pub state_dict: HashMap<String, Tensor>,
}

impl OptimizerState {
    pub fn new() -> Self {
        Self {
            step: 0,
            state_dict: HashMap::new(),
        }
    }

    pub fn increment_step(&mut self) {
        self.step += 1;
    }

    pub fn get_state(&self, key: &str) -> Option<&Tensor> {
        self.state_dict.get(key)
    }

    pub fn set_state(&mut self, key: String, value: Tensor) {
        self.state_dict.insert(key, value);
    }
}

/// Learning rate scheduler trait
pub trait LearningRateScheduler: Send + Sync {
    /// Updates the learning rate based on the current epoch and metrics
    fn step(&mut self, epoch: usize, metrics: &HashMap<String, f32>) -> Result<(), BellandeError>;

    /// Gets the last computed learning rate
    fn get_last_lr(&self) -> f32;

    /// Gets the name of the scheduler
    fn name(&self) -> &str {
        "GenericScheduler"
    }
}

pub mod utils {
    use super::*;

    /// Applies weight decay to parameters
    pub fn apply_weight_decay(param: &mut Tensor, weight_decay: f32) -> Result<(), BellandeError> {
        if weight_decay != 0.0 {
            let grad = param.grad()?;
            grad.add_scaled(param, weight_decay)?;
        }
        Ok(())
    }

    /// Clips gradients by norm
    pub fn clip_grad_norm(
        parameters: &[Tensor],
        max_norm: f32,
        norm_type: f32,
    ) -> Result<f32, BellandeError> {
        let total_norm = compute_grad_norm(parameters, norm_type)?;

        if total_norm > max_norm {
            let scale = max_norm / (total_norm + 1e-6);
            for param in parameters {
                if let Some(grad) = param.grad() {
                    grad.mul_scalar(scale)?;
                }
            }
        }

        Ok(total_norm)
    }

    /// Computes the norm of gradients
    fn compute_grad_norm(parameters: &[Tensor], norm_type: f32) -> Result<f32, BellandeError> {
        let mut total_norm = 0.0;

        for param in parameters {
            if let Some(grad) = param.grad() {
                let param_norm = grad.norm(norm_type)?;
                total_norm += param_norm.powf(norm_type);
            }
        }

        Ok(total_norm.powf(1.0 / norm_type))
    }
}
