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
use crate::loss::bce::Reduction;

/// Cross Entropy Loss implementation with support for class weights and ignored indices
pub struct CrossEntropyLoss {
    reduction: Reduction,
    weight: Option<Tensor>,
    ignore_index: Option<i64>,
}

impl CrossEntropyLoss {
    /// Creates a new CrossEntropyLoss with the specified parameters
    pub fn new(reduction: Reduction, weight: Option<Tensor>, ignore_index: Option<i64>) -> Self {
        CrossEntropyLoss {
            reduction,
            weight,
            ignore_index,
        }
    }

    /// Creates a new CrossEntropyLoss with default parameters
    pub fn default() -> Self {
        CrossEntropyLoss {
            reduction: Reduction::Mean,
            weight: None,
            ignore_index: None,
        }
    }

    /// Forward pass of the Cross Entropy Loss calculation
    pub fn forward(&self, prediction: &Tensor, target: &Tensor) -> Result<Tensor, BellandeError> {
        // Validate input shapes
        self.validate_input(prediction, target)?;

        // Get the number of classes (from the prediction shape)
        let num_classes = prediction.shape()[1];

        // Apply log softmax to predictions
        let log_softmax = self.compute_log_softmax(prediction)?;

        // Convert target to one-hot encoding if necessary
        let target_one_hot = self.convert_to_one_hot(target, num_classes)?;

        // Compute the negative log likelihood
        let mut loss = self.compute_nll_loss(&log_softmax, &target_one_hot)?;

        // Apply class weights if provided
        if let Some(weight) = &self.weight {
            loss = self.apply_class_weights(&loss, weight)?;
        }

        // Apply ignore index masking if specified
        if let Some(ignore_idx) = self.ignore_index {
            loss = self.apply_ignore_mask(&loss, target, ignore_idx)?;
        }

        // Apply reduction
        self.apply_reduction(&loss)
    }

    /// Backward pass of the Cross Entropy Loss calculation
    pub fn backward(&self, prediction: &Tensor, target: &Tensor) -> Result<Tensor, BellandeError> {
        // Get softmax probabilities
        let softmax = self.compute_softmax(prediction)?;

        // Convert target to one-hot encoding
        let num_classes = prediction.shape()[1];
        let target_one_hot = self.convert_to_one_hot(target, num_classes)?;

        // Compute gradients: softmax - target
        let mut grad = softmax.sub(&target_one_hot)?;

        // Apply class weights to gradients if provided
        if let Some(weight) = &self.weight {
            grad = self.apply_class_weights(&grad, weight)?;
        }

        // Apply ignore index masking if specified
        if let Some(ignore_idx) = self.ignore_index {
            grad = self.apply_ignore_mask(&grad, target, ignore_idx)?;
        }

        // Apply reduction factor
        match self.reduction {
            Reduction::Mean => {
                let batch_size = prediction.shape()[0] as f32;
                grad.mul_scalar(1.0 / batch_size)
            }
            Reduction::Sum => Ok(grad),
            Reduction::None => Ok(grad),
        }
    }

    // Helper methods

    fn validate_input(&self, prediction: &Tensor, target: &Tensor) -> Result<(), BellandeError> {
        if prediction.dim() != 2 {
            return Err(BellandeError::InvalidInput(
                "Prediction tensor must be 2-dimensional (batch_size, num_classes)".to_string(),
            ));
        }

        if target.dim() != 1 {
            return Err(BellandeError::InvalidInput(
                "Target tensor must be 1-dimensional (batch_size)".to_string(),
            ));
        }

        if prediction.shape()[0] != target.shape()[0] {
            return Err(BellandeError::ShapeMismatch(
                "Batch sizes of prediction and target must match".to_string(),
            ));
        }

        Ok(())
    }

    fn compute_log_softmax(&self, input: &Tensor) -> Result<Tensor, BellandeError> {
        // Compute max for numerical stability
        let max = input.max_dim(1, true)?;
        let shifted = input.sub(&max)?;

        // Compute exp and sum
        let exp = shifted.exp()?;
        let sum = exp.sum_dim(1, true)?;

        // Compute log softmax
        let log_sum = sum.log()?;
        shifted.sub(&log_sum)
    }

    fn compute_softmax(&self, input: &Tensor) -> Result<Tensor, BellandeError> {
        // Compute max for numerical stability
        let max = input.max_dim(1, true)?;
        let shifted = input.sub(&max)?;

        // Compute exp and sum
        let exp = shifted.exp()?;
        let sum = exp.sum_dim(1, true)?;

        // Compute softmax
        exp.div(&sum)
    }

    fn convert_to_one_hot(
        &self,
        target: &Tensor,
        num_classes: usize,
    ) -> Result<Tensor, BellandeError> {
        let batch_size = target.shape()[0];
        let mut one_hot = Tensor::zeros(&[batch_size, num_classes])?;

        for i in 0..batch_size {
            let idx = target.get(i)? as usize;
            if idx >= num_classes {
                return Err(BellandeError::InvalidInput(format!(
                    "Target class {} is out of range (0, {})",
                    idx,
                    num_classes - 1
                )));
            }
            one_hot.set(i, idx, 1.0)?;
        }

        Ok(one_hot)
    }

    fn compute_nll_loss(
        &self,
        log_probs: &Tensor,
        target: &Tensor,
    ) -> Result<Tensor, BellandeError> {
        // Compute negative log likelihood
        let mut nll = Tensor::zeros(&log_probs.shape())?;
        for i in 0..target.shape()[0] {
            for j in 0..target.shape()[1] {
                if target.get(i, j)? > 0.0 {
                    nll.set(i, j, -log_probs.get(i, j)?)?;
                }
            }
        }
        Ok(nll)
    }

    fn apply_class_weights(&self, loss: &Tensor, weight: &Tensor) -> Result<Tensor, BellandeError> {
        // Apply class weights to the loss
        loss.mul(weight)
    }

    fn apply_ignore_mask(
        &self,
        loss: &Tensor,
        target: &Tensor,
        ignore_idx: i64,
    ) -> Result<Tensor, BellandeError> {
        let mut masked_loss = loss.clone();
        for i in 0..target.shape()[0] {
            if target.get(i)? as i64 == ignore_idx {
                for j in 0..masked_loss.shape()[1] {
                    masked_loss.set(i, j, 0.0)?;
                }
            }
        }
        Ok(masked_loss)
    }

    fn apply_reduction(&self, loss: &Tensor) -> Result<Tensor, BellandeError> {
        match self.reduction {
            Reduction::Mean => loss.mean(),
            Reduction::Sum => loss.sum(),
            Reduction::None => Ok(loss.clone()),
        }
    }
}
