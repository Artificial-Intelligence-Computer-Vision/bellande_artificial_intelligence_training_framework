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
use crate::layer::Layer;

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    pub fn add(&mut self, layer: Box<dyn Layer>) -> &mut Self {
        self.layers.push(layer);
        self
    }

    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        let mut current = input.clone();
        for layer in &mut self.layers {
            current = layer.forward(&current)?;
        }
        Ok(current)
    }

    pub fn backward(&mut self, grad: &Tensor) -> Result<Tensor, BellandeError> {
        let mut current_grad = grad.clone();
        for layer in self.layers.iter_mut().rev() {
            current_grad = layer.backward(&current_grad)?;
        }
        Ok(current_grad)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}
