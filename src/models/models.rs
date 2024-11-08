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

use crate::core::{device::Device, dtype::DataType, error::BellandeError, tensor::Tensor};
use crate::layer::Layer;
use crate::layer::{
    activation::ReLU, batch_norm::BatchNorm2d, conv::Conv2d, dropout::Dropout, linear::Linear,
    pooling::MaxPool2d,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trait defining the base functionality for all models
pub trait Model: Send + Sync {
    fn forward(&self, input: &Tensor) -> Result<Tensor, BellandeError>;
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, BellandeError>;
    fn parameters(&self) -> Vec<Tensor>;
    fn train(&mut self);
    fn eval(&mut self);
    fn save(&self, path: &str) -> Result<(), BellandeError>;
    fn load(&mut self, path: &str) -> Result<(), BellandeError>;
    fn state_dict(&self) -> HashMap<String, Tensor>;
    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>)
        -> Result<(), BellandeError>;
}

/// State configuration for model serialization
#[derive(Serialize, Deserialize)]
pub struct ModelState {
    pub model_type: String,
    pub state_dict: HashMap<String, Vec<f32>>,
    pub shapes: HashMap<String, Vec<usize>>,
    pub config: ModelConfig,
}

/// Configuration for model architecture
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelConfig {
    pub input_shape: Vec<usize>,
    pub num_classes: usize,
    pub dropout_rate: f32,
    pub hidden_layers: Vec<usize>,
}

/// Sequential model implementation
#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    training: bool,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential {
            layers: Vec::new(),
            training: true,
        }
    }

    pub fn add(&mut self, layer: Box<dyn Layer>) -> &mut Self {
        self.layers.push(layer);
        self
    }

    pub fn get_layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }

    pub fn get_layers_mut(&mut self) -> &mut [Box<dyn Layer>] {
        &mut self.layers
    }
}

impl Model for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor, BellandeError> {
        let mut current = input.clone();
        for layer in &self.layers {
            current = layer.forward(&current)?;
        }
        Ok(current)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, BellandeError> {
        let mut current_grad = grad.clone();
        for layer in self.layers.iter_mut().rev() {
            current_grad = layer.backward(&current_grad)?;
        }
        Ok(current_grad)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn save(&self, path: &str) -> Result<(), BellandeError> {
        let state = ModelState {
            model_type: "Sequential".to_string(),
            state_dict: self
                .state_dict()
                .into_iter()
                .map(|(k, v)| (k, v.data))
                .collect(),
            shapes: self
                .state_dict()
                .into_iter()
                .map(|(k, v)| (k, v.shape))
                .collect(),
            config: ModelConfig {
                input_shape: vec![],
                num_classes: 0,
                dropout_rate: 0.0,
                hidden_layers: vec![],
            },
        };

        let file = std::fs::File::create(path)?;
        serde_json::to_writer(file, &state)
            .map_err(|e| BellandeError::SerializationError(format!("Failed to save model: {}", e)))
    }

    fn load(&mut self, path: &str) -> Result<(), BellandeError> {
        let file = std::fs::File::open(path)?;
        let state: ModelState = serde_json::from_reader(file).map_err(|e| {
            BellandeError::SerializationError(format!("Failed to load model: {}", e))
        })?;

        let mut state_dict = HashMap::new();
        for (key, data) in state.state_dict {
            let shape = state.shapes.get(&key).ok_or_else(|| {
                BellandeError::SerializationError(format!("Missing shape for key: {}", key))
            })?;

            state_dict.insert(
                key,
                Tensor::new(data, shape.clone(), true, Device::CPU, DataType::Float32),
            );
        }

        self.load_state_dict(state_dict)
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut state_dict = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.named_parameters() {
                state_dict.insert(format!("layer_{}.{}", i, name), param);
            }
        }
        state_dict
    }

    fn load_state_dict(
        &mut self,
        state_dict: HashMap<String, Tensor>,
    ) -> Result<(), BellandeError> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            for (name, _) in layer.named_parameters() {
                let key = format!("layer_{}.{}", i, name);
                if let Some(param) = state_dict.get(&key) {
                    layer.set_parameter(&name, param.clone())?;
                } else {
                    return Err(BellandeError::SerializationError(format!(
                        "Missing parameter: {}",
                        key
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Create a simple feed-forward neural network
pub fn create_mlp(config: &ModelConfig) -> Result<Sequential, BellandeError> {
    let mut model = Sequential::new();

    let mut current_size = config.input_shape.iter().product();

    // Input layer
    model.add(Box::new(Linear::new(
        current_size,
        config.hidden_layers[0],
        true,
    )));
    model.add(Box::new(ReLU::new()));

    // Hidden layers
    for i in 1..config.hidden_layers.len() {
        if config.dropout_rate > 0.0 {
            model.add(Box::new(Dropout::new(config.dropout_rate)));
        }
        model.add(Box::new(Linear::new(
            config.hidden_layers[i - 1],
            config.hidden_layers[i],
            true,
        )));
        model.add(Box::new(ReLU::new()));
    }

    // Output layer
    model.add(Box::new(Linear::new(
        config.hidden_layers[config.hidden_layers.len() - 1],
        config.num_classes,
        true,
    )));

    Ok(model)
}

/// Create a simple CNN
pub fn create_cnn(config: &ModelConfig) -> Result<Sequential, BellandeError> {
    let mut model = Sequential::new();

    // First conv block
    model.add(Box::new(Conv2d::new(3, 64, 3, 1, 1, true)));
    model.add(Box::new(BatchNorm2d::new(64, 1e-5, 0.1, true)));
    model.add(Box::new(ReLU::new()));
    model.add(Box::new(MaxPool2d::new(2, 2)));

    // Second conv block
    model.add(Box::new(Conv2d::new(64, 128, 3, 1, 1, true)));
    model.add(Box::new(BatchNorm2d::new(128, 1e-5, 0.1, true)));
    model.add(Box::new(ReLU::new()));
    model.add(Box::new(MaxPool2d::new(2, 2)));

    // Classification head
    model.add(Box::new(Linear::new(128 * 7 * 7, 512, true)));
    model.add(Box::new(ReLU::new()));
    if config.dropout_rate > 0.0 {
        model.add(Box::new(Dropout::new(config.dropout_rate)));
    }
    model.add(Box::new(Linear::new(512, config.num_classes, true)));

    Ok(model)
}
