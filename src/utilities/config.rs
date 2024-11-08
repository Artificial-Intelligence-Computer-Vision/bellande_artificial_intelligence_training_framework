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

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct Configuration {
    // Training configuration
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub optimizer: OptimizerConfig,

    // Model configuration
    pub model: ModelConfig,

    // Data configuration
    pub data: DataConfig,

    // System configuration
    pub system: SystemConfig,

    // Custom parameters
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub name: String,
    pub momentum: Option<f32>,
    pub beta1: Option<f32>,
    pub beta2: Option<f32>,
    pub weight_decay: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: String,
    pub input_shape: Vec<usize>,
    pub num_classes: usize,
    pub hidden_layers: Vec<usize>,
    pub dropout_rate: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DataConfig {
    pub train_path: String,
    pub val_path: Option<String>,
    pub test_path: Option<String>,
    pub augmentation: bool,
    pub normalize: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemConfig {
    pub num_workers: usize,
    pub device: String,
    pub precision: String,
    pub seed: Option<u64>,
}

impl Configuration {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let content = fs::read_to_string(path)?;
        let config: Configuration = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let content = serde_yaml::to_string(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        // Validate batch size
        if self.batch_size == 0 {
            return Err("Batch size must be greater than 0".to_string());
        }

        // Validate learning rate
        if self.learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }

        // Validate model configuration
        if self.model.input_shape.is_empty() {
            return Err("Input shape cannot be empty".to_string());
        }

        // Validate data paths
        if !Path::new(&self.data.train_path).exists() {
            return Err("Training data path does not exist".to_string());
        }

        Ok(())
    }
}
