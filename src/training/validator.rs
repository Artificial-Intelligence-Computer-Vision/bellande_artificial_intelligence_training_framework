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

use crate::core::{device::Device, error::BellandeError};

use crate::data::dataloader::DataLoader;
use crate::metrics::metrics::Metric;
use crate::models::models::Model;
use std::collections::HashMap;

pub struct Validator {
    model: Box<dyn Model>,
    metrics: Vec<Box<dyn Metric>>,
    device: Device,
}

impl Validator {
    pub fn new(model: Box<dyn Model>, metrics: Vec<Box<dyn Metric>>, device: Device) -> Self {
        Validator {
            model,
            metrics,
            device,
        }
    }

    pub fn validate(
        &mut self,
        val_loader: DataLoader,
    ) -> Result<HashMap<String, f32>, BellandeError> {
        self.model.eval();
        let mut metrics = RunningMetrics::new();

        for (data, target) in val_loader {
            let output = self.model.forward(&data.to(self.device))?;

            for metric in &mut self.metrics {
                let value = metric.compute(&output, &target.to(self.device))?;
                metrics.update(&metric.name(), value);
            }
        }

        Ok(metrics.get_average())
    }
}

struct RunningMetrics {
    values: HashMap<String, Vec<f32>>,
}

impl RunningMetrics {
    fn new() -> Self {
        RunningMetrics {
            values: HashMap::new(),
        }
    }

    fn update(&mut self, name: &str, value: f32) {
        self.values
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(value);
    }

    fn get_average(&self) -> HashMap<String, f32> {
        self.values
            .iter()
            .map(|(k, v)| {
                let avg = v.iter().sum::<f32>() / v.len() as f32;
                (k.clone(), avg)
            })
            .collect()
    }

    fn get_current(&self) -> HashMap<String, f32> {
        self.values
            .iter()
            .map(|(k, v)| (k.clone(), *v.last().unwrap()))
            .collect()
    }
}

pub enum CallbackEvent {
    TrainBegin,
    TrainEnd,
    EpochBegin,
    EpochEnd,
    BatchBegin,
    BatchEnd,
}
