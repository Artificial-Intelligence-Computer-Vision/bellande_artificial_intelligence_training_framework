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
use crate::models::models::Model;
use crate::training::{callbacks::Callback, history::TrainingHistory, validator::CallbackEvent};

// Import all loss functions
use crate::loss::{
    bce::BCELoss, cross_entropy::CrossEntropyLoss, custom::CustomLossFunction, mse::MSELoss, Loss,
};

// Import all optimizers and scheduler
use crate::optim::{adam::Adam, rmsprop::RMSprop, scheduler::LRScheduler, sgd::SGD, Optimizer};

use std::collections::HashMap;

/// Helper struct for tracking metrics during training
#[derive(Default)]
pub struct RunningMetrics {
    metrics: HashMap<String, (f32, usize)>, // (sum, count)
}

impl RunningMetrics {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    pub fn update(&mut self, name: &str, value: f32) {
        let entry = self.metrics.entry(name.to_string()).or_insert((0.0, 0));
        entry.0 += value;
        entry.1 += 1;
    }

    pub fn get_average(&self) -> HashMap<String, f32> {
        self.metrics
            .iter()
            .map(|(k, (sum, count))| (k.clone(), sum / *count as f32))
            .collect()
    }

    pub fn get_current(&self) -> HashMap<String, f32> {
        self.get_average()
    }
}

pub struct Trainer {
    model: Box<dyn Model>,
    optimizer: Box<dyn Optimizer>,
    loss_fn: Box<dyn Loss>,
    device: Device,
    callbacks: Vec<Box<dyn Callback>>,
    history: TrainingHistory,
    scheduler: Option<Box<dyn LRScheduler>>,
}

impl Trainer {
    pub fn new(
        model: Box<dyn Model>,
        optimizer: Box<dyn Optimizer>,
        loss_fn: Box<dyn Loss>,
        device: Device,
    ) -> Self {
        Trainer {
            model,
            optimizer,
            loss_fn,
            device,
            callbacks: Vec::new(),
            history: TrainingHistory::new(),
            scheduler: None,
        }
    }

    /// Create a new trainer with MSELoss and Adam optimizer
    pub fn new_with_adam(
        model: Box<dyn Model>,
        learning_rate: f32,
        device: Device,
    ) -> Result<Self, BellandeError> {
        let loss_fn = Box::new(MSELoss::new());
        let optimizer = Box::new(Adam::new(model.parameters(), learning_rate)?);

        Ok(Self::new(model, optimizer, loss_fn, device))
    }

    /// Create a new trainer with CrossEntropyLoss and SGD optimizer
    pub fn new_with_sgd(
        model: Box<dyn Model>,
        learning_rate: f32,
        momentum: f32,
        device: Device,
    ) -> Result<Self, BellandeError> {
        let loss_fn = Box::new(CrossEntropyLoss::new());
        let optimizer = Box::new(SGD::new(model.parameters(), learning_rate, momentum)?);

        Ok(Self::new(model, optimizer, loss_fn, device))
    }

    /// Create a new trainer with BCELoss and RMSprop optimizer
    pub fn new_with_rmsprop(
        model: Box<dyn Model>,
        learning_rate: f32,
        alpha: f32,
        device: Device,
    ) -> Result<Self, BellandeError> {
        let loss_fn = Box::new(BCELoss::new());
        let optimizer = Box::new(RMSprop::new(model.parameters(), learning_rate, alpha)?);

        Ok(Self::new(model, optimizer, loss_fn, device))
    }

    /// Add a learning rate scheduler
    pub fn add_scheduler(&mut self, scheduler: Box<dyn LRScheduler>) {
        self.scheduler = Some(scheduler);
    }

    pub fn add_callback(&mut self, callback: Box<dyn Callback>) {
        self.callbacks.push(callback);
    }

    pub fn fit(
        &mut self,
        train_loader: DataLoader,
        val_loader: Option<DataLoader>,
        epochs: usize,
    ) -> Result<TrainingHistory, BellandeError> {
        let mut logs = HashMap::new();
        self.call_callbacks(CallbackEvent::TrainBegin, &logs)?;

        for epoch in 0..epochs {
            logs.clear();
            logs.insert("epoch".to_string(), epoch as f32);
            self.call_callbacks(CallbackEvent::EpochBegin, &logs)?;

            // Training phase
            self.model.train();
            let train_metrics = self.train_epoch(train_loader.clone(), epoch)?;
            logs.extend(train_metrics);

            // Validation phase
            if let Some(val_loader) = &val_loader {
                self.model.eval();
                let val_metrics = self.validate(val_loader.clone())?;
                logs.extend(
                    val_metrics
                        .into_iter()
                        .map(|(k, v)| (format!("val_{}", k), v)),
                );
            }

            // Update learning rate if scheduler is present
            if let Some(scheduler) = &mut self.scheduler {
                scheduler.step(epoch, &logs)?;
            }

            self.history.update(epoch, logs.clone());
            self.call_callbacks(CallbackEvent::EpochEnd, &logs)?;
        }

        self.call_callbacks(CallbackEvent::TrainEnd, &logs)?;
        Ok(self.history.clone())
    }

    fn train_epoch(
        &mut self,
        train_loader: DataLoader,
        _epoch: usize,
    ) -> Result<HashMap<String, f32>, BellandeError> {
        let mut metrics = RunningMetrics::new();

        for (_batch_idx, (data, target)) in train_loader.enumerate() {
            let batch_logs = HashMap::new();
            self.call_callbacks(CallbackEvent::BatchBegin, &batch_logs)?;

            // Forward pass
            let data = data.to(self.device.clone());
            let target = target.to(self.device.clone());
            let output = self.model.forward(&data)?;
            let loss = self.loss_fn.forward(&output, &target)?;

            // Backward pass
            self.optimizer.zero_grad();
            let grad = self.loss_fn.backward(&output, &target)?;
            output.backward_with_grad(&grad)?;
            self.optimizer.step()?;

            // Update metrics
            metrics.update("loss", loss.data()[0]);

            let batch_logs = metrics.get_current();
            self.call_callbacks(CallbackEvent::BatchEnd, &batch_logs)?;
        }

        Ok(metrics.get_average())
    }

    fn validate(&mut self, val_loader: DataLoader) -> Result<HashMap<String, f32>, BellandeError> {
        let mut metrics = RunningMetrics::new();

        for (data, target) in val_loader {
            let data = data.to(self.device.clone());
            let target = target.to(self.device.clone());
            let output = self.model.forward(&data)?;
            let loss = self.loss_fn.forward(&output, &target)?;
            metrics.update("loss", loss.data()[0]);
        }

        Ok(metrics.get_average())
    }

    fn call_callbacks(
        &mut self,
        event: CallbackEvent,
        logs: &HashMap<String, f32>,
    ) -> Result<(), BellandeError> {
        for callback in &mut self.callbacks {
            match event {
                CallbackEvent::TrainBegin => callback.on_train_begin(logs)?,
                CallbackEvent::TrainEnd => callback.on_train_end(logs)?,
                CallbackEvent::EpochBegin => {
                    callback.on_epoch_begin(logs.get("epoch").unwrap().clone() as usize, logs)?
                }
                CallbackEvent::EpochEnd => {
                    callback.on_epoch_end(logs.get("epoch").unwrap().clone() as usize, logs)?
                }
                CallbackEvent::BatchBegin => callback.on_batch_begin(0, logs)?,
                CallbackEvent::BatchEnd => callback.on_batch_end(0, logs)?,
            }
        }
        Ok(())
    }
}
