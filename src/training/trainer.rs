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

pub struct TrainingConfig {
    pub num_epochs: usize,
    pub save_frequency: usize,
    pub checkpoint_dir: String,
    pub device: Device,
    pub distributed: bool,
    pub world_size: usize,
    pub rank: usize,
}

pub struct Trainer<M: Model> {
    model: M,
    optimizer: Box<dyn Optimizer>,
    loss_fn: Box<dyn Loss>,
    metrics: Vec<Box<dyn Metric>>,
    config: TrainingConfig,
    validator: Option<Validator<M>>,
    distributed_trainer: Option<DistributedTrainer>,
}

impl<M: Model> Trainer<M> {
    pub fn new(
        model: M,
        optimizer: Box<dyn Optimizer>,
        loss_fn: Box<dyn Loss>,
        metrics: Vec<Box<dyn Metric>>,
        config: TrainingConfig,
    ) -> Self {
        let distributed_trainer = if config.distributed {
            Some(DistributedTrainer::new(
                Box::new(model.clone()),
                optimizer.clone(),
                config.world_size,
                config.rank,
            ))
        } else {
            None
        };

        Trainer {
            model,
            optimizer,
            loss_fn,
            metrics,
            config,
            validator: None,
            distributed_trainer,
        }
    }

    pub fn with_validator(mut self, validator: Validator<M>) -> Self {
        self.validator = Some(validator);
        self
    }

    pub async fn train(
        &mut self,
        train_dataloader: &mut DataLoader,
        val_dataloader: Option<&mut DataLoader>,
    ) -> TrainingHistory {
        let mut history = TrainingHistory::new();

        for epoch in 0..self.config.num_epochs {
            // Training phase
            self.model.train();
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            for metric in &mut self.metrics {
                metric.reset();
            }

            for (batch_x, batch_y) in train_dataloader {
                let loss = if let Some(trainer) = &self.distributed_trainer {
                    trainer.train_step((batch_x, batch_y)).await
                } else {
                    self.train_step((batch_x, batch_y))
                };

                epoch_loss += loss;
                num_batches += 1;
            }

            let avg_loss = epoch_loss / num_batches as f32;
            
            // Collect training metrics
            let mut metrics = HashMap::new();
            metrics.insert("loss".to_string(), avg_loss);
            for metric in &self.metrics {
                metrics.insert(metric.name().to_string(), metric.compute());
            }

            // Validation phase
            if let Some(dataloader) = val_dataloader {
                if let Some(validator) = &mut self.validator {
                    let val_metrics = validator.validate(dataloader);
                    for (name, value) in val_metrics {
                        metrics.insert(format!("val_{}", name), value);
                    }
                }
            }

            // Save checkpoint if needed
            if epoch % self.config.save_frequency == 0 {
                let checkpoint_path = format!("{}/checkpoint_epoch_{}.mbellande", 
                    self.config.checkpoint_dir, epoch);
                self.save_checkpoint(&checkpoint_path, epoch, &metrics)?;
            }

            // Update history
            history.update(epoch, metrics);

            // Print progress
            self.print_progress(epoch, &metrics);
        }

        history
    }

    fn train_step(&mut self, batch: (Tensor, Tensor)) -> f32 {
        let (batch_x, batch_y) = batch;
        
        self.optimizer.zero_grad();

        // Forward pass
        let prediction = self.model.forward(&batch_x);
        let loss = self.loss_fn.forward(&prediction, &batch_y);

        // Update metrics
        for metric in &mut self.metrics {
            metric.update(&prediction, &batch_y);
        }

        // Backward pass
        let grad = self.loss_fn.backward(&prediction, &batch_y);
        self.model.backward(&grad);

        // Update parameters
        self.optimizer.step();

        loss
    }

    fn save_checkpoint(
        &self,
        path: &str,
        epoch: usize,
        metrics: &HashMap<String, f32>,
    ) -> std::io::Result<()> {
        let checkpoint = Checkpoint {
            epoch,
            model_state: self.model.state_dict(),
            optimizer_state: self.optimizer.state_dict(),
            metrics: metrics.clone(),
        };

        save_model(&checkpoint, path)
    }

    fn print_progress(&self, epoch: usize, metrics: &HashMap<String, f32>) {
        print!("Epoch [{}/{}] ", epoch + 1, self.config.num_epochs);
        for (name, value) in metrics {
            print!("{}: {:.4} ", name, value);
        }
        println!();
    }
}
