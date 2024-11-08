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

use crate::core::error::BellandeError;
use crate::models::models::Model;
use crate::training::callbacks::Callback;
use glob::glob;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum CheckpointMode {
    Min,
    Max,
}

#[derive(Debug)]
pub struct ModelCheckpoint {
    filepath: String,
    monitor: String,
    save_best_only: bool,
    save_weights_only: bool,
    mode: CheckpointMode,
    best_value: f32,
    model: Option<Box<dyn Model>>,
    save_format: SaveFormat,
    verbose: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum SaveFormat {
    Json,
    Binary,
}

#[derive(Serialize, Deserialize)]
struct CheckpointMetadata {
    epoch: usize,
    best_value: f32,
    monitor: String,
    mode: CheckpointMode,
    metrics: HashMap<String, f32>,
}

impl ModelCheckpoint {
    pub fn new(
        filepath: String,
        monitor: String,
        save_best_only: bool,
        save_weights_only: bool,
        mode: CheckpointMode,
    ) -> Self {
        ModelCheckpoint {
            filepath,
            monitor,
            save_best_only,
            save_weights_only,
            mode,
            best_value: match mode {
                CheckpointMode::Min => f32::INFINITY,
                CheckpointMode::Max => f32::NEG_INFINITY,
            },
            model: None,
            save_format: SaveFormat::Binary,
            verbose: true,
        }
    }

    pub fn with_model(mut self, model: Box<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    pub fn with_save_format(mut self, format: SaveFormat) -> Self {
        self.save_format = format;
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    fn is_better(&self, current: f32) -> bool {
        match self.mode {
            CheckpointMode::Min => current < self.best_value,
            CheckpointMode::Max => current > self.best_value,
        }
    }

    fn save_checkpoint(
        &self,
        filepath: &Path,
        epoch: usize,
        metrics: &HashMap<String, f32>,
    ) -> Result<(), BellandeError> {
        // Create directory if it doesn't exist
        if let Some(parent) = filepath.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                BellandeError::IOError(format!("Failed to create directory: {}", e))
            })?;
        }

        // Save model or weights
        if let Some(model) = &self.model {
            if self.save_weights_only {
                self.save_weights(model.as_ref(), filepath)?;
            } else {
                self.save_model(model.as_ref(), filepath)?;
            }

            // Save metadata
            let metadata = CheckpointMetadata {
                epoch,
                best_value: self.best_value,
                monitor: self.monitor.clone(),
                mode: self.mode,
                metrics: metrics.clone(),
            };

            let metadata_path = filepath.with_extension("meta.json");
            let file = File::create(metadata_path).map_err(|e| {
                BellandeError::IOError(format!("Failed to create metadata file: {}", e))
            })?;

            serde_json::to_writer_pretty(file, &metadata).map_err(|e| {
                BellandeError::SerializationError(format!("Failed to write metadata: {}", e))
            })?;

            if self.verbose {
                println!("Saved checkpoint to {}", filepath.display());
            }
        }

        Ok(())
    }

    fn load_weights(&self, model: &mut dyn Model, path: &Path) -> Result<(), BellandeError> {
        match self.save_format {
            SaveFormat::Json => {
                let file = File::open(path).map_err(|e| {
                    BellandeError::IOError(format!("Failed to open weights file: {}", e))
                })?;
                let weights = serde_json::from_reader(file).map_err(|e| {
                    BellandeError::SerializationError(format!(
                        "Failed to deserialize weights: {}",
                        e
                    ))
                })?;
                model.set_weights(weights)?;
            }
            SaveFormat::Binary => {
                let file = File::open(path).map_err(|e| {
                    BellandeError::IOError(format!("Failed to open weights file: {}", e))
                })?;
                let weights = bincode::deserialize_from(file).map_err(|e| {
                    BellandeError::SerializationError(format!(
                        "Failed to deserialize weights: {}",
                        e
                    ))
                })?;
                model.set_weights(weights)?;
            }
        }
        Ok(())
    }

    fn load_model(&self, model: &mut dyn Model, path: &Path) -> Result<(), BellandeError> {
        match self.save_format {
            SaveFormat::Json => {
                let file = File::open(path).map_err(|e| {
                    BellandeError::IOError(format!("Failed to open model file: {}", e))
                })?;
                let state = serde_json::from_reader(file).map_err(|e| {
                    BellandeError::SerializationError(format!("Failed to deserialize model: {}", e))
                })?;
                model.load_state(state)?;
            }
            SaveFormat::Binary => {
                let file = File::open(path).map_err(|e| {
                    BellandeError::IOError(format!("Failed to open model file: {}", e))
                })?;
                let state = bincode::deserialize_from(file).map_err(|e| {
                    BellandeError::SerializationError(format!("Failed to deserialize model: {}", e))
                })?;
                model.load_state(state)?;
            }
        }
        Ok(())
    }

    fn cleanup_old_checkpoints(&self, keep_best_n: usize) -> Result<(), BellandeError> {
        let meta_pattern = self.filepath.replace("{epoch}", "*").replace("{val}", "*");
        let meta_pattern = format!("{}.meta.json", meta_pattern);

        let mut checkpoints: Vec<_> = glob::glob(&meta_pattern)
            .map_err(|e| {
                BellandeError::IOError(format!("Failed to read checkpoint directory: {}", e))
            })?
            .filter_map(Result::ok)
            .filter_map(|path| {
                if let Ok(file) = File::open(&path) {
                    if let Ok(metadata) = serde_json::from_reader::<_, CheckpointMetadata>(file) {
                        return Some((path, metadata));
                    }
                }
                None
            })
            .collect();

        // Sort checkpoints by performance
        checkpoints.sort_by(|a, b| {
            match self.mode {
                CheckpointMode::Min => a.1.best_value.partial_cmp(&b.1.best_value),
                CheckpointMode::Max => b.1.best_value.partial_cmp(&a.1.best_value),
            }
            .unwrap()
        });

        // Remove older checkpoints, keeping the best n
        for (path, _) in checkpoints.into_iter().skip(keep_best_n) {
            let base_path = path.with_extension("");
            // Remove model/weights file
            if let Err(e) = fs::remove_file(&base_path) {
                eprintln!(
                    "Warning: Failed to remove checkpoint file {}: {}",
                    base_path.display(),
                    e
                );
            }
            // Remove metadata file
            if let Err(e) = fs::remove_file(&path) {
                eprintln!(
                    "Warning: Failed to remove metadata file {}: {}",
                    path.display(),
                    e
                );
            }
        }

        Ok(())
    }
}

impl Callback for ModelCheckpoint {
    fn on_epoch_end(
        &mut self,
        epoch: usize,
        logs: &HashMap<String, f32>,
    ) -> Result<(), BellandeError> {
        if let Some(&current) = logs.get(&self.monitor) {
            if !self.save_best_only || self.is_better(current) {
                self.best_value = current;

                let filepath = PathBuf::from(
                    self.filepath
                        .replace("{epoch}", &epoch.to_string())
                        .replace("{val}", &format!("{:.4}", current)),
                );

                self.save_checkpoint(&filepath, epoch, logs)?;
            }
        }
        Ok(())
    }

    fn on_train_begin(&mut self, logs: &HashMap<String, f32>) -> Result<(), BellandeError> {
        // Check if checkpoint directory exists and create if necessary
        if let Some(parent) = Path::new(&self.filepath).parent() {
            fs::create_dir_all(parent).map_err(|e| {
                BellandeError::IOError(format!("Failed to create checkpoint directory: {}", e))
            })?;
        }

        // Try to load existing checkpoint metadata
        let meta_pattern = self.filepath.replace("{epoch}", "*").replace("{val}", "*");
        let meta_pattern = format!("{}.meta.json", meta_pattern);

        let existing_checkpoints: Vec<_> = glob::glob(&meta_pattern)
            .map_err(|e| {
                BellandeError::IOError(format!("Failed to read checkpoint directory: {}", e))
            })?
            .filter_map(Result::ok)
            .collect();

        if !existing_checkpoints.is_empty() {
            // Find the best checkpoint based on the monitoring metric
            let mut best_checkpoint = None;
            let mut best_value = match self.mode {
                CheckpointMode::Min => f32::INFINITY,
                CheckpointMode::Max => f32::NEG_INFINITY,
            };

            for checkpoint_path in existing_checkpoints {
                if let Ok(file) = File::open(&checkpoint_path) {
                    if let Ok(metadata) = serde_json::from_reader::<_, CheckpointMetadata>(file) {
                        if self.is_better(metadata.best_value) {
                            best_value = metadata.best_value;
                            best_checkpoint = Some((checkpoint_path, metadata));
                        }
                    }
                }
            }

            // Load the best checkpoint if found
            if let Some((path, metadata)) = best_checkpoint {
                self.best_value = metadata.best_value;

                if self.verbose {
                    println!(
                        "Resuming from checkpoint: {} (best {} = {})",
                        path.display(),
                        self.monitor,
                        self.best_value
                    );
                }

                // Load model or weights if available
                if let Some(model) = &mut self.model {
                    let model_path = path.with_extension(match self.save_format {
                        SaveFormat::Json => "json",
                        SaveFormat::Binary => "bin",
                    });

                    if model_path.exists() {
                        if self.save_weights_only {
                            self.load_weights(model.as_mut(), &model_path)?;
                        } else {
                            self.load_model(model.as_mut(), &model_path)?;
                        }
                    }
                }
            }
        } else if self.verbose {
            println!("No existing checkpoints found, starting from scratch");
        }

        Ok(())
    }

    fn on_train_end(&mut self, logs: &HashMap<String, f32>) -> Result<(), BellandeError> {
        // Save final checkpoint regardless of performance
        if let Some(&final_value) = logs.get(&self.monitor) {
            let filepath = PathBuf::from(
                self.filepath
                    .replace("{epoch}", "final")
                    .replace("{val}", &format!("{:.4}", final_value)),
            );

            // Create final checkpoint metadata
            let metadata = CheckpointMetadata {
                epoch: usize::MAX, // Indicate this is the final checkpoint
                best_value: self.best_value,
                monitor: self.monitor.clone(),
                mode: self.mode,
                metrics: logs.clone(),
            };

            // Save the checkpoint
            if let Some(model) = &self.model {
                if self.save_weights_only {
                    self.save_weights(model.as_ref(), &filepath)?;
                } else {
                    self.save_model(model.as_ref(), &filepath)?;
                }

                // Save metadata
                let metadata_path = filepath.with_extension("meta.json");
                let file = File::create(metadata_path).map_err(|e| {
                    BellandeError::IOError(format!("Failed to create final metadata file: {}", e))
                })?;

                serde_json::to_writer_pretty(file, &metadata).map_err(|e| {
                    BellandeError::SerializationError(format!(
                        "Failed to write final metadata: {}",
                        e
                    ))
                })?;

                if self.verbose {
                    println!(
                        "Saved final checkpoint to {} (best {} = {})",
                        filepath.display(),
                        self.monitor,
                        self.best_value
                    );
                }
            }

            // Clean up old checkpoints if configured
            if let Some(keep_best_n) = self.keep_best_n {
                self.cleanup_old_checkpoints(keep_best_n)?;
            }
        }

        Ok(())
    }
}
