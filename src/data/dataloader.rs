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

use crate::core::tensor::Tensor;
use crate::data::{dataset::Dataset, sampler::Sampler};
use rayon::prelude::*;
use std::sync::Arc;

pub struct DataLoader {
    dataset: Arc<Dataset>,
    batch_size: usize,
    shuffle: bool,
    num_workers: usize,
    sampler: Option<Box<dyn Sampler>>,
    drop_last: bool,
}

impl DataLoader {
    pub fn new(
        dataset: Dataset,
        batch_size: usize,
        shuffle: bool,
        num_workers: usize,
        sampler: Option<Box<dyn Sampler>>,
        drop_last: bool,
    ) -> Self {
        DataLoader {
            dataset: Arc::new(dataset),
            batch_size,
            shuffle,
            num_workers,
            sampler,
            drop_last,
        }
    }

    pub fn iter(&self) -> DataLoaderIterator {
        DataLoaderIterator {
            dataloader: self,
            index: 0,
        }
    }
}

pub struct DataLoaderIterator<'a> {
    dataloader: &'a DataLoader,
    index: usize,
}

impl<'a> Iterator for DataLoaderIterator<'a> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataloader.dataset.len() {
            return None;
        }

        let batch_indices: Vec<usize> = if let Some(sampler) = &self.dataloader.sampler {
            sampler.sample(self.dataloader.batch_size)
        } else if self.dataloader.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..self.dataloader.dataset.len()).collect();
            indices.shuffle(&mut rng);
            indices[..self.dataloader.batch_size].to_vec()
        } else {
            (self.index..self.index + self.dataloader.batch_size)
                .filter(|&i| i < self.dataloader.dataset.len())
                .collect()
        };

        let batch: Vec<(Tensor, Tensor)> = if self.dataloader.num_workers > 1 {
            batch_indices
                .par_iter()
                .map(|&idx| self.dataloader.dataset.get(idx))
                .collect()
        } else {
            batch_indices
                .iter()
                .map(|&idx| self.dataloader.dataset.get(idx))
                .collect()
        };

        if batch.is_empty() {
            return None;
        }

        self.index += self.dataloader.batch_size;

        Some(collate_batch(batch))
    }
}

fn collate_batch(batch: Vec<(Tensor, Tensor)>) -> (Tensor, Tensor) {
    // Implement batch collation
    unimplemented!()
}
