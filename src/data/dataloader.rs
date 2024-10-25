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

use image::{ImageBuffer, Rgb};
use rand::seq::SliceRandom;
use std::path::Path;

pub struct DataLoader {
    batch_size: usize,
    shuffle: bool,
    dataset: Dataset,
    indices: Vec<usize>,
    current_idx: usize,
}

impl DataLoader {
    pub fn new(dataset: Dataset, batch_size: usize, shuffle: bool) -> Self {
        let indices: Vec<usize> = (0..dataset.len()).collect();
        DataLoader {
            batch_size,
            shuffle,
            dataset,
            indices,
            current_idx: 0,
        }
    }
}

impl Iterator for DataLoader {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.len() {
            if self.shuffle {
                let mut rng = rand::thread_rng();
                self.indices.shuffle(&mut rng);
            }
            self.current_idx = 0;
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.dataset.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];

        let mut batch_x = Vec::new();
        let mut batch_y = Vec::new();

        for &idx in batch_indices {
            let (x, y) = self.dataset.get(idx);
            batch_x.extend_from_slice(&x.data);
            batch_y.push(y);
        }

        self.current_idx = end_idx;

        Some((
            Tensor::new(
                batch_x,
                vec![batch_indices.len(), self.dataset.input_shape()],
                false,
            ),
            Tensor::new(batch_y, vec![batch_indices.len()], false),
        ))
    }
}
