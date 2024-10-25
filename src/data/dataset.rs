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

pub struct Dataset {
    data: Vec<(Vec<f32>, f32)>,
    input_shape: Vec<usize>,
    augmentation: Option<Box<dyn DataAugmentation>>,
}

impl Dataset {
    pub fn new(data: Vec<(Vec<f32>, f32)>, input_shape: Vec<usize>) -> Self {
        Dataset {
            data,
            input_shape,
            augmentation: None,
        }
    }

    pub fn with_augmentation(mut self, augmentation: Box<dyn DataAugmentation>) -> Self {
        self.augmentation = Some(augmentation);
        self
    }

    pub fn get(&self, idx: usize) -> (Tensor, f32) {
        let (mut x, y) = self.data[idx].clone();

        if let Some(aug) = &self.augmentation {
            x = aug.apply(&x, &self.input_shape);
        }

        (Tensor::new(x, self.input_shape.clone(), false), y)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn input_shape(&self) -> Vec<usize> {
        self.input_shape.clone()
    }
}
