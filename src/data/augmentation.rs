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

pub trait DataAugmentation: Send + Sync {
    fn apply(&self, data: &[f32], shape: &[usize]) -> Vec<f32>;
}

pub struct Compose {
    transformations: Vec<Box<dyn DataAugmentation>>,
}

impl Compose {
    pub fn new(transformations: Vec<Box<dyn DataAugmentation>>) -> Self {
        Compose { transformations }
    }
}

impl DataAugmentation for Compose {
    fn apply(&self, data: &[f32], shape: &[usize]) -> Vec<f32> {
        let mut result = data.to_vec();
        for transform in &self.transformations {
            result = transform.apply(&result, shape);
        }
        result
    }
}

pub struct RandomHorizontalFlip {
    p: f32,
}

impl RandomHorizontalFlip {
    pub fn new(p: f32) -> Self {
        RandomHorizontalFlip { p }
    }
}

impl DataAugmentation for RandomHorizontalFlip {
    fn apply(&self, data: &[f32], shape: &[usize]) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() > self.p {
            return data.to_vec();
        }

        let mut result = vec![0.0; data.len()];
        let channels = shape[0];
        let height = shape[1];
        let width = shape[2];

        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let src_idx = ((c * height + h) * width + w) as usize;
                    let dst_idx = ((c * height + h) * width + (width - 1 - w)) as usize;
                    result[dst_idx] = data[src_idx];
                }
            }
        }

        result
    }
}
