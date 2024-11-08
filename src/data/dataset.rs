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
use crate::core::tensor::Tensor;
use crate::data::augmentation::Transform;
use std::path::PathBuf;

pub trait Dataset: Send + Sync {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> (Tensor, Tensor);
}

pub struct ImageFolder {
    root: PathBuf,
    samples: Vec<(PathBuf, usize)>,
    transform: Option<Box<dyn Transform>>,
    target_transform: Option<Box<dyn Transform>>,
}

impl ImageFolder {
    pub fn new(
        root: PathBuf,
        transform: Option<Box<dyn Transform>>,
        target_transform: Option<Box<dyn Transform>>,
    ) -> Result<Self, BellandeError> {
        let mut samples = Vec::new();
        let mut class_to_idx = std::collections::HashMap::new();

        for (idx, entry) in std::fs::read_dir(&root)?.enumerate() {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                class_to_idx.insert(
                    path.file_name().unwrap().to_string_lossy().into_owned(),
                    idx,
                );
                for image in std::fs::read_dir(path)? {
                    let image = image?;
                    if image
                        .path()
                        .extension()
                        .map_or(false, |ext| ext == "jpg" || ext == "jpeg" || ext == "png")
                    {
                        samples.push((image.path(), idx));
                    }
                }
            }
        }

        Ok(ImageFolder {
            root,
            samples,
            transform,
            target_transform,
        })
    }
}

impl Dataset for ImageFolder {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> (Tensor, Tensor) {
        let (path, class_idx) = &self.samples[index];
        let image = image::open(path).unwrap();
        let mut input = image_to_tensor(image);
        let mut target = Tensor::new(
            vec![*class_idx as f32],
            vec![1],
            false,
            input.device.clone(),
            input.dtype,
        );

        if let Some(transform) = &self.transform {
            input = transform.apply(&input).unwrap();
        }

        if let Some(target_transform) = &self.target_transform {
            target = target_transform.apply(&target).unwrap();
        }

        (input, target)
    }
}
