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

use crate::core::{error::BellandeError, tensor::Tensor};
use crate::layer::dropout::Dropout;
use crate::layer::{
    activation::ReLU, avgpool2d::AvgPool2d, conv::Conv2d, linear::Linear, pooling::MaxPool2d,
};
use crate::models::sequential::Sequential;

pub struct VGG {
    features: Sequential,
    avgpool: AvgPool2d,
    classifier: Sequential,
}

impl VGG {
    pub fn vgg16(num_classes: usize) -> Self {
        let mut features = Sequential::new();

        // Block 1
        features.add(Box::new(Conv2d::new(3, 64, 3, 1, 1, true)));
        features.add(Box::new(ReLU::new()));
        features.add(Box::new(Conv2d::new(64, 64, 3, 1, 1, true)));
        features.add(Box::new(ReLU::new()));
        features.add(Box::new(MaxPool2d::new(2, 2)));

        // Block 2
        features.add(Box::new(Conv2d::new(64, 128, 3, 1, 1, true)));
        features.add(Box::new(ReLU::new()));
        features.add(Box::new(Conv2d::new(128, 128, 3, 1, 1, true)));
        features.add(Box::new(ReLU::new()));
        features.add(Box::new(MaxPool2d::new(2, 2)));

        // Block 3
        features.add(Box::new(Conv2d::new(128, 256, 3, 1, 1, true)));
        features.add(Box::new(ReLU::new()));
        features.add(Box::new(Conv2d::new(256, 256, 3, 1, 1, true)));
        features.add(Box::new(ReLU::new()));
        features.add(Box::new(Conv2d::new(256, 256, 3, 1, 1, true)));
        features.add(Box::new(ReLU::new()));
        features.add(Box::new(MaxPool2d::new(2, 2)));

        // Block 4
        features.add(Box::new(Conv2d::new(256, 512, 3, 1, 1, true)));
        features.add(Box::new(ReLU::new()));
        features.add(Box::new(Conv2d::new(512, 512, 3, 1, 1, true)));
        features.add(Box::new(ReLU::new()));
        features.add(Box::new(Conv2d::new(512, 512, 3, 1, 1, true)));
        features.add(Box::new(ReLU::new()));
        features.add(Box::new(MaxPool2d::new(2, 2)));

        // Block 5
        features.add(Box::new(Conv2d::new(512, 512, 3, 1, 1, true)));
        features.add(Box::new(ReLU::new()));
        features.add(Box::new(Conv2d::new(512, 512, 3, 1, 1, true)));
        features.add(Box::new(ReLU::new()));
        features.add(Box::new(Conv2d::new(512, 512, 3, 1, 1, true)));
        features.add(Box::new(ReLU::new()));
        features.add(Box::new(MaxPool2d::new(2, 2)));

        let mut classifier = Sequential::new();
        classifier.add(Box::new(Linear::new(512 * 7 * 7, 4096, true)));
        classifier.add(Box::new(ReLU::new()));
        classifier.add(Box::new(Dropout::new(0.5)));
        classifier.add(Box::new(Linear::new(4096, 4096, true)));
        classifier.add(Box::new(ReLU::new()));
        classifier.add(Box::new(Dropout::new(0.5)));
        classifier.add(Box::new(Linear::new(4096, num_classes, true)));

        VGG {
            features,
            avgpool: AvgPool2d::new(7, 1),
            classifier,
        }
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor, BellandeError> {
        let mut out = self.features.forward(x)?;
        out = self.avgpool.forward(&out)?;
        out = out.reshape(&[out.shape[0], -1])?;
        out = self.classifier.forward(&out)?;
        Ok(out)
    }
}
