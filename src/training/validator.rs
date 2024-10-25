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

pub struct Validator<M: Model> {
    model: M,
    metrics: Vec<Box<dyn Metric>>,
}

impl<M: Model> Validator<M> {
    pub fn new(model: M, metrics: Vec<Box<dyn Metric>>) -> Self {
        Validator { model, metrics }
    }

    pub fn validate(&mut self, dataloader: &mut DataLoader) -> HashMap<String, f32> {
        self.model.eval();

        for metric in &mut self.metrics {
            metric.reset();
        }

        for (batch_x, batch_y) in dataloader {
            let prediction = self.model.forward(&batch_x);

            for metric in &mut self.metrics {
                metric.update(&prediction, &batch_y);
            }
        }

        let mut results = HashMap::new();
        for metric in &self.metrics {
            results.insert(metric.name().to_string(), metric.compute());
        }

        self.model.train();
        results
    }
}
