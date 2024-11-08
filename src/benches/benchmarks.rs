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

#![feature(test)]
extern crate test;

use bellande_training_framework::{
    core::{DataType, Device, Tensor},
    layer::{BatchNorm2d, Conv2d, Linear},
    models::{Model, ResNet, VGG},
    optim::{Adam, RMSprop, SGD},
};
use test::Bencher;

// Tensor Operations Benchmarks
#[bench]
fn bench_tensor_matmul(b: &mut Bencher) {
    let a = Tensor::randn(&[1000, 1000], Device::CPU, DataType::Float32);
    let c = Tensor::randn(&[1000, 1000], Device::CPU, DataType::Float32);

    b.iter(|| {
        let _ = a.matmul(&c).unwrap();
    });
}

#[bench]
fn bench_tensor_elementwise_ops(b: &mut Bencher) {
    let a = Tensor::randn(&[1000000], Device::CPU, DataType::Float32);
    let c = Tensor::randn(&[1000000], Device::CPU, DataType::Float32);

    b.iter(|| {
        let _ = (&a + &c).unwrap();
        let _ = (&a * &c).unwrap();
        let _ = (&a - &c).unwrap();
    });
}

// Layer Benchmarks
#[bench]
fn bench_conv2d_forward(b: &mut Bencher) {
    let layer = Conv2d::new(64, 128, 3, 1, 1, true);
    let input = Tensor::randn(&[32, 64, 32, 32], Device::CPU, DataType::Float32);

    b.iter(|| {
        let _ = layer.forward(&input).unwrap();
    });
}

#[bench]
fn bench_linear_forward(b: &mut Bencher) {
    let layer = Linear::new(1024, 1024, true);
    let input = Tensor::randn(&[32, 1024], Device::CPU, DataType::Float32);

    b.iter(|| {
        let _ = layer.forward(&input).unwrap();
    });
}

#[bench]
fn bench_batchnorm_forward(b: &mut Bencher) {
    let layer = BatchNorm2d::new(64, 1e-5, 0.1, true);
    let input = Tensor::randn(&[32, 64, 32, 32], Device::CPU, DataType::Float32);

    b.iter(|| {
        let _ = layer.forward(&input).unwrap();
    });
}

// Model Benchmarks
#[bench]
fn bench_resnet18_forward(b: &mut Bencher) {
    let model = ResNet::resnet18(1000);
    let input = Tensor::randn(&[1, 3, 224, 224], Device::CPU, DataType::Float32);

    b.iter(|| {
        let _ = model.forward(&input).unwrap();
    });
}

#[bench]
fn bench_vgg16_forward(b: &mut Bencher) {
    let model = VGG::vgg16(1000);
    let input = Tensor::randn(&[1, 3, 224, 224], Device::CPU, DataType::Float32);

    b.iter(|| {
        let _ = model.forward(&input).unwrap();
    });
}

// Optimizer Benchmarks
#[bench]
fn bench_adam_step(b: &mut Bencher) {
    let params = vec![
        Tensor::randn(&[1024, 1024], Device::CPU, DataType::Float32),
        Tensor::randn(&[1024], Device::CPU, DataType::Float32),
    ];
    let mut optimizer = Adam::new(params, 0.001, (0.9, 0.999), 1e-8, 0.0);

    b.iter(|| {
        optimizer.step().unwrap();
    });
}

#[bench]
fn bench_sgd_step(b: &mut Bencher) {
    let params = vec![
        Tensor::randn(&[1024, 1024], Device::CPU, DataType::Float32),
        Tensor::randn(&[1024], Device::CPU, DataType::Float32),
    ];
    let mut optimizer = SGD::new(params, 0.01, 0.9, 0.0, false);

    b.iter(|| {
        optimizer.step().unwrap();
    });
}

// Memory Benchmarks
#[bench]
fn bench_memory_allocation(b: &mut Bencher) {
    b.iter(|| {
        let _ = Tensor::randn(&[1024, 1024], Device::CPU, DataType::Float32);
    });
}

// Training Pipeline Benchmarks
#[bench]
fn bench_training_step(b: &mut Bencher) {
    // Setup model and training components
    let model = ResNet::resnet18(1000);
    let optimizer = Adam::new(model.parameters(), 0.001, (0.9, 0.999), 1e-8, 0.0);
    let loss_fn = CrossEntropyLoss::new();

    // Create dummy batch
    let input = Tensor::randn(&[32, 3, 224, 224], Device::CPU, DataType::Float32);
    let target = Tensor::randn(&[32], Device::CPU, DataType::Float32);

    b.iter(|| {
        // Forward pass
        let output = model.forward(&input).unwrap();
        let loss = loss_fn.forward(&output, &target).unwrap();

        // Backward pass
        optimizer.zero_grad();
        loss.backward().unwrap();
        optimizer.step().unwrap();
    });
}

// Device Transfer Benchmarks
#[bench]
#[cfg(feature = "cuda")]
fn bench_cpu_to_cuda_transfer(b: &mut Bencher) {
    let tensor = Tensor::randn(&[1024, 1024], Device::CPU, DataType::Float32);

    b.iter(|| {
        let _ = tensor.to(Device::CUDA(0)).unwrap();
    });
}

// Serialization Benchmarks
#[bench]
fn bench_model_save_load(b: &mut Bencher) {
    let model = ResNet::resnet18(1000);
    let path = "benchmark_model.pt";

    b.iter(|| {
        model.save(path).unwrap();
        let _ = ResNet::load(path).unwrap();
    });

    std::fs::remove_file(path).unwrap();
}

// Custom Layer Benchmarks
struct BenchmarkLayer {
    weight: Tensor,
    bias: Tensor,
}

impl BenchmarkLayer {
    fn new(in_features: usize, out_features: usize) -> Self {
        BenchmarkLayer {
            weight: Tensor::randn(&[out_features, in_features], Device::CPU, DataType::Float32),
            bias: Tensor::randn(&[out_features], Device::CPU, DataType::Float32),
        }
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, BellandeError> {
        let output = input.matmul(&self.weight.transpose()?)?;
        output + &self.bias
    }
}

#[bench]
fn bench_custom_layer(b: &mut Bencher) {
    let layer = BenchmarkLayer::new(1024, 1024);
    let input = Tensor::randn(&[32, 1024], Device::CPU, DataType::Float32);

    b.iter(|| {
        let _ = layer.forward(&input).unwrap();
    });
}

// Batch Processing Benchmarks
#[bench]
fn bench_batch_processing(b: &mut Bencher) {
    let batch_size = 32;
    let feature_size = 1024;
    let batches: Vec<Tensor> = (0..batch_size)
        .map(|_| Tensor::randn(&[feature_size], Device::CPU, DataType::Float32))
        .collect();

    b.iter(|| {
        let _ = process_batch(&batches).unwrap();
    });
}

fn process_batch(batch: &[Tensor]) -> Result<Tensor, BellandeError> {
    let mut result = batch[0].clone();
    for tensor in &batch[1..] {
        result = (&result + tensor)?;
    }
    Ok(result)
}

// Configuration
#[derive(Debug)]
pub struct BenchmarkConfig {
    pub batch_sizes: Vec<usize>,
    pub model_sizes: Vec<usize>,
    pub iterations: usize,
    pub warmup_iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        BenchmarkConfig {
            batch_sizes: vec![1, 8, 16, 32, 64],
            model_sizes: vec![64, 128, 256, 512],
            iterations: 100,
            warmup_iterations: 10,
        }
    }
}

// Benchmark Suite
pub struct BenchmarkSuite {
    config: BenchmarkConfig,
    results: HashMap<String, Vec<Duration>>,
}

impl BenchmarkSuite {
    pub fn new(config: BenchmarkConfig) -> Self {
        BenchmarkSuite {
            config,
            results: HashMap::new(),
        }
    }

    pub fn run_all(&mut self) -> Result<(), BellandeError> {
        // Run tensor operation benchmarks
        self.benchmark_tensor_ops()?;

        // Run model benchmarks
        self.benchmark_models()?;

        // Run optimizer benchmarks
        self.benchmark_optimizers()?;

        // Run memory benchmarks
        self.benchmark_memory()?;

        Ok(())
    }

    pub fn get_results(&self) -> &HashMap<String, Vec<Duration>> {
        &self.results
    }

    pub fn print_results(&self) {
        println!("Benchmark Results:");
        for (name, durations) in &self.results {
            let avg = durations.iter().sum::<Duration>() / durations.len() as u32;
            println!("{}: {:?} average", name, avg);
        }
    }

    fn benchmark_tensor_ops(&mut self) -> Result<(), BellandeError> {
        for &size in &self.config.model_sizes {
            // Benchmark matrix multiplication
            let name = format!("matmul_{}", size);
            self.benchmark_operation(&name, || {
                let a = Tensor::randn(&[size, size], Device::CPU, DataType::Float32);
                let b = Tensor::randn(&[size, size], Device::CPU, DataType::Float32);
                a.matmul(&b)
            })?;
        }
        Ok(())
    }

    fn benchmark_models(&mut self) -> Result<(), BellandeError> {
        for &batch_size in &self.config.batch_sizes {
            // Benchmark ResNet forward pass
            let name = format!("resnet18_batch_{}", batch_size);
            let model = ResNet::resnet18(1000);
            let input = Tensor::randn(&[batch_size, 3, 224, 224], Device::CPU, DataType::Float32);

            self.benchmark_operation(&name, || model.forward(&input))?;
        }
        Ok(())
    }

    fn benchmark_optimizers(&mut self) -> Result<(), BellandeError> {
        // Implementation of optimizer benchmarks
        Ok(())
    }

    fn benchmark_memory(&mut self) -> Result<(), BellandeError> {
        // Implementation of memory benchmarks
        Ok(())
    }

    fn benchmark_operation<F, T>(&mut self, name: &str, operation: F) -> Result<(), BellandeError>
    where
        F: Fn() -> Result<T, BellandeError>,
    {
        let mut durations = Vec::with_capacity(self.config.iterations);

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            operation()?;
        }

        // Actual benchmarking
        for _ in 0..self.config.iterations {
            let start = Instant::now();
            operation()?;
            durations.push(start.elapsed());
        }

        self.results.insert(name.to_string(), durations);
        Ok(())
    }
}
