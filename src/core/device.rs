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

#[derive(Clone, Debug, PartialEq)]
pub enum Device {
    CPU,
    CUDA(usize),
}

impl Device {
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::CUDA(_))
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::CPU)
    }

    pub fn cuda_device_count() -> usize {
        #[cfg(feature = "cuda")]
        {
            // CUDA device count implementation would go here
            unimplemented!()
        }
        #[cfg(not(feature = "cuda"))]
        0
    }

    pub fn default() -> Self {
        Device::CPU
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Device::CPU => write!(f, "cpu"),
            Device::CUDA(device_id) => write!(f, "cuda:{}", device_id),
        }
    }
}
