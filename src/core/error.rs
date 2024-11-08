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

use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum BellandeError {
    NoGradients,
    InvalidShape,
    DimensionMismatch,
    InvalidBackward,
    DeviceNotAvailable,
    InvalidDevice,
    InvalidDataType,
    InvalidInputs,
    CUDAError(String),
    IOError(std::io::Error),
    RuntimeError(String),
}

impl Error for BellandeError {}

impl fmt::Display for BellandeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BellandeError::NoGradients => write!(f, "Gradients not enabled for this tensor"),
            BellandeError::InvalidShape => write!(f, "Invalid tensor shape"),
            BellandeError::DimensionMismatch => write!(f, "Tensor dimensions do not match"),
            BellandeError::InvalidBackward => write!(f, "Invalid backward call"),
            BellandeError::DeviceNotAvailable => write!(f, "Requested device not available"),
            BellandeError::InvalidDevice => write!(f, "Invalid device specification"),
            BellandeError::InvalidDataType => write!(f, "Invalid data type"),
            BellandeError::InvalidInputs => write!(f, "Invalid number of inputs"),
            BellandeError::CUDAError(msg) => write!(f, "CUDA error: {}", msg),
            BellandeError::IOError(err) => write!(f, "IO error: {}", err),
            BellandeError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
        }
    }
}
