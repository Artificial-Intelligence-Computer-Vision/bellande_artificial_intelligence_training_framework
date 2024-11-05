#[derive(Debug)]
pub enum BellandeError {
    NoGradients,
    InvalidShape,
    DimensionMismatch,
    InvalidBackward,
    DeviceNotAvailable,
    CUDAError(String),
    IOError(std::io::Error),
    Other(String),
}

impl std::error::Error for BellandeError {}

impl std::fmt::Display for BellandeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            BellandeError::NoGradients => write!(f, "Gradients not enabled for this tensor"),
            BellandeError::InvalidShape => write!(f, "Invalid tensor shape"),
            BellandeError::DimensionMismatch => write!(f, "Tensor dimensions do not match"),
            BellandeError::InvalidBackward => write!(f, "Invalid backward call"),
            BellandeError::DeviceNotAvailable => write!(f, "Requested device not available"),
            BellandeError::CUDAError(msg) => write!(f, "CUDA error: {}", msg),
            BellandeError::IOError(err) => write!(f, "IO error: {}", err),
            BellandeError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}
