#[derive(Clone, Debug, PartialEq)]
pub enum Device {
    CPU,
    CUDA(usize),
}

impl Device {
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::CUDA(_))
    }

    pub fn cuda_device_count() -> usize {
        #[cfg(feature = "cuda")]
        {
            // Implement CUDA device count check
            unimplemented!()
        }
        #[cfg(not(feature = "cuda"))]
        0
    }
}
