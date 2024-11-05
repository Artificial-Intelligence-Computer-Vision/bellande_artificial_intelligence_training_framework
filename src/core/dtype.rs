#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DataType {
    Float32,
    Float64,
    Int32,
    Int64,
}

impl DataType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
        }
    }
}
