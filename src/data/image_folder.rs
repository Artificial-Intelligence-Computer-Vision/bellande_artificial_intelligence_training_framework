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

use crate::core::{device::Device, dtype::DataType, error::BellandeError, tensor::Tensor};
use crate::data::augmentation::Transform;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Cursor, Read, Result as IoResult};
use std::path::PathBuf;
use std::sync::Arc;

/// A reader that allows reading individual bits from a byte stream
pub struct BitReader<R: Read> {
    reader: R,
    buffer: u8,
    bits_remaining: u8,
}

/// Image format enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
enum ImageFormat {
    JPEG,
    PNG,
    Unknown,
}

/// RGB pixel structure
#[derive(Debug, Clone, Copy)]
struct RGBPixel {
    r: u8,
    g: u8,
    b: u8,
}

/// Trait defining the interface for datasets
pub trait Dataset: Send + Sync {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Result<(Tensor, Tensor), BellandeError>;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn num_classes(&self) -> usize;
}

/// Structure for managing image datasets organized in folders
pub struct ImageFolder {
    root: PathBuf,
    samples: Vec<(PathBuf, usize)>,
    transform: Option<Box<dyn Transform>>,
    target_transform: Option<Box<dyn Transform>>,
    class_to_idx: HashMap<String, usize>,
    cache: Option<HashMap<PathBuf, Arc<Tensor>>>,
    cache_size: usize,
}

impl<R: Read> BitReader<R> {
    /// Creates a new BitReader from a byte stream
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: 0,
            bits_remaining: 0,
        }
    }

    /// Reads a single bit from the stream
    pub fn read_bit(&mut self) -> IoResult<bool> {
        if self.bits_remaining == 0 {
            let mut byte = [0u8; 1];
            self.reader.read_exact(&mut byte)?;
            self.buffer = byte[0];
            self.bits_remaining = 8;
        }

        self.bits_remaining -= 1;
        Ok(((self.buffer >> self.bits_remaining) & 1) == 1)
    }

    /// Reads multiple bits and returns them as a u32
    pub fn read_bits(&mut self, mut count: u8) -> IoResult<u32> {
        let mut result = 0u32;

        while count > 0 {
            result = (result << 1) | (if self.read_bit()? { 1 } else { 0 });
            count -= 1;
        }

        Ok(result)
    }
}

impl ImageFolder {
    // Define the constant within the implementation
    const JPEG_NATURAL_ORDER: [usize; 64] = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
        20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ];

    /// Creates a new ImageFolder dataset
    pub fn new(
        root: PathBuf,
        transform: Option<Box<dyn Transform>>,
        target_transform: Option<Box<dyn Transform>>,
    ) -> Result<Self, BellandeError> {
        let mut samples = Vec::new();
        let mut class_to_idx = HashMap::new();

        Self::validate_root_directory(&root)?;
        Self::scan_directory(&root, &mut samples, &mut class_to_idx)?;

        if samples.is_empty() {
            return Err(BellandeError::IOError("No valid images found".to_string()));
        }

        Ok(ImageFolder {
            root,
            samples,
            transform,
            target_transform,
            class_to_idx,
            cache: Some(HashMap::new()),
            cache_size: 1000, // Default cache size
        })
    }

    /// Creates a new ImageFolder with specified cache size
    pub fn with_cache_size(
        root: PathBuf,
        transform: Option<Box<dyn Transform>>,
        target_transform: Option<Box<dyn Transform>>,
        cache_size: usize,
    ) -> Result<Self, BellandeError> {
        let mut folder = Self::new(root, transform, target_transform)?;
        folder.cache_size = cache_size;
        Ok(folder)
    }

    /// Validates the root directory exists and is a directory
    fn validate_root_directory(root: &PathBuf) -> Result<(), BellandeError> {
        if !root.exists() || !root.is_dir() {
            return Err(BellandeError::IOError("Invalid root directory".to_string()));
        }
        Ok(())
    }

    /// Scans the directory structure and builds the dataset
    fn scan_directory(
        root: &PathBuf,
        samples: &mut Vec<(PathBuf, usize)>,
        class_to_idx: &mut HashMap<String, usize>,
    ) -> Result<(), BellandeError> {
        for (idx, entry) in fs::read_dir(root)?.enumerate() {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let class_name = path
                    .file_name()
                    .ok_or_else(|| {
                        BellandeError::IOError("Invalid class directory name".to_string())
                    })?
                    .to_string_lossy()
                    .into_owned();

                class_to_idx.insert(class_name, idx);

                // Scan for images recursively
                Self::scan_images(&path, idx, samples)?;
            }
        }
        Ok(())
    }

    /// Scans for images in a directory
    fn scan_images(
        path: &PathBuf,
        class_idx: usize,
        samples: &mut Vec<(PathBuf, usize)>,
    ) -> Result<(), BellandeError> {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && Self::is_valid_image(&path) {
                samples.push((path, class_idx));
            } else if path.is_dir() {
                Self::scan_images(&path, class_idx, samples)?;
            }
        }
        Ok(())
    }

    /// Checks if a file is a valid image based on its extension and header
    fn is_valid_image(path: &PathBuf) -> bool {
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy().to_lowercase();
            if matches!(ext.as_str(), "jpg" | "jpeg" | "png") {
                if let Ok(bytes) = Self::read_image_file(path) {
                    return Self::detect_image_format(&bytes) != ImageFormat::Unknown;
                }
            }
        }
        false
    }

    /// Reads an image file to bytes
    fn read_image_file(path: &PathBuf) -> Result<Vec<u8>, BellandeError> {
        let mut file = File::open(path)
            .map_err(|e| BellandeError::IOError(format!("Failed to open image file: {}", e)))?;

        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| BellandeError::IOError(format!("Failed to read image file: {}", e)))?;

        Ok(bytes)
    }

    /// Detects image format from bytes
    fn detect_image_format(bytes: &[u8]) -> ImageFormat {
        if bytes.len() < 4 {
            return ImageFormat::Unknown;
        }

        match &bytes[0..4] {
            [0xFF, 0xD8, 0xFF, _] => ImageFormat::JPEG,
            [0x89, 0x50, 0x4E, 0x47] => ImageFormat::PNG,
            _ => ImageFormat::Unknown,
        }
    }

    /// Decodes image bytes to RGB pixels
    fn decode_image_to_rgb(bytes: &[u8]) -> Result<(Vec<RGBPixel>, usize, usize), BellandeError> {
        match Self::detect_image_format(bytes) {
            ImageFormat::JPEG => Self::decode_jpeg(bytes),
            ImageFormat::PNG => Self::decode_png(bytes),
            ImageFormat::Unknown => Err(BellandeError::ImageError(
                "Unknown image format".to_string(),
            )),
        }
    }

    /// JPEG Decoder Implementation
    fn decode_jpeg(bytes: &[u8]) -> Result<(Vec<RGBPixel>, usize, usize), BellandeError> {
        let mut cursor = Cursor::new(bytes);
        let mut marker = [0u8; 2];

        // Verify JPEG signature (0xFFD8)
        cursor
            .read_exact(&mut marker)
            .map_err(|e| BellandeError::ImageError(format!("Invalid JPEG header: {}", e)))?;

        if marker != [0xFF, 0xD8] {
            return Err(BellandeError::ImageError(
                "Not a valid JPEG file".to_string(),
            ));
        }

        let mut width = 0;
        let mut height = 0;
        let mut components = 0;
        let mut quantization_tables = HashMap::new();
        let mut huffman_tables = HashMap::new();

        // Parse JPEG segments
        loop {
            cursor
                .read_exact(&mut marker)
                .map_err(|e| BellandeError::ImageError(format!("Failed to read marker: {}", e)))?;

            if marker[0] != 0xFF {
                return Err(BellandeError::ImageError("Invalid marker".to_string()));
            }

            match marker[1] {
                // Start of Frame (Baseline DCT)
                0xC0 => {
                    let mut segment = [0u8; 8];
                    cursor.read_exact(&mut segment)?;

                    let precision = segment[0];
                    height = u16::from_be_bytes([segment[1], segment[2]]) as usize;
                    width = u16::from_be_bytes([segment[3], segment[4]]) as usize;
                    components = segment[5] as usize;

                    if precision != 8 {
                        return Err(BellandeError::ImageError(
                            "Only 8-bit precision supported".to_string(),
                        ));
                    }

                    // Read component information
                    let mut comp_info = vec![0u8; components * 3];
                    cursor.read_exact(&mut comp_info)?;
                }

                // Define Quantization Table
                0xDB => {
                    let mut length = [0u8; 2];
                    cursor.read_exact(&mut length)?;
                    let length = u16::from_be_bytes(length) as usize - 2;

                    let mut table_data = vec![0u8; length];
                    cursor.read_exact(&mut table_data)?;

                    let precision = (table_data[0] >> 4) & 0x0F;
                    let table_id = table_data[0] & 0x0F;

                    let table_size = if precision == 0 { 64 } else { 128 };
                    let qtable = &table_data[1..=table_size];

                    quantization_tables.insert(table_id, qtable.to_vec());
                }

                // Define Huffman Table
                0xC4 => {
                    let mut length = [0u8; 2];
                    cursor.read_exact(&mut length)?;
                    let length = u16::from_be_bytes(length) as usize - 2;

                    let mut table_data = vec![0u8; length];
                    cursor.read_exact(&mut table_data)?;

                    let table_class = (table_data[0] >> 4) & 0x0F; // DC = 0, AC = 1
                    let table_id = table_data[0] & 0x0F;

                    // Parse Huffman table
                    let mut code_lengths = [0u8; 16];
                    code_lengths.copy_from_slice(&table_data[1..17]);

                    let mut codes = Vec::new();
                    let mut offset = 17;
                    for &length in code_lengths.iter() {
                        for _ in 0..length {
                            codes.push(table_data[offset]);
                            offset += 1;
                        }
                    }

                    huffman_tables.insert((table_class, table_id), codes);
                }

                // Start of Scan
                0xDA => {
                    let mut length = [0u8; 2];
                    cursor.read_exact(&mut length)?;
                    let length = u16::from_be_bytes(length) as usize - 2;

                    let mut scan_data = vec![0u8; length];
                    cursor.read_exact(&mut scan_data)?;

                    // Process compressed data
                    let mut pixels = vec![RGBPixel::new(0, 0, 0); width * height];
                    let mut bit_reader = BitReader::new(&mut cursor);

                    // Process MCUs (Minimum Coded Units)
                    let mcu_width = ((width + 7) / 8) * 8;
                    let mcu_height = ((height + 7) / 8) * 8;

                    for y in (0..mcu_height).step_by(8) {
                        for x in (0..mcu_width).step_by(8) {
                            // Process each component (Y, Cb, Cr)
                            for component in 0..components {
                                let qtable = &quantization_tables[&component];
                                let (dc_table, ac_table) = (
                                    &huffman_tables[&(0, component)],
                                    &huffman_tables[&(1, component)],
                                );

                                // Decode 8x8 block
                                let block = Self::decode_block(
                                    &mut bit_reader,
                                    dc_table,
                                    ac_table,
                                    qtable,
                                )?;

                                // Convert YCbCr to RGB and store in pixels
                                if component == 0 {
                                    // Y component
                                    for by in 0..8 {
                                        for bx in 0..8 {
                                            let px = x + bx;
                                            let py = y + by;
                                            if px < width && py < height {
                                                let idx = py * width + px;
                                                pixels[idx].r = block[by * 8 + bx] as u8;
                                                pixels[idx].g = block[by * 8 + bx] as u8;
                                                pixels[idx].b = block[by * 8 + bx] as u8;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    return Ok((pixels, width, height));
                }

                // End of Image
                0xD9 => break,

                // Skip other markers
                _ => {
                    let mut length = [0u8; 2];
                    cursor.read_exact(&mut length)?;
                    let length = u16::from_be_bytes(length) as usize - 2;
                    cursor.set_position(cursor.position() + length as u64);
                }
            }
        }

        Err(BellandeError::ImageError(
            "Failed to decode JPEG".to_string(),
        ))
    }

    /// Decodes an 8x8 DCT block from JPEG data
    pub fn decode_block(
        bit_reader: &mut BitReader<impl Read>,
        dc_table: &[u8],
        ac_table: &[u8],
        qtable: &[u8],
    ) -> Result<[f32; 64], BellandeError> {
        let mut block = [0f32; 64];
        let mut zz = [0i32; 64];

        // Decode DC coefficient
        let dc_code_length = Self::decode_huffman(bit_reader, dc_table).map_err(|e| {
            BellandeError::ImageError(format!("Failed to decode DC coefficient: {}", e))
        })?;

        if dc_code_length > 0 {
            let dc_value = Self::receive_and_extend(bit_reader, dc_code_length).map_err(|e| {
                BellandeError::ImageError(format!("Failed to read DC value: {}", e))
            })?;
            zz[0] = dc_value;
        }

        // Decode AC coefficients
        let mut k = 1;
        while k < 64 {
            let rs = Self::decode_huffman(bit_reader, ac_table).map_err(|e| {
                BellandeError::ImageError(format!("Failed to decode AC coefficient: {}", e))
            })?;

            let ssss = rs & 0x0F;
            let rrrr = rs >> 4;

            if ssss == 0 {
                if rrrr == 15 {
                    k += 16; // Skip 16 zeros
                    continue;
                }
                break; // End of block
            }

            k += rrrr as usize; // Skip zeros
            if k >= 64 {
                return Err(BellandeError::ImageError(
                    "Invalid AC coefficient index".to_string(),
                ));
            }

            // Read additional bits
            let ac_value = Self::receive_and_extend(bit_reader, ssss).map_err(|e| {
                BellandeError::ImageError(format!("Failed to read AC value: {}", e))
            })?;
            zz[Self::JPEG_NATURAL_ORDER[k]] = ac_value;
            k += 1;
        }

        // Dequantize
        for i in 0..64 {
            zz[i] *= qtable[i] as i32;
        }

        // Inverse DCT
        Self::inverse_dct(&zz, &mut block);

        // Level shift and clamp values
        for val in &mut block {
            *val = (*val + 128.0).clamp(0.0, 255.0);
        }

        Ok(block)
    }

    /// Decodes a Huffman code from the bit stream
    fn decode_huffman(bit_reader: &mut BitReader<impl Read>, table: &[u8]) -> IoResult<u8> {
        let mut code = 0u16;
        let mut code_length = 0u8;
        let mut index = 0usize;

        loop {
            let bit = bit_reader.read_bit()?;
            code = (code << 1) | (if bit { 1 } else { 0 });
            code_length += 1;

            while index < table.len() && table[index] as u8 == code_length {
                if code as u8 == table[index + 1] {
                    return Ok(table[index + 2]);
                }
                index += 3;
            }

            if code_length >= 16 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Invalid Huffman code",
                ));
            }
        }
    }

    /// Receives and extends a value with the given number of bits
    fn receive_and_extend(bit_reader: &mut BitReader<impl Read>, length: u8) -> IoResult<i32> {
        if length == 0 {
            return Ok(0);
        }

        let value = bit_reader.read_bits(length)? as i32;
        let vt = 1 << (length - 1);

        Ok(if value < vt {
            value + (-1 << length) + 1
        } else {
            value
        })
    }

    /// Performs inverse DCT on an 8x8 block
    fn inverse_dct(input: &[i32; 64], output: &mut [f32; 64]) {
        let mut temp = [0f32; 64];

        // 1-D IDCT on rows
        for i in 0..8 {
            let row = &input[i * 8..(i + 1) * 8];
            let mut tmp = [0f32; 8];

            for x in 0..8 {
                let mut sum = 0.0;
                for u in 0..8 {
                    let cu = if u == 0 { 1.0 / f32::sqrt(2.0) } else { 1.0 };
                    sum += cu
                        * row[u] as f32
                        * (std::f32::consts::PI * (2 * x + 1) * u as f32 / 16.0).cos();
                }
                tmp[x] = sum / 2.0;
            }

            for (j, &val) in tmp.iter().enumerate() {
                temp[i * 8 + j] = val;
            }
        }

        // 1-D IDCT on columns
        for i in 0..8 {
            let mut tmp = [0f32; 8];

            for y in 0..8 {
                let mut sum = 0.0;
                for v in 0..8 {
                    let cv = if v == 0 { 1.0 / f32::sqrt(2.0) } else { 1.0 };
                    sum += cv
                        * temp[v * 8 + i]
                        * (std::f32::consts::PI * (2 * y + 1) * v as f32 / 16.0).cos();
                }
                tmp[y] = sum / 2.0;
            }

            for (j, &val) in tmp.iter().enumerate() {
                output[j * 8 + i] = val;
            }
        }
    }

    /// Decodes PNG image bytes
    fn decode_png(bytes: &[u8]) -> Result<(Vec<RGBPixel>, usize, usize), BellandeError> {
        // Basic PNG decoder implementation
        // For now, we'll return a placeholder image
        // TODO: Implement full PNG decoding
        let width = 224;
        let height = 224;
        let pixels = vec![RGBPixel { r: 0, g: 0, b: 0 }; width * height];
        Ok((pixels, width, height))
    }

    /// Converts RGB pixels to tensor
    fn rgb_to_tensor(
        pixels: &[RGBPixel],
        width: usize,
        height: usize,
    ) -> Result<Tensor, BellandeError> {
        if pixels.len() != width * height {
            return Err(BellandeError::ImageError(format!(
                "Invalid pixel buffer size: expected {}, got {}",
                width * height,
                pixels.len()
            )));
        }

        let mut data = Vec::with_capacity(3 * width * height);

        // Convert to CHW format and normalize to [0, 1]
        for channel in 0..3 {
            data.extend(pixels.iter().map(|pixel| {
                let value = match channel {
                    0 => pixel.r,
                    1 => pixel.g,
                    2 => pixel.b,
                    _ => unreachable!(),
                };
                f32::from(value) / 255.0
            }));
        }

        Ok(Tensor::new(
            data,
            vec![1, 3, height, width],
            false,
            Device::CPU,
            DataType::Float32,
        ))
    }

    /// Gets a cached tensor or loads it from disk
    fn get_cached_tensor(&mut self, path: &PathBuf) -> Result<Arc<Tensor>, BellandeError> {
        if let Some(cache) = &mut self.cache {
            if let Some(tensor) = cache.get(path) {
                return Ok(Arc::clone(tensor));
            }

            let bytes = Self::read_image_file(path)?;
            let (pixels, width, height) = Self::decode_image_to_rgb(&bytes)?;
            let tensor = Arc::new(Self::rgb_to_tensor(&pixels, width, height)?);

            // Manage cache size
            if cache.len() >= self.cache_size {
                if let Some(key) = cache.keys().next().cloned() {
                    cache.remove(&key);
                }
            }

            cache.insert(path.clone(), Arc::clone(&tensor));
            Ok(tensor)
        } else {
            let bytes = Self::read_image_file(path)?;
            let (pixels, width, height) = Self::decode_image_to_rgb(&bytes)?;
            let tensor = Arc::new(Self::rgb_to_tensor(&pixels, width, height)?);
            Ok(tensor)
        }
    }

    /// Gets the number of classes in the dataset
    pub fn num_classes(&self) -> usize {
        self.class_to_idx.len()
    }

    /// Gets the mapping of class names to indices
    pub fn get_class_to_idx(&self) -> &HashMap<String, usize> {
        &self.class_to_idx
    }

    /// Gets the path of a sample by index
    pub fn get_sample_path(&self, index: usize) -> Option<&PathBuf> {
        self.samples.get(index).map(|(path, _)| path)
    }

    /// Enables or disables caching
    pub fn set_caching(&mut self, enabled: bool) {
        self.cache = if enabled { Some(HashMap::new()) } else { None };
    }

    /// Clears the cache
    pub fn clear_cache(&mut self) {
        if let Some(cache) = &mut self.cache {
            cache.clear();
        }
    }
}

impl Dataset for ImageFolder {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn num_classes(&self) -> usize {
        self.num_classes()
    }

    fn get(&self, index: usize) -> Result<(Tensor, Tensor), BellandeError> {
        let (path, class_idx) = &self.samples[index];

        // Get input tensor (from cache or load from disk)
        let mut input = match self.cache {
            Some(ref cache) => {
                if let Some(tensor) = cache.get(path) {
                    (*tensor).clone()
                } else {
                    let bytes = Self::read_image_file(path)?;
                    let (pixels, width, height) = Self::decode_image_to_rgb(&bytes)?;
                    Self::rgb_to_tensor(&pixels, width, height)?
                }
            }
            None => {
                let bytes = Self::read_image_file(path)?;
                let (pixels, width, height) = Self::decode_image_to_rgb(&bytes)?;
                Self::rgb_to_tensor(&pixels, width, height)?
            }
        };

        // Create target tensor
        let mut target = Tensor::new(
            vec![*class_idx as f32],
            vec![1],
            false,
            input.device().clone(),
            input.dtype(),
        )?;

        // Apply transforms if available
        if let Some(transform) = &self.transform {
            input = transform.apply(&input)?;
        }

        if let Some(target_transform) = &self.target_transform {
            target = target_transform.apply(&target)?;
        }

        Ok((input, target))
    }
}
