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
use crate::layer::linear::Linear;
use crate::layer::{activation::ReLU, layer_norm::LayerNorm};
use crate::models::sequential::Sequential;

pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    dropout: Dropout,
    cache: Option<AttentionCache>,
}

struct AttentionCache {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_weights: Tensor,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize, dropout: f32) -> Self {
        assert!(
            embed_dim % num_heads == 0,
            "Embedding dimension must be divisible by number of heads"
        );

        let head_dim = embed_dim / num_heads;

        MultiHeadAttention {
            num_heads,
            head_dim,
            q_proj: Linear::new(embed_dim, embed_dim, true),
            k_proj: Linear::new(embed_dim, embed_dim, true),
            v_proj: Linear::new(embed_dim, embed_dim, true),
            out_proj: Linear::new(embed_dim, embed_dim, true),
            dropout: Dropout::new(dropout),
            cache: None,
        }
    }

    pub fn forward(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, BellandeError> {
        let batch_size = query.shape[0];
        let tgt_len = query.shape[1];
        let src_len = key.shape[1];

        // Linear projections
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(key)?;
        let v = self.v_proj.forward(value)?;

        // Reshape for multi-head attention
        let q = q
            .reshape(&[batch_size, tgt_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape(&[batch_size, src_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape(&[batch_size, src_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Calculate attention scores
        let scale = (self.head_dim as f32).sqrt();
        let attention_weights = q.matmul(&k.transpose(2, 3)?)? / scale;

        // Apply mask if provided
        if let Some(mask) = mask {
            attention_weights.masked_fill(mask, f32::NEG_INFINITY)?;
        }

        // Apply softmax and dropout
        let attention_weights = attention_weights.softmax(-1)?;
        let attention_weights = self.dropout.forward(&attention_weights)?;

        // Apply attention to values
        let output = attention_weights.matmul(&v)?;

        // Reshape and project output
        let output = output.transpose(1, 2)?.reshape(&[
            batch_size,
            tgt_len,
            self.num_heads * self.head_dim,
        ])?;
        let output = self.out_proj.forward(&output)?;

        // Cache for backward pass
        self.cache = Some(AttentionCache {
            query: query.clone(),
            key: key.clone(),
            value: value.clone(),
            attention_weights,
        });

        Ok(output)
    }
}

pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    ff_network: Sequential,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: Dropout,
}

impl TransformerEncoderLayer {
    pub fn new(embed_dim: usize, num_heads: usize, ff_dim: usize, dropout: f32) -> Self {
        let ff_network = Sequential::new()
            .add(Linear::new(embed_dim, ff_dim, true))
            .add(ReLU::new())
            .add(Linear::new(ff_dim, embed_dim, true));

        TransformerEncoderLayer {
            self_attn: MultiHeadAttention::new(embed_dim, num_heads, dropout),
            ff_network,
            norm1: LayerNorm::new(embed_dim),
            norm2: LayerNorm::new(embed_dim),
            dropout: Dropout::new(dropout),
        }
    }

    pub fn forward(
        &mut self,
        src: &Tensor,
        src_mask: Option<&Tensor>,
    ) -> Result<Tensor, BellandeError> {
        // Self attention block
        let residual = src.clone();
        let mut output = self.norm1.forward(src)?;
        output = self
            .self_attn
            .forward(&output, &output, &output, src_mask)?;
        output = self.dropout.forward(&output)?;
        output = output + residual;

        // Feed forward block
        let residual = output.clone();
        output = self.norm2.forward(&output)?;
        output = self.ff_network.forward(&output)?;
        output = self.dropout.forward(&output)?;
        output = output + residual;

        Ok(output)
    }
}

pub struct TransformerDecoderLayer {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    ff_network: Sequential,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    dropout: Dropout,
}

impl TransformerDecoderLayer {
    pub fn new(embed_dim: usize, num_heads: usize, ff_dim: usize, dropout: f32) -> Self {
        let ff_network = Sequential::new()
            .add(Linear::new(embed_dim, ff_dim, true))
            .add(ReLU::new())
            .add(Linear::new(ff_dim, embed_dim, true));

        TransformerDecoderLayer {
            self_attn: MultiHeadAttention::new(embed_dim, num_heads, dropout),
            cross_attn: MultiHeadAttention::new(embed_dim, num_heads, dropout),
            ff_network,
            norm1: LayerNorm::new(embed_dim),
            norm2: LayerNorm::new(embed_dim),
            norm3: LayerNorm::new(embed_dim),
            dropout: Dropout::new(dropout),
        }
    }

    pub fn forward(
        &mut self,
        tgt: &Tensor,
        memory: &Tensor,
        tgt_mask: Option<&Tensor>,
        memory_mask: Option<&Tensor>,
    ) -> Result<Tensor, BellandeError> {
        // Self attention block
        let residual = tgt.clone();
        let mut output = self.norm1.forward(tgt)?;
        output = self
            .self_attn
            .forward(&output, &output, &output, tgt_mask)?;
        output = self.dropout.forward(&output)?;
        output = output + residual;

        // Cross attention block
        let residual = output.clone();
        output = self.norm2.forward(&output)?;
        output = self
            .cross_attn
            .forward(&output, memory, memory, memory_mask)?;
        output = self.dropout.forward(&output)?;
        output = output + residual;

        // Feed forward block
        let residual = output.clone();
        output = self.norm3.forward(&output)?;
        output = self.ff_network.forward(&output)?;
        output = self.dropout.forward(&output)?;
        output = output + residual;

        Ok(output)
    }
}
