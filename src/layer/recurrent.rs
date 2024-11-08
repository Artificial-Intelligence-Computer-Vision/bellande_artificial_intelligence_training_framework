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

pub struct LSTMCell {
    input_size: usize,
    hidden_size: usize,
    weight_ih: Tensor, // Input-hidden weights
    weight_hh: Tensor, // Hidden-hidden weights
    bias_ih: Option<Tensor>,
    bias_hh: Option<Tensor>,
    cache: Option<LSTMCache>,
}

struct LSTMCache {
    input: Tensor,
    hidden: Tensor,
    cell: Tensor,
    gates: Tensor,
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize, bias: bool) -> Self {
        let weight_ih = Tensor::randn(&[4 * hidden_size, input_size]);
        let weight_hh = Tensor::randn(&[4 * hidden_size, hidden_size]);

        let bias_ih = if bias {
            Some(Tensor::zeros(&[4 * hidden_size]))
        } else {
            None
        };

        let bias_hh = if bias {
            Some(Tensor::zeros(&[4 * hidden_size]))
        } else {
            None
        };

        LSTMCell {
            input_size,
            hidden_size,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            cache: None,
        }
    }

    pub fn forward(
        &mut self,
        input: &Tensor,
        hidden: Option<(Tensor, Tensor)>,
    ) -> Result<(Tensor, Tensor), BellandeError> {
        let batch_size = input.shape[0];

        let (h_prev, c_prev) = match hidden {
            Some((h, c)) => (h, c),
            None => (
                Tensor::zeros(&[batch_size, self.hidden_size]),
                Tensor::zeros(&[batch_size, self.hidden_size]),
            ),
        };

        // Calculate gates
        let gates = self.compute_gates(input, &h_prev)?;

        // Split gates into i, f, g, o
        let chunks = self.split_gates(&gates);
        let (i_gate, f_gate, g_gate, o_gate) = (&chunks[0], &chunks[1], &chunks[2], &chunks[3]);

        // Apply gate operations
        let c_next = f_gate * &c_prev + i_gate * g_gate;
        let h_next = o_gate * c_next.tanh()?;

        // Cache for backward
        self.cache = Some(LSTMCache {
            input: input.clone(),
            hidden: h_prev,
            cell: c_prev,
            gates,
        });

        Ok((h_next, c_next))
    }

    fn compute_gates(&self, input: &Tensor, h_prev: &Tensor) -> Result<Tensor, BellandeError> {
        let ih = input.matmul(&self.weight_ih.transpose()?)?;
        let hh = h_prev.matmul(&self.weight_hh.transpose()?)?;

        let mut gates = ih + hh;

        if let Some(ref bias_ih) = self.bias_ih {
            gates = gates + bias_ih;
        }
        if let Some(ref bias_hh) = self.bias_hh {
            gates = gates + bias_hh;
        }

        Ok(gates)
    }

    fn split_gates(&self, gates: &Tensor) -> Vec<Tensor> {
        let chunk_size = self.hidden_size;
        let mut chunks = Vec::with_capacity(4);

        for i in 0..4 {
            let start = i * chunk_size;
            let end = start + chunk_size;
            chunks.push(gates.narrow(1, start, chunk_size)?);
        }

        chunks
    }
}

pub struct GRUCell {
    input_size: usize,
    hidden_size: usize,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Option<Tensor>,
    bias_hh: Option<Tensor>,
    cache: Option<GRUCache>,
}

struct GRUCache {
    input: Tensor,
    hidden: Tensor,
    gates: Tensor,
}

impl GRUCell {
    pub fn new(input_size: usize, hidden_size: usize, bias: bool) -> Self {
        let weight_ih = Tensor::randn(&[3 * hidden_size, input_size]);
        let weight_hh = Tensor::randn(&[3 * hidden_size, hidden_size]);

        let bias_ih = if bias {
            Some(Tensor::zeros(&[3 * hidden_size]))
        } else {
            None
        };

        let bias_hh = if bias {
            Some(Tensor::zeros(&[3 * hidden_size]))
        } else {
            None
        };

        GRUCell {
            input_size,
            hidden_size,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            cache: None,
        }
    }

    pub fn forward(
        &mut self,
        input: &Tensor,
        hidden: Option<Tensor>,
    ) -> Result<Tensor, BellandeError> {
        let batch_size = input.shape[0];

        let h_prev = match hidden {
            Some(h) => h,
            None => Tensor::zeros(&[batch_size, self.hidden_size]),
        };

        // Calculate gates
        let gates = self.compute_gates(input, &h_prev)?;
        let chunks = self.split_gates(&gates);
        let (r_gate, z_gate, n_gate) = (&chunks[0], &chunks[1], &chunks[2]);

        // Apply GRU update
        let h_next = (z_gate * &h_prev) + ((Tensor::ones(&z_gate.shape) - z_gate) * n_gate);

        // Cache for backward
        self.cache = Some(GRUCache {
            input: input.clone(),
            hidden: h_prev,
            gates,
        });

        Ok(h_next)
    }

    // Similar helper methods as LSTMCell
}
