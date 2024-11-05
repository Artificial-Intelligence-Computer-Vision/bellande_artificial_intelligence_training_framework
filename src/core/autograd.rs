use std::cell::RefCell;
use std::sync::Arc;

pub trait AutogradFunction: Send + Sync {
    fn forward(&self, input: &[&Tensor]) -> Result<Tensor, BellandeError>;
    fn backward(&self, grad_output: &[f32]) -> Result<(), BellandeError>;
}

pub struct AutogradContext {
    saved_tensors: RefCell<Vec<Tensor>>,
    needs_input_grad: Vec<bool>,
}

impl AutogradContext {
    pub fn save_for_backward(&self, tensor: Tensor) {
        self.saved_tensors.borrow_mut().push(tensor);
    }

    pub fn saved_tensors(&self) -> Vec<Tensor> {
        self.saved_tensors.borrow().clone()
    }
}
