use super::{linear::QLinear, vision_model::ClipVisionTransformer};
use candle_core::{IndexOp, Module, Tensor};
use candle_nn::{
    sequential::{seq, Sequential},
    Activation,
};
use candle_transformers::{models::clip::vision_model::ClipVisionConfig, quantized_var_builder};

pub struct MMProjector {
    pub modules: Sequential,
}

impl MMProjector {
    pub fn new(vb: quantized_var_builder::VarBuilder) -> candle_core::Result<Self> {
        let text_hidden_size: usize = 3072;
        let mm_hidden_size: usize = 1024;
        let modules = {
            let layer = QLinear::load(mm_hidden_size, text_hidden_size, vb.pp("0"))?;
            let mut modules = seq().add(layer);
            let mlp_depth = 2;
            for i in 1..mlp_depth {
                let layer = QLinear::load(
                    text_hidden_size,
                    text_hidden_size,
                    vb.pp(format!("{}", i * 2)),
                )?;
                modules = modules.add(Activation::Gelu).add(layer);
            }
            modules
        };
        Ok(Self { modules })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.modules.forward(x)
    }
}

pub struct ClipVisionTower {
    model: ClipVisionTransformer,
    pub config: ClipVisionConfig,
}

impl ClipVisionTower {
    pub fn new(vb: quantized_var_builder::VarBuilder) -> candle_core::Result<Self> {
        let clip_vision_config = ClipVisionConfig::clip_vit_large_patch14_336();
        let vision_model = ClipVisionTransformer::new(vb, &clip_vision_config)?;
        Ok(Self {
            model: vision_model,
            config: clip_vision_config,
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let select_layer = -2;
        let result = self.model.output_hidden_states(x)?;
        let index = result.len() as isize + select_layer;
        let result = result[index as usize].clone();
        Ok(result.i((.., 1..))?)
    }
}
