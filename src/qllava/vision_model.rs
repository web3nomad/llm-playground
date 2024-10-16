use super::text_model::ClipEncoder;
use candle_core::{IndexOp, Result, Shape, Tensor, D};
use candle_nn::{Conv2dConfig, Module};
use candle_transformers::{
    models::clip::{vision_model::ClipVisionConfig, EncoderConfig},
    quantized_var_builder,
};

fn layer_norm(
    size: usize,
    config: candle_nn::LayerNormConfig,
    vb: quantized_var_builder::VarBuilder,
) -> Result<candle_nn::LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let weight = weight.dequantize(vb.device())?;
    if config.affine {
        let bias = vb.get(size, "bias")?;
        let bias = bias.dequantize(vb.device())?;
        Ok(candle_nn::LayerNorm::new(weight, bias, config.eps))
    } else {
        Ok(candle_nn::LayerNorm::new_no_bias(weight, config.eps))
    }
}

// https://github.com/huggingface/transformers/blob/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip/modeling_clip.py#L112
#[derive(Clone, Debug)]
struct ClipVisionEmbeddings {
    patch_embedding: candle_nn::Conv2d,
    position_ids: Tensor,
    class_embedding: Tensor,
    position_embedding: candle_nn::Embedding,
}

impl ClipVisionEmbeddings {
    fn new(vs: quantized_var_builder::VarBuilder, c: &ClipVisionConfig) -> Result<Self> {
        // originally nn.Parameter
        let class_embedding = vs.get(c.embed_dim, "class_embd")?;
        let class_embedding = class_embedding.dequantize(vs.device())?;

        let num_patches = (c.image_size / c.patch_size).pow(2);
        let num_positions = num_patches + 1;
        let position_ids = Tensor::arange(0, num_positions as i64, vs.device())?;

        let conv2dconfig = Conv2dConfig {
            stride: c.patch_size,
            ..Default::default()
        };

        let position_embedding = {
            // candle_nn::embedding(num_positions, c.embed_dim, vs.pp("position_embedding"))?;
            let (in_size, out_size) = (num_positions, c.embed_dim);
            let embeddings = vs.get((in_size, out_size), "position_embd.weight")?;
            let embeddings = embeddings.dequantize(vs.device())?;
            candle_nn::Embedding::new(embeddings, out_size)
        };

        let patch_embedding = {
            // let patch_embedding = candle_nn::conv2d_no_bias(c.num_channels, c.embed_dim, c.patch_size, conv2dconfig, vs.pp("patch_embedding"))?;
            let (in_channels, out_channels, kernel_size) =
                (c.num_channels, c.embed_dim, c.patch_size);
            let ws = vs.get(
                (
                    out_channels,
                    in_channels / conv2dconfig.groups,
                    kernel_size,
                    kernel_size,
                ),
                "patch_embd.weight",
            )?;
            let ws = ws.dequantize(vs.device())?;
            candle_nn::Conv2d::new(ws, None, conv2dconfig)
        };

        Ok(Self {
            patch_embedding,
            position_ids,
            class_embedding,
            position_embedding,
        })
    }
}

impl Module for ClipVisionEmbeddings {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let batch_size = pixel_values.shape().dims();
        let patch_embeds = self
            .patch_embedding
            .forward(pixel_values)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        let shape = Shape::from((batch_size[0], 1, self.class_embedding.dim(D::Minus1)?));
        let class_embeds = self.class_embedding.expand(shape)?;
        let embeddings = Tensor::cat(&[class_embeds, patch_embeds], 1)?;
        let position_embedding = self.position_embedding.forward(&self.position_ids)?;
        embeddings.broadcast_add(&position_embedding)
    }
}

// https://github.com/huggingface/transformers/blob/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip/modeling_clip.py#L743
#[derive(Clone, Debug)]
pub struct ClipVisionTransformer {
    embeddings: ClipVisionEmbeddings,
    encoder: ClipEncoder,
    pre_layer_norm: candle_nn::LayerNorm,
    // final_layer_norm: candle_nn::LayerNorm,
}

impl ClipVisionTransformer {
    pub fn new(vs: quantized_var_builder::VarBuilder, c: &ClipVisionConfig) -> Result<Self> {
        let embeddings = ClipVisionEmbeddings::new(vs.clone(), c)?;

        let pre_layer_norm = layer_norm(c.embed_dim, 1e-5.into(), vs.pp("pre_ln"))?;
        let encoder = ClipEncoder::new(vs.pp("blk"), &EncoderConfig::Vision(c.clone()))?;
        // gguf 里面没有 post layernorm, 这里先用一样的
        // let final_layer_norm = layer_norm(c.embed_dim, 1e-5.into(), vs.pp("post_ln"))?;

        Ok(Self {
            embeddings,
            encoder,
            // final_layer_norm,
            pre_layer_norm,
        })
    }
    // required by LLaVA
    pub fn output_hidden_states(&self, pixel_values: &Tensor) -> Result<Vec<Tensor>> {
        let hidden_states = pixel_values
            .apply(&self.embeddings)?
            .apply(&self.pre_layer_norm)?;
        let mut result = self.encoder.output_hidden_states(&hidden_states, None)?;
        let encoder_outputs = result.last().unwrap();
        let pooled_output = encoder_outputs.i((.., 0, ..))?;
        // result.push(self.final_layer_norm.forward(&pooled_output)?.clone());
        result.push(pooled_output);
        Ok(result)
    }
}

impl Module for ClipVisionTransformer {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let hidden_states = pixel_values
            .apply(&self.embeddings)?
            .apply(&self.pre_layer_norm)?;

        let encoder_outputs = self.encoder.forward(&hidden_states, None)?;
        // https://github.com/huggingface/transformers/blob/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip/modeling_clip.py#L787
        // pooled_output = encoder_outputs[:, 0, :]
        let pooled_output = encoder_outputs.i((.., 0, ..))?;
        // self.final_layer_norm.forward(&pooled_output)
        Ok(pooled_output)
    }
}
