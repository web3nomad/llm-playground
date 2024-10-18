use super::linear::QLinear;
use candle_core::{shape::D, DType, Module, Result, Tensor};
use candle_transformers::{
    models::clip::{text_model::Activation, EncoderConfig},
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

#[derive(Clone, Debug)]
struct ClipAttention {
    k_proj: QLinear,
    v_proj: QLinear,
    q_proj: QLinear,
    out_proj: QLinear,
    head_dim: usize,
    scale: f64,
    num_attention_heads: usize,
}

impl ClipAttention {
    fn new(vs: quantized_var_builder::VarBuilder, c: &EncoderConfig) -> Result<Self> {
        let embed_dim = c.embed_dim();
        let num_attention_heads = c.num_attention_heads();
        let k_proj = QLinear::load(embed_dim, embed_dim, vs.pp("attn_k"))?;
        let v_proj = QLinear::load(embed_dim, embed_dim, vs.pp("attn_v"))?;
        let q_proj = QLinear::load(embed_dim, embed_dim, vs.pp("attn_q"))?;
        let out_proj = QLinear::load(embed_dim, embed_dim, vs.pp("attn_out"))?;
        let head_dim = embed_dim / num_attention_heads;
        let scale = (head_dim as f64).powf(-0.5);

        Ok(ClipAttention {
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            head_dim,
            scale,
            num_attention_heads,
        })
    }

    fn shape(&self, xs: &Tensor, seq_len: usize, bsz: usize) -> Result<Tensor> {
        xs.reshape((bsz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let (bsz, seq_len, embed_dim) = xs.dims3()?;

        let query_states = (self.q_proj.forward(xs)? * self.scale)?;
        let proj_shape = (bsz * self.num_attention_heads, seq_len, self.head_dim);
        let query_states = self
            .shape(&query_states, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let key_states = self
            .shape(&self.k_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let value_states = self
            .shape(&self.v_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let attn_weights = query_states.matmul(&key_states.transpose(1, 2)?)?;

        let src_len = key_states.dim(1)?;

        let attn_weights = if let Some(causal_attention_mask) = causal_attention_mask {
            attn_weights
                .reshape((bsz, self.num_attention_heads, seq_len, src_len))?
                .broadcast_add(causal_attention_mask)?
                .reshape((bsz * self.num_attention_heads, seq_len, src_len))?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

        let attn_output = attn_weights.matmul(&value_states)?.to_dtype(in_dtype)?;
        let attn_output = attn_output
            .reshape((bsz, self.num_attention_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bsz, seq_len, embed_dim))?;
        self.out_proj.forward(&attn_output)
    }
}

#[derive(Clone, Debug)]
struct ClipMlp {
    ffn_down: QLinear,
    ffn_up: QLinear,
    activation: Activation,
}

impl ClipMlp {
    fn new(vs: quantized_var_builder::VarBuilder, c: &EncoderConfig) -> Result<Self> {
        let ffn_down = QLinear::load(c.embed_dim(), c.intermediate_size(), vs.pp("ffn_down"))?;
        let ffn_up = QLinear::load(c.intermediate_size(), c.embed_dim(), vs.pp("ffn_up"))?;
        Ok(ClipMlp {
            ffn_down,
            ffn_up,
            activation: c.activation(),
        })
    }
}

impl ClipMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ffn_down.forward(xs)?;
        self.ffn_up.forward(&self.activation.forward(&xs)?)
    }
}

#[derive(Clone, Debug)]
struct ClipEncoderLayer {
    self_attn: ClipAttention,
    layer_norm1: candle_nn::LayerNorm,
    mlp: ClipMlp,
    layer_norm2: candle_nn::LayerNorm,
}

impl ClipEncoderLayer {
    fn new(vs: quantized_var_builder::VarBuilder, c: &EncoderConfig) -> Result<Self> {
        let self_attn = ClipAttention::new(vs.clone(), c)?;

        let layer_norm1 = layer_norm(c.embed_dim(), 1e-5.into(), vs.pp("ln1"))?;
        let mlp = ClipMlp::new(vs.clone(), c)?;
        let layer_norm2 = layer_norm(c.embed_dim(), 1e-5.into(), vs.pp("ln2"))?;

        Ok(ClipEncoderLayer {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = self.layer_norm1.forward(xs)?;
        let xs = self.self_attn.forward(&xs, causal_attention_mask)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self.layer_norm2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

#[derive(Clone, Debug)]
pub struct ClipEncoder {
    layers: Vec<ClipEncoderLayer>,
}

impl ClipEncoder {
    pub fn new(vs: quantized_var_builder::VarBuilder, c: &EncoderConfig) -> Result<Self> {
        // let vs = vs.pp("layers");
        let mut layers: Vec<ClipEncoderLayer> = Vec::new();

        // let num_hidden_layers = c.num_hidden_layers();
        // gguf 里面 clip.vision.block_count	是 23
        let num_hidden_layers = 23;
        for index in 0..num_hidden_layers {
            let layer = ClipEncoderLayer::new(vs.pp(index.to_string()), c)?;
            layers.push(layer);
        }
        Ok(ClipEncoder { layers })
    }

    pub fn forward(&self, xs: &Tensor, causal_attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, causal_attention_mask)?;
        }
        Ok(xs)
    }
    // required by LLaVA
    pub fn output_hidden_states(
        &self,
        xs: &Tensor,
        causal_attention_mask: Option<&Tensor>,
    ) -> Result<Vec<Tensor>> {
        let mut xs = xs.clone();
        let mut hidden_states = Vec::new();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, causal_attention_mask)?;
            hidden_states.push(xs.clone());
        }
        Ok(hidden_states)
    }
}
