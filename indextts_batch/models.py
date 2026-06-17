import importlib.util
import logging
import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TypicalLogitsWarper,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.model_v2 import (
    ConditioningEncoder,
    LearnedPositionEmbeddings,
    MelEncoder,
)
from indextts.gpt.perceiver import PerceiverResampler
from indextts.gpt.transformers_gpt2 import GPT2PreTrainedModel

logger = logging.getLogger(__name__)


# ──────── 基础辅助层与空位置嵌入 ─────────────────────────
class NullPositionEmbeddings(nn.Module):
    """提供全零值的位置嵌入，在不需要额外位置编码的层中作为占位"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, range, *args, **kwargs):
        """返回与输入形状一致的全零位置编码"""
        return torch.zeros((range.shape[0], range.shape[1], self.dim), device=range.device)


# ────────  Accel 采样器 ──────────────────────────────
class BatchAccelSampler:
    """替换 AccelInferenceEngine 的默认采样器，补全各采样器对齐 HuggingFace"""

    def __init__(self, accel_engine):
        self.accel_engine = accel_engine
        self.top_k: int = 30
        self.top_p: float = 0.8
        self.repetition_penalty: float = 10.0
        self.typical_sampling: bool = True
        self.typical_mass: float = 0.9

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 1. repetition_penalty：修正负数 logits 的数学逻辑
        if self.repetition_penalty > 1.0:
            sequences = getattr(self.accel_engine, "current_sequences", None)
            if sequences is not None:
                penalty = torch.ones_like(logits)
                for i, seq in enumerate(sequences):
                    if seq.token_ids:
                        idx = torch.tensor(seq.token_ids, device=logits.device, dtype=torch.long)
                        penalty[i].index_fill_(0, idx, self.repetition_penalty)
                # HuggingFace 逻辑：小于0时乘惩罚项（让负得更多），大于0时除惩罚项
                logits = torch.where(logits < 0, logits * penalty, logits / penalty)

        # 2. 温度缩放
        temperatures = temperatures.clamp(min=1e-8)
        greedy_mask = temperatures < 1e-5
        temp_for_scaling = torch.where(greedy_mask, 1.0, temperatures)
        scaled_logits = logits / temp_for_scaling.unsqueeze(-1)

        # 3. top-k 截断
        if self.top_k > 0:
            top_k = min(int(self.top_k), scaled_logits.size(-1))
            values, _ = torch.topk(scaled_logits, top_k, dim=-1)
            threshold = values[..., -1, None]
            scaled_logits = torch.where(scaled_logits < threshold, float("-inf"), scaled_logits)

        # 4. Typical Sampling 典型采样
        if self.typical_sampling and 0.0 < self.typical_mass < 1.0:
            normalized_logits = F.log_softmax(scaled_logits, dim=-1)
            probs_typical = torch.exp(normalized_logits)
            entropy = -(normalized_logits * probs_typical).nansum(dim=-1, keepdim=True)
            shifted_scores = torch.abs(-normalized_logits - entropy)

            sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False, dim=-1)
            sorted_probs = probs_typical.gather(-1, sorted_indices)
            cumulative_probs = sorted_probs.cumsum(dim=-1)

            last_ind = (cumulative_probs < self.typical_mass).sum(dim=-1)
            last_ind.clamp_(max=scaled_logits.size(-1) - 1)

            sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))

            scaled_logits = scaled_logits.scatter(
                -1,
                sorted_indices,
                scaled_logits.gather(-1, sorted_indices).masked_fill(sorted_indices_to_remove, float("-inf")),
            )

        # 5. top-p 核采样
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits.float(), dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.top_p
            # 向右平移一位，确保最先越过 top_p 的 token 会被保留
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            sorted_logits[sorted_indices_to_remove] = float("-inf")
            scaled_logits = torch.scatter(scaled_logits, -1, sorted_indices, sorted_logits)

        # 6. 多项式采样
        probs = torch.softmax(scaled_logits.float(), dim=-1)

        # 防止 float16 极端下溢导致概率全为0
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(probs_sum == 0, torch.ones_like(probs) / probs.size(-1), probs)

        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        greedy_tokens = logits.argmax(dim=-1)

        return torch.where(greedy_mask, greedy_tokens, sampled_tokens)


# ──────── GPT2 推理适配与缓存接驳 ─────────────────────────
class GPT2InferenceModel(GPT2PreTrainedModel):
    """GPT2 模型外层包装器"""

    def __init__(self, config, gpt, text_pos_emb, embeddings, norm, linear, kv_cache=False):
        super().__init__(config)
        self.transformer = gpt
        self.text_pos_embedding: Any = text_pos_emb
        self.embeddings: Any = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache
        self.model_parallel = False
        self.device_map = None
        self.cached_mel_emb: Any = None

    def store_mel_emb(self, mel_emb):
        self.cached_mel_emb = mel_emb

    def clear_mel_emb(self):
        self.cached_mel_emb = None

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, *args, **kwargs):
        token_type_ids = kwargs.get("token_type_ids")
        if not self.kv_cache:
            past_key_values = None

        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids")

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if self.cached_mel_emb is None:
            raise ValueError("cached_mel_emb 为 None")
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask 为 None")
        if inputs_embeds is not None or labels is not None:
            raise ValueError("inputs_embeds and labels 为 None")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # L_total = 音频特征(34) + 文本长度(L) + 文本起止符(2)
        mel_len = self.cached_mel_emb.shape[1]  # [b, L_total]

        # 获取预填充时的 embedding
        if input_ids.shape[1] != 1:  # [b, L_total + 1(GPT起始符)]
            text_inputs = input_ids[:, mel_len:]  # [b, 1] GPT起始符
            text_emb = self.embeddings(text_inputs)
            text_emb = text_emb + self.text_pos_embedding(text_emb)  # 起始符8193的embedding加上位置编码 [b, 1, 1280]

            if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
                mel_emb = self.cached_mel_emb.repeat_interleave(text_emb.shape[0] // self.cached_mel_emb.shape[0], 0)
            else:
                mel_emb = self.cached_mel_emb  # [b, L_total, 1280]
            emb = torch.cat([mel_emb, text_emb], dim=1)  # [b, L_total + 1, 1280]
        # 获取自回归时的 embedding
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.text_pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] - mel_len, attention_mask.device
            )  # [b, 1, 1280]

        # GPT 自回归生成
        transformer_outputs: Any = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]  # GPT 最后一个隐藏层作为输出

        # 兜底操作，设备对齐
        if self.model_parallel:
            if torch.backends.mps.is_available():
                self.to(self.transformer.first_device)
            else:
                torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # GPT输出映射到词表维度 [b, 1, 1280] -> [b, 1, 8194]
        lm_logits = self.lm_head(hidden_states)

        # 输出打包
        if not return_dict:
            return (lm_logits, *tuple(transformer_outputs)[1:])

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )


def build_hf_gpt_transformer(layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing):
    gpt_config = GPT2Config(
        vocab_size=256,
        n_positions=max_mel_seq_len + max_text_seq_len,
        n_ctx=max_mel_seq_len + max_text_seq_len,
        n_embd=model_dim,
        n_layer=layers,
        n_head=heads,
        use_cache=not checkpointing,
    )
    gpt = GPT2Model(gpt_config)
    if checkpointing:
        gpt.gradient_checkpointing_enable()
    if hasattr(gpt, "wpe"):
        delattr(gpt, "wpe")
    gpt.wpe = NullPositionEmbeddings(model_dim)  # type: ignore[assignment]
    del gpt.wte
    return (
        gpt,
        LearnedPositionEmbeddings(max_mel_seq_len, model_dim),
        LearnedPositionEmbeddings(max_text_seq_len, model_dim),
        None,
        None,
    )


# ──────── 统一声纹/情感自回归多通道骨干模型 ─────────────────────────
class UnifiedVoice(nn.Module):
    def __init__(
        self,
        layers=8,
        model_dim=512,
        heads=8,
        max_text_tokens=120,
        max_mel_tokens=250,
        max_conditioning_inputs=1,
        mel_length_compression=1024,
        number_text_tokens=256,
        start_text_token=0,
        stop_text_token=1,
        number_mel_codes=8194,
        start_mel_token=8192,
        stop_mel_token=8193,
        train_solo_embeddings=False,
        use_mel_codes_as_input=True,
        checkpointing=True,
        types=1,
        condition_num_latent=32,
        condition_type="perceiver",
        condition_module=None,
        emo_condition_module=None,
        use_accel=False,
    ):
        super().__init__()
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.condition_type = condition_type
        self.cond_num = condition_num_latent
        self.cond_mask_pad = nn.ConstantPad1d((self.cond_num, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)

        if condition_type == "perceiver":
            self.conditioning_encoder = ConditioningEncoder(1024, model_dim, num_attn_heads=heads)
            self.perceiver_encoder = PerceiverResampler(model_dim, dim_context=model_dim, num_latents=self.cond_num)
        elif condition_type == "conformer_perceiver" or condition_type == "conformer_encoder":
            if condition_module is None:
                raise ValueError(f"condition_module must be provided when condition_type is {condition_type}")
            self.conditioning_encoder = ConformerEncoder(
                input_size=1024,
                output_size=condition_module["output_size"],
                linear_units=condition_module["linear_units"],
                attention_heads=condition_module["attention_heads"],
                num_blocks=condition_module["num_blocks"],
                input_layer=condition_module["input_layer"],
            )
            if condition_type == "conformer_perceiver":
                self.perceiver_encoder = PerceiverResampler(
                    model_dim,
                    dim_context=condition_module["output_size"],
                    ff_mult=condition_module["perceiver_mult"],
                    heads=condition_module["attention_heads"],
                    num_latents=self.cond_num,
                )
        else:
            self.conditioning_encoder = ConditioningEncoder(1024, model_dim, num_attn_heads=heads, mean=True)

        if emo_condition_module is None:
            raise ValueError("emo_condition_module must be provided")

        self.emo_conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=emo_condition_module["output_size"],
            linear_units=emo_condition_module["linear_units"],
            attention_heads=emo_condition_module["attention_heads"],
            num_blocks=emo_condition_module["num_blocks"],
            input_layer=emo_condition_module["input_layer"],
        )
        self.emo_perceiver_encoder = PerceiverResampler(
            1024,
            dim_context=emo_condition_module["output_size"],
            ff_mult=emo_condition_module["perceiver_mult"],
            heads=emo_condition_module["attention_heads"],
            num_latents=1,
        )

        self.text_embedding = nn.Embedding(self.number_text_tokens * types + 1, model_dim)
        self.emo_layer = nn.Linear(model_dim, model_dim)
        self.emovec_layer = nn.Linear(1024, model_dim)

        if use_mel_codes_as_input:
            self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        else:
            self.mel_embedding = MelEncoder(model_dim, resblocks_per_reduction=1)

        (
            self.gpt,
            self.mel_pos_embedding,
            self.text_pos_embedding,
            self.mel_layer_pos_embedding,
            self.text_layer_pos_embedding,
        ) = build_hf_gpt_transformer(
            layers,
            model_dim,
            heads,
            self.max_mel_tokens + 2 + self.max_conditioning_inputs,
            self.max_text_tokens + 2,
            checkpointing,
        )

        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02, requires_grad=True)
            self.text_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02, requires_grad=True)
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        self.speed_emb = nn.Embedding(2, model_dim)
        self.speed_emb.weight.data.normal_(mean=0.0, std=0.0)

        if self.text_embedding.weight is not None:
            self.text_embedding.weight.data.normal_(mean=0.0, std=0.02)

        if use_mel_codes_as_input:
            if isinstance(self.mel_embedding, nn.Embedding) and self.mel_embedding.weight is not None:
                self.mel_embedding.weight.data.normal_(mean=0.0, std=0.02)

        self.use_accel = use_accel
        self.accel_engine = None
        self.inference_model: GPT2InferenceModel | None = None

    def post_init_gpt2_config(self, use_deepspeed=False, kv_cache=False, half=False):
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        gpt_config = GPT2Config(
            vocab_size=self.number_mel_codes,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            use_cache=True,
        )

        self.inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )

        if self.use_accel and torch.cuda.is_available():
            if importlib.util.find_spec("flash_attn") is None:
                logger.warning("[GPT自回归解码] flash_attn 未安装，Accel 加速不可用")
                self.use_accel = False

        if self.use_accel and torch.cuda.is_available():
            from indextts.accel import AccelInferenceEngine, GPT2AccelModel

            accel_gpt = GPT2AccelModel(gpt_config)
            accel_gpt.load_state_dict(self.gpt.state_dict(), strict=False)

            target_device = self.gpt.device if hasattr(self.gpt, "device") else next(self.gpt.parameters()).device

            if half:
                accel_gpt = accel_gpt.half().to(target_device)  # type: ignore[call-arg]
            else:
                accel_gpt = accel_gpt.to(target_device)  # type: ignore[call-arg]
            accel_gpt.eval()

            lm_head_with_norm = nn.Sequential(self.final_norm, self.mel_head)
            self.accel_engine = AccelInferenceEngine(
                model=accel_gpt,
                lm_head=lm_head_with_norm,
                num_layers=self.layers,
                num_heads=self.heads,
                head_dim=self.model_dim // self.heads,
                block_size=256,
                num_blocks=64,
                use_cuda_graph=True,
            )
            self.accel_engine.sampler = BatchAccelSampler(self.accel_engine)  # type: ignore[assignment]
            _orig_prepare_decode = self.accel_engine._prepare_decode

            def _patched_prepare_decode(self, requests):
                input_ids, positions = _orig_prepare_decode(requests)
                tts_mode = getattr(self, "_tts_mode", False)
                if tts_mode:
                    # pos = len(req) - num_prompt_tokens (生成步数)
                    fixed = torch.tensor(
                        [len(r) - r.num_prompt_tokens for r in requests],
                        dtype=positions.dtype,
                        device=positions.device,
                    )
                    positions.copy_(fixed)
                return input_ids, positions

            self.accel_engine._prepare_decode = types.MethodType(_patched_prepare_decode, self.accel_engine)
            logger.info("[GPT自回归解码] Accel 推理加速引擎已启用")

        # DeepSpeed 推理加速
        if use_deepspeed and torch.cuda.is_available():
            if importlib.util.find_spec("deepspeed") is None:
                logger.warning("[GPT自回归解码] deepspeed 未安装，跳过 DeepSpeed 加速")
                self.inference_model = self.inference_model.eval()
            else:
                import deepspeed  # type: ignore[import]

                ds_dtype = torch.float16 if half else torch.float32
                try:
                    self.ds_engine = deepspeed.init_inference(
                        model=self.inference_model,
                        mp_size=1,
                        replace_with_kernel_inject=True,
                        dtype=ds_dtype,
                    )
                    self.inference_model = self.ds_engine  # type: ignore[assignment]
                    logger.info("[GPT自回归解码] DeepSpeed 推理加速已启用")
                except Exception as e:
                    logger.warning(f"[GPT自回归解码] DeepSpeed 加载失败，回退到普通推理: {e}")
                    self.inference_model = self.inference_model.eval()  # type: ignore[union-attr]
        else:
            self.inference_model = self.inference_model.eval()

        self.gpt.wte = self.mel_embedding

    def get_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        """
        功能简述: 将输入的参考音频提取高频声纹特征，并经由 Perceiver 重采样映射至自回归可用的声学上下文序列中。
        参数:
            - speech_conditioning_input (torch.Tensor): 输入的高维特征张量。
            - cond_mel_lengths (torch.Tensor | None): 各个批次参考音频的有效序列长度，默认值为 None。
        返回值:
            - torch.Tensor: 由重采样器约束输出的统一维数声学条件序列表示。
        """
        conds: Any = None
        if self.condition_type == "perceiver":
            if speech_conditioning_input.ndim == 4:
                speech_conditioning_input = speech_conditioning_input.squeeze(1)
            speech_conditioning_input = self.conditioning_encoder(speech_conditioning_input)
            conds = self.perceiver_encoder(speech_conditioning_input.transpose(1, 2))
        elif self.condition_type == "conformer_perceiver":
            # Conformer 编码器：输入为 (b, 1024, T) -> transpose -> (b, T, 1024) -> encode -> (b, T/4, 512)
            speech_conditioning_input, mask = self.conditioning_encoder(
                speech_conditioning_input.transpose(1, 2), cond_mel_lengths
            )
            # mask squeeze(1) 变成 [b, s]，然后 cond_mask_pad 在左侧添加 32 个 True，变成 [b, s+32]
            conds_mask = self.cond_mask_pad(mask.squeeze(1))
            # perceiver_encoder 将变长的 [b, s, 512] 转换为固定长度的 [b, 32, 1280]
            # 这 32 个 Token 包含了说话人的声学特征精华
            conds = self.perceiver_encoder(speech_conditioning_input, conds_mask)
        return conds

    def get_emo_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        # 时域下采样 (Downsampling)
        # 情感 Conformer 编码器：输入为 (b, 1024, T) -> transpose -> (b, T, 1024) -> encode -> (b, s, 512), 其中 s 是下采样后的长度 (T/4)
        speech_conditioning_input, mask = self.emo_conditioning_encoder(
            speech_conditioning_input.transpose(1, 2), cond_mel_lengths
        )
        # 掩码准备
        # mask.squeeze(1) 将形状从 (b, 1, s) 变为 (b, s)
        # emo_cond_mask_pad 在左侧添加 1 个 True（因为情感重采样只对应 1 个 Latent Token）->  (b, s + 1)
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        # 特征重采样 (Resampling)
        # emo_perceiver_encoder 使用 1 个可学习的隐向量去查询下采样后的 s 帧特征
        # 将变长序列压缩为固定长度的 1 个情感 Token，输出形状为 (b, 1, d)
        conds = self.emo_perceiver_encoder(speech_conditioning_input, conds_mask)
        # 压缩掉数量维度，返回最终的全局情感向量，形状为 (b, d)，即 (b, 1024)
        return conds.squeeze(1)

    def get_emovec(self, emo_speech_conditioning_latent, emo_cond_lengths):
        emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1, 2), emo_cond_lengths)
        emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
        return self.emo_layer(emo_vec_syn)

    def merge_emovec(
        self,
        speech_conditioning_latent,
        emo_speech_conditioning_latent,
        cond_lengths,
        emo_cond_lengths,
        alpha=1.0,
    ):
        emo_vec = self.get_emovec(emo_speech_conditioning_latent, emo_cond_lengths)
        base_vec = self.get_emovec(speech_conditioning_latent, cond_lengths)

        return base_vec + alpha * (emo_vec - base_vec)

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, mel_lengths):
        if mel_lengths is None:
            raise ValueError("mel_lengths must not be None")
        for b in range(len(mel_lengths)):
            actual_end = mel_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def set_text_padding(self, text_input_tokens, text_lengths):
        if text_lengths is None:
            raise ValueError("text_lengths must not be None")
        for b in range(len(text_lengths)):
            actual_end = text_lengths[b]
            if actual_end < text_input_tokens.shape[-1]:
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens

    def get_logits(
        self,
        speech_conditioning_inputs,
        first_inputs,
        first_head,
        second_inputs=None,
        second_head=None,
        get_attns=False,
        return_latent=False,
        attention_mask=None,
    ):
        # speech_conditioning_inputs：音频和控制前缀 [b, 34, 1280]
        # first_inputs: 文本嵌入 [b, L_text, 1280]
        # second_inputs: MEL code嵌入 [b, L_mel, 1280]，GPT自回归生成的codes
        # emb: [b, 34+L_text+L_mel, 1280]
        if second_inputs is not None:
            emb = torch.cat([speech_conditioning_inputs, first_inputs, second_inputs], dim=1)
        else:
            emb = torch.cat([speech_conditioning_inputs, first_inputs], dim=1)

        # 输入 GPT 骨干网络，得到最后一层隐状态
        gpt_out: Any = self.gpt(
            inputs_embeds=emb,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=get_attns,
        )
        if get_attns:
            return gpt_out.attentions

        # 裁剪控制前缀（前34个token）
        offset = speech_conditioning_inputs.shape[1]
        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)

        # 推理使用：返回最后一层隐状态
        if return_latent:
            if second_inputs is None:
                raise ValueError("second_inputs must be provided when return_latent is True")
            return enc[:, : first_inputs.shape[1]], enc[:, -second_inputs.shape[1] :]

        # 训练使用，忽略
        first_logits = enc[:, : first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            if second_head is None:
                raise ValueError("second_head must be provided when second_inputs is provided")
            second_logits = enc[:, -second_inputs.shape[1] :]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        return first_logits

    def forward(
        self,
        speech_conditioning_latent,
        text_inputs,
        text_lengths,
        mel_codes,
        mel_codes_lengths,
        emo_speech_conditioning_latent,
        cond_mel_lengths=None,
        emo_cond_mel_lengths=None,
        emo_vec=None,
        use_speed=None,
        do_spk_cond=False,
    ):
        if text_lengths is None or mel_codes_lengths is None:
            raise ValueError("text_lengths and mel_codes_lengths must not be None")

        if use_speed is None:
            use_speed = torch.zeros(text_inputs.shape[0], device=text_inputs.device, dtype=torch.long)

        if do_spk_cond:
            speech_conditioning_latent = self.get_conditioning(
                speech_conditioning_latent.transpose(1, 2), cond_mel_lengths
            )
        else:
            speech_conditioning_latent = speech_conditioning_latent

        if emo_vec is None:
            emo_vec_syn_ori = self.get_emo_conditioning(
                emo_speech_conditioning_latent.transpose(1, 2), emo_cond_mel_lengths
            )
            emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
            emo_vec = self.emo_layer(emo_vec_syn)

        if emo_vec is None:
            raise RuntimeError("emo_vec generation failed")

        text_inputs = self.set_text_padding(text_inputs, text_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)

        mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)

        # 确定当前的推理 Batch
        B = text_inputs.shape[0]
        if use_speed.shape[0] != B:
            use_speed = use_speed.new_zeros(B)

        # 此处使用 .long() 显式转型以防御因浮点型 zeros_like 导致 nn.Embedding 出错的潜在 Bug
        duration_emb = self.speed_emb(torch.zeros_like(use_speed).long())
        duration_emb_half = self.speed_emb(torch.ones_like(use_speed).long())
        if speech_conditioning_latent.shape[0] != B:
            speech_conditioning_latent = speech_conditioning_latent.expand(B, -1, -1)

        # 广播 emo_vec 到 Batch 维度
        emo_vec_expanded = emo_vec.unsqueeze(1) if emo_vec.ndim == 2 else emo_vec

        if emo_vec_expanded.shape[0] != B:
            emo_vec_expanded = emo_vec_expanded.expand(B, -1, -1)

        # 音频语义嵌入：[b, 34, 1280]
        conds = torch.cat(
            (
                speech_conditioning_latent + emo_vec_expanded,
                duration_emb_half.unsqueeze(1),
                duration_emb.unsqueeze(1),
            ),
            1,
        )

        # 文本转换为嵌入 + 位置编码
        text_inputs, _text_targets = self.build_aligned_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token
        )
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        # 生成的梅尔代码转换为嵌入 + 位置编码
        mel_codes, _mel_targets = self.build_aligned_inputs_and_targets(
            mel_codes, self.start_mel_token, self.stop_mel_token
        )
        mel_emb = self.mel_embedding(mel_codes)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)

        # 构建注意力掩码：屏蔽 text padding 位置，防止后续 mel 位置 attend 到无效文本
        conds_len = conds.shape[1]
        text_seq_len = text_inputs.shape[1]
        mel_seq_len = mel_emb.shape[1]
        total_len = conds_len + text_seq_len + mel_seq_len
        attention_mask = torch.ones(B, total_len, device=text_inputs.device, dtype=torch.bool)
        for b in range(B):
            text_valid_end = conds_len + text_lengths[b] + 2
            attention_mask[b, text_valid_end : conds_len + text_seq_len] = False

        # 进行一次前向推理，获取最后一层的隐状态
        _text_logits, mel_logits = self.get_logits(
            conds,
            text_emb,
            self.text_head,
            mel_emb,
            self.mel_head,
            get_attns=False,
            return_latent=True,
            attention_mask=attention_mask,
        )
        return mel_logits[:, :-2]  # 尽管名为 logits，但实际并非 logits。裁剪掉本次前向添加的两个 token

    def prepare_gpt_inputs(
        self,
        conditional_latents: torch.Tensor,
        text_inputs: torch.Tensor,
    ):
        b, L = text_inputs.shape[:2]
        device = text_inputs.device

        single_cond = conditional_latents.ndim == 3 and conditional_latents.shape[0] == 1
        if not single_cond:
            if conditional_latents.shape[0] != b:
                raise ValueError(f"batch size mismatch: {conditional_latents.shape[0]} vs {b}")

        batched_mel_emb = []
        attention_masks = []

        # 目标总长度 = 音频特征(34) + 文本最大长度(L) + 文本起止符(2)
        target_len = conditional_latents.shape[1] + L + 2

        for i in range(b):
            # 1. 过滤填充位，得到真实的变长文本
            valid_mask = (text_inputs[i] != self.stop_text_token) & (text_inputs[i] != self.start_text_token)
            text_input = text_inputs[i][valid_mask]

            # 2. 文本打标
            text_input = F.pad(text_input, (1, 0), value=self.start_text_token)
            text_input = F.pad(text_input, (0, 1), value=self.stop_text_token)

            # 3. 升维获取文本嵌入及位置编码，再降回一维
            text_input_2d = text_input.unsqueeze(0)
            text_emb = self.text_embedding(text_input_2d) + self.text_pos_embedding(text_input_2d)
            text_emb = text_emb.squeeze(0)  # [真实长度 L_true, 1280]

            # 4. 序列拼接
            conds_text_emb = [
                conditional_latents.squeeze(0) if single_cond else conditional_latents[i],
                text_emb,
            ]

            # 5. 左侧 Padding 处理
            attention_mask = torch.ones(target_len + 1, dtype=torch.long, device=device)
            padding = L + 2 - text_input.size(-1)
            if padding > 0:
                pad = torch.zeros(
                    (padding, conditional_latents.size(-1)),
                    dtype=text_emb.dtype,
                    device=device,
                )
                conds_text_emb.insert(0, pad)
                attention_mask[:padding] = 0  # Padding 处无注意力

            mel_emb = torch.cat(conds_text_emb)
            batched_mel_emb.append(mel_emb)
            attention_masks.append(attention_mask)

        batched_mel_emb = torch.stack(batched_mel_emb, dim=0)
        attention_mask = torch.stack(attention_masks, dim=0)

        fake_inputs = torch.ones(
            (batched_mel_emb.shape[0], batched_mel_emb.shape[1] + 1),
            dtype=torch.long,
            device=device,
        )
        fake_inputs[:, -1] = self.start_mel_token
        return fake_inputs, batched_mel_emb, attention_mask

    def inference_speech(
        self,
        spk_cond_emb=None,
        text_inputs=None,
        emo_cond_emb=None,
        cond_lengths=None,
        emo_cond_lengths=None,
        emovec=None,
        speech_conditioning_latent=None,
        use_speed=False,
        input_tokens=None,
        num_return_sequences=1,
        max_generate_length=None,
        typical_sampling=False,
        typical_mass=0.9,
        **hf_generate_kwargs,
    ):
        assert self.inference_model is not None
        if speech_conditioning_latent is None:
            if spk_cond_emb is None:
                raise ValueError("必须提供 spk_cond_emb 或 speech_conditioning_latent")
            if spk_cond_emb.ndim == 2:
                spk_cond_emb = spk_cond_emb.unsqueeze(0)
            if cond_lengths is None:
                cond_lengths = torch.tensor(
                    [spk_cond_emb.shape[-1]] * spk_cond_emb.shape[0],
                    device=spk_cond_emb.device,
                )
            speech_conditioning_latent = self.get_conditioning(spk_cond_emb.transpose(1, 2), cond_lengths)

        if text_inputs is None or emovec is None:
            raise ValueError("text_inputs and emovec must not be None")

        tmp = torch.zeros(text_inputs.size(0), dtype=torch.long, device=text_inputs.device)
        duration_emb = self.speed_emb(tmp)
        duration_emb_half = self.speed_emb(torch.ones_like(tmp))

        emovec_expanded = emovec.unsqueeze(1) if emovec.ndim == 2 else emovec
        conds_latent = speech_conditioning_latent + emovec_expanded
        # 自适应广播
        B = text_inputs.size(0)
        if conds_latent.shape[0] != B:
            conds_latent = conds_latent.expand(B, -1, -1)

        # 情感注入，将情感向量广播到说话人向量上
        conds_latent = torch.cat(
            (
                conds_latent,
                duration_emb_half.unsqueeze(1),
                duration_emb.unsqueeze(1),
            ),
            1,
        )  # [1, 34, 1280]

        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(conds_latent, text_inputs)
        if hasattr(self.inference_model, "store_mel_emb"):
            self.inference_model.store_mel_emb(inputs_embeds)

        trunc_index = input_ids.shape[1]
        max_length = (
            (trunc_index + self.max_mel_tokens - 1)
            if max_generate_length is None
            else trunc_index + max_generate_length
        )

        # Accel 引擎推理路径（flash attention + CUDA graph）
        if self.accel_engine is not None and num_return_sequences == 1:
            if hf_generate_kwargs.get("num_beams", 1) > 1:
                logger.warning("[GPT推理] Accel 路径不支持束搜索 (num_beams)，已静默回退到贪婪采样")
            if hf_generate_kwargs.get("length_penalty", 0.0) != 0.0:
                logger.warning("[GPT推理] Accel 路径不支持长度惩罚 (length_penalty)，参数已忽略")
            # 同步采样参数，与非 accel 路径保持一致
            _sampler = self.accel_engine.sampler  # type: ignore[attr-defined]
            _sampler.top_k = hf_generate_kwargs.get("top_k", 30)
            _sampler.top_p = hf_generate_kwargs.get("top_p", 0.8)
            _sampler.repetition_penalty = hf_generate_kwargs.get("repetition_penalty", 10.0)
            _sampler.typical_sampling = typical_sampling  # type: ignore[arg-type]
            if typical_sampling and not (typical_mass > 0.0 and typical_mass < 1.0):
                raise ValueError(f"`typical_mass` must be a float > 0 and < 1, but got {typical_mass}")
            _sampler.typical_mass = typical_mass  # type: ignore[arg-type]
            output = self.accel_engine.generate(
                input_ids,
                max_new_tokens=max_length - trunc_index,
                attention_mask=attention_mask,
                temperature=hf_generate_kwargs.get("temperature", 0.8),
                top_p=hf_generate_kwargs.get("top_p", 0.8),
                top_k=hf_generate_kwargs.get("top_k", 30),
                stop_tokens=[self.stop_mel_token],
                tts_embeddings=inputs_embeds,
                tts_mel_embedding=self.inference_model.embeddings,
                tts_text_pos_embedding=self.inference_model.text_pos_embedding,
            )
        else:
            logits_processor = LogitsProcessorList()
            if typical_sampling:
                if not (typical_mass > 0.0 and typical_mass < 1.0):
                    raise ValueError(f"`typical_mass` must be a float > 0 and < 1, but got {typical_mass}")
                min_tokens_to_keep = 2 if hf_generate_kwargs.get("num_beams", 1) > 1 else 1
                logits_processor.append(TypicalLogitsWarper(mass=typical_mass, min_tokens_to_keep=min_tokens_to_keep))
            output = self.inference_model.generate(
                input_ids,
                bos_token_id=self.start_mel_token,
                pad_token_id=self.stop_mel_token,
                eos_token_id=self.stop_mel_token,
                attention_mask=attention_mask,
                max_length=max_length,
                logits_processor=logits_processor,
                num_return_sequences=num_return_sequences,
                **hf_generate_kwargs,
            )

        if isinstance(output, torch.Tensor):
            return output[:, trunc_index:], speech_conditioning_latent
        return output.sequences[:, trunc_index:], speech_conditioning_latent


# ──────── 剥离出的独立轻量化情感提取器 ─────────────────────────
class EmotionExtractor(nn.Module):
    """
    完全剥离自 UnifiedVoice 的轻量化情感提取器
    """

    def __init__(self, cfg_gpt):
        super().__init__()
        self.emo_conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=cfg_gpt.emo_condition_module["output_size"],
            linear_units=cfg_gpt.emo_condition_module["linear_units"],
            attention_heads=cfg_gpt.emo_condition_module["attention_heads"],
            num_blocks=cfg_gpt.emo_condition_module["num_blocks"],
            input_layer=cfg_gpt.emo_condition_module["input_layer"],
        )
        self.emo_perceiver_encoder = PerceiverResampler(
            1024,
            dim_context=cfg_gpt.emo_condition_module["output_size"],
            ff_mult=cfg_gpt.emo_condition_module["perceiver_mult"],
            heads=cfg_gpt.emo_condition_module["attention_heads"],
            num_latents=1,
        )
        self.emo_layer = nn.Linear(cfg_gpt.model_dim, cfg_gpt.model_dim)
        self.emovec_layer = nn.Linear(1024, cfg_gpt.model_dim)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)

    def get_emo_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        speech_conditioning_input, mask = self.emo_conditioning_encoder(
            speech_conditioning_input.transpose(1, 2), cond_mel_lengths
        )
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(speech_conditioning_input, conds_mask)
        return conds.squeeze(1)

    def get_emovec(self, emo_speech_conditioning_latent, emo_cond_lengths):
        emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1, 2), emo_cond_lengths)
        emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
        return self.emo_layer(emo_vec_syn)

    def merge_emovec(
        self,
        speech_conditioning_latent,
        emo_speech_conditioning_latent,
        cond_lengths,
        emo_cond_lengths,
        alpha=1.0,
    ):
        emo_vec = self.get_emovec(emo_speech_conditioning_latent, emo_cond_lengths)
        base_vec = self.get_emovec(speech_conditioning_latent, cond_lengths)
        return base_vec + alpha * (emo_vec - base_vec)
