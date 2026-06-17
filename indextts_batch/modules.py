import abc
import gc
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Literal, Optional

import safetensors.torch
import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from torch.amp.autocast_mode import autocast
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SeamlessM4TFeatureExtractor,
)

from indextts.s2mel.modules.audio import mel_spectrogram
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.commons import MyModel, load_checkpoint2, sequence_mask
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.maskgct_utils import (
    build_semantic_codec,
    build_semantic_model,
)
from indextts_batch.device_wrapper import get_memory_allocated
from indextts_batch.models import EmotionExtractor, UnifiedVoice
from indextts_batch.utils import PACKAGE_ROOT
from indextts_batch.utils import logger as _root_logger


def normalize_emo_vec(emo_vector, apply_bias: bool = True):
    """对 8 维情感向量应用预训练 bias 并把总和钳制在 0.8 以内"""
    if apply_bias:
        # [高兴, 愤怒, 悲伤, 恐惧, 厌恶, 低落, 惊喜, 平静]
        emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
        emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]

    emo_sum = sum(emo_vector)
    if emo_sum > 0.8:
        scale_factor = 0.8 / emo_sum
        emo_vector = [vec * scale_factor for vec in emo_vector]

    return emo_vector


# ──────── 模块基类定义 ───────────────────────────────────────────────────────
class BaseIndexModule(abc.ABC):
    def __init__(
        self,
        cfg_path: str,
        model_dir: str | Path,
        device: str,
        dtype: torch.dtype,
        pipeline: Any = None,
    ):
        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        self.pipeline = pipeline
        self._is_loaded = False
        self._cfg_path = cfg_path
        self.logger = _root_logger.getChild(self.__class__.__name__)

    def ensure_loaded(self) -> None:
        """按需懒加载入口"""
        if not self._is_loaded:
            self.load_model()
            self._is_loaded = True

    @property
    def shared_dependencies(self) -> list:
        """和其他模块共享的模型"""
        return []

    @abc.abstractmethod
    def load_model(self) -> None:
        """加载模型到指定设备"""

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """前向推理接口，接受输入并返回输出"""

    def _recursive_to(self, obj: Any, target_device: str, shared_deps_ids: set) -> Any:
        """递归搬运容器内的数据到目标设备"""
        if id(obj) in shared_deps_ids:
            return obj

        if isinstance(obj, torch.nn.Module):
            return obj.to(target_device)
        elif isinstance(obj, torch.Tensor):
            return obj.to(target_device)
        elif isinstance(obj, dict):
            return {k: self._recursive_to(v, target_device, shared_deps_ids) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._recursive_to(v, target_device, shared_deps_ids) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._recursive_to(v, target_device, shared_deps_ids) for v in obj)
        elif isinstance(obj, set):
            return {self._recursive_to(v, target_device, shared_deps_ids) for v in obj}
        return obj

    def to(self, target_device: str) -> None:
        """移动模型与张量到目标设备，自动深层遍历，并跳过 shared_dependencies"""
        shared_deps_ids = {id(dep) for dep in self.shared_dependencies if dep is not None}

        for attr_name, attr_val in list(vars(self).items()):
            new_val = self._recursive_to(attr_val, target_device, shared_deps_ids)
            if new_val is not attr_val:
                setattr(self, attr_name, new_val)

        self.device = target_device


# ──────── 文本情感推理模块（Qwen）─────────────────────────────────────────────
class QwenEmotionModule(BaseIndexModule):
    """基于 Qwen 的情感分类模块，继承 BaseIndexModule 接入 VRAMManager 管控"""

    def __init__(
        self,
        cfg_path: str,
        model_dir: str | Path,
        device: str,
        dtype: torch.dtype,
        infer_cfg: DictConfig | None = None,
        pipeline: Any = None,
    ):
        super().__init__(cfg_path, model_dir, device, dtype, pipeline=pipeline)
        self.infer_cfg = infer_cfg or OmegaConf.create()
        self.tokenizer: Any = None
        self.model: Any = None

        # 情感分类常量
        self.prompt = "文本情感分类"
        self.cn_key_to_en = {
            "高兴": "happy",
            "愤怒": "angry",
            "悲伤": "sad",
            "恐惧": "afraid",
            "反感": "disgusted",
            "低落": "melancholic",
            "惊讶": "surprised",
            "自然": "calm",
        }
        self.desired_vector_order = [
            "高兴",
            "愤怒",
            "悲伤",
            "恐惧",
            "反感",
            "低落",
            "惊讶",
            "自然",
        ]
        self.melancholic_words = {
            "低落",
            "melancholy",
            "melancholic",
            "depression",
            "depressed",
            "gloomy",
        }
        self.max_score = 1.2
        self.min_score = 0.0
        self._batch_max_new_tokens = 2048

    def load_model(self) -> None:
        """加载 Qwen 模型并注册到 VRAMManager"""
        self.logger.info("加载模型组件 (Qwen 情感推理)")
        qwen_model_dir = str(self.model_dir / self.cfg.qwen_emo_path)
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_dir)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                qwen_model_dir,
                torch_dtype=torch.float16,
            )
            .to(self.device)
            .eval()
        )
        self._is_loaded = True

    def __call__(self, texts: set[str]) -> dict[str, list[float]]:
        """批量 Qwen 文本情感推理，返回 {文本: 8维向量} 映射"""
        if not texts:
            return {}
        qwen_texts_list = list(texts)
        results_list = self.batch_inference(qwen_texts_list)
        mapping: dict[str, list[float]] = {
            text: list(d.values()) if d is not None else [0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.0, 0.0]
            for text, d in zip(qwen_texts_list, results_list, strict=False)
        }
        self.logger.info(f"Qwen 批量推理完成: {len(mapping)} 条情感文本")
        return mapping

    def clamp_score(self, value: float) -> float:
        """把单维情感得分钳制到 [min_score, max_score] 区间"""
        return max(self.min_score, min(self.max_score, value))

    def convert(self, content: dict[str, Any]) -> dict[str, float]:
        """将模型输出的中文键-数值字典转成英文有序字典，并做缺失值补零与全零默认"""
        emotion_dict = {
            self.cn_key_to_en[cn_key]: self.clamp_score(float(content.get(cn_key, 0.0)))
            for cn_key in self.desired_vector_order
        }
        if all(val <= 0.0 for val in emotion_dict.values()):
            self.logger.info("模型未检测到情感; 默认为：反感 0.35")
            emotion_dict["disgusted"] = 0.35
        return emotion_dict

    def _parse_single_output(self, output_ids: list[int], text_input: str) -> dict[str, float]:
        """对单条生成的 token 序列切分 thinking 块、解析 JSON"""
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)
        try:
            content = json.loads(content)
        except json.decoder.JSONDecodeError:
            content = {m.group(1): float(m.group(2)) for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)}

        text_input_lower = text_input.lower()
        if any(word in text_input_lower for word in self.melancholic_words):
            content["悲伤"], content["低落"] = (
                content.get("低落", 0.0),
                content.get("悲伤", 0.0),
            )

        return self.convert(content)

    @torch.inference_mode()
    def inference(self, text_input: str) -> dict[str, float]:
        """对单条文本做情感分类推理，返回 8 维英文情感字典，使用2048而不是原始项目的32k作为默认上下文大小"""
        messages = [
            {"role": "system", "content": f"{self.prompt}"},
            {"role": "user", "content": f"{text_input}"},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=2048,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        return self._parse_single_output(output_ids, text_input)

    @torch.inference_mode()
    def batch_inference(self, text_inputs: list[str], batch_size: int = 32) -> list[Optional[dict[str, float]]]:
        """对一组文本做批量情感分类推理"""
        if not text_inputs:
            return []

        results: list[Optional[dict[str, float]]] = [None] * len(text_inputs)

        for chunk_start in range(0, len(text_inputs), batch_size):
            chunk = text_inputs[chunk_start : chunk_start + batch_size]

            templated = [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": t},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for t in chunk
            ]

            orig_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            model_inputs = self.tokenizer(
                templated,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            self.tokenizer.padding_side = orig_padding_side

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self._batch_max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            for i, gen_ids in enumerate(generated_ids):
                input_len = model_inputs.input_ids[i].size(0)
                output_ids = gen_ids[input_len:].tolist()
                results[chunk_start + i] = self._parse_single_output(output_ids, chunk[i])

        return results


# ──────── 参考音频编码器 (特征 + 声纹 + 情感) ─────────────────────────────────
class ReferenceEncoder(BaseIndexModule):
    """
    在音频不变的情况下，输出是可以缓存的，因此可以在获取一次之后卸载用到的模型。

    输出：
        spk_cond_emb：参考人音频特征嵌入
        style：参考人音频声纹
        prompt_condition：和梅尔频谱时间同步的语义特征
        ref_mel：梅尔频谱
        S_ref：参考人音频的量化的离散特征
        emovec：情感向量
    """

    def __init__(
        self,
        cfg_path: str,
        model_dir: str,
        device: str,
        dtype: torch.dtype,
        infer_cfg: DictConfig | None = None,
        pipeline: Any = None,
    ):
        super().__init__(cfg_path, model_dir, device, dtype, pipeline=pipeline)
        self.infer_cfg = infer_cfg or OmegaConf.create()

        # 声纹提取子模块
        self.extract_features: Any = None
        self.semantic_model: Any = None
        self.semantic_mean: Any = None
        self.semantic_std: Any = None
        self.semantic_codec: Any = None
        self.campplus_model: Any = None
        self.mel_fn: Any = None

        # 抽离出的专属轻量化情感提取器
        self.emo_extractor: Optional[EmotionExtractor] = None
        self.emo_matrix: Any = None
        self.spk_matrix: Any = None
        self.emo_num: list = []
        self.emo_audios: dict[str, tuple[torch.Tensor, int]] = {}

    def load_model(self) -> None:
        start_mem = get_memory_allocated(self.device)
        start_time = time.perf_counter()
        self.logger.info("加载模型组件 (W2v-BERT + CAMPPlus)")

        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

        w2v_stat_path = self.model_dir / self.cfg.w2v_stat
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(str(w2v_stat_path))
        self.semantic_model = self.semantic_model.to(self.device).eval()
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)

        campplus_ckpt_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = self.campplus_model.to(self.device).eval()

        spect_params = self.cfg.s2mel["preprocess_params"]["spect_params"]

        mel_fn_args = {
            "n_fft": spect_params["n_fft"],
            "win_size": spect_params["win_length"],
            "hop_size": spect_params["hop_length"],
            "num_mels": spect_params["n_mels"],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": spect_params.get("fmin", 0),
            "fmax": None if spect_params.get("fmax", "None") == "None" else 8000,
            "center": False,
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

        self.load_emotion_matrices()
        self._load_emotion_extractor()

        elapsed = time.perf_counter() - start_time
        mem_diff = (get_memory_allocated(self.device) - start_mem) / 1e9
        total_mem = get_memory_allocated(self.device) / 1e9
        self.logger.info(
            f"模型加载完成，占用显存：{mem_diff:.2f} GB, 当前已占用显存：{total_mem:.2f} GB, 耗时：{elapsed:.2f}秒"
        )

    def load_emotion_matrices(self) -> None:
        """加载情感声纹映射矩阵"""
        emo_matrix_path = self.model_dir / self.cfg.emo_matrix
        spk_matrix_path = self.model_dir / self.cfg.spk_matrix
        self.emo_num = list(self.cfg.emo_num)

        emo_matrix_raw = torch.load(emo_matrix_path, map_location=self.device)
        spk_matrix_raw = torch.load(spk_matrix_path, map_location=self.device)

        self.emo_matrix = torch.split(emo_matrix_raw, self.emo_num)
        self.spk_matrix = torch.split(spk_matrix_raw, self.emo_num)

        self.logger.debug("情感声纹映射矩阵加载完成")

    def _load_emotion_extractor(self) -> None:
        """自托管自动缓存机制：首次提取GPT模型的 30MB 权重切片并缓存，后续直接读取"""
        self.emo_extractor = EmotionExtractor(self.cfg.gpt)
        cache_file = PACKAGE_ROOT / "models" / "emotion_extractor.ckpt"

        if cache_file.exists():
            self.logger.info("已缓存情感提取器参数，直接加载")
            state = torch.load(str(cache_file), map_location="cpu")
            self.emo_extractor.load_state_dict(state)
        else:
            self.logger.info("执行首次运行GPT权重切片提取")
            gpt_path = self.model_dir / self.cfg.gpt_checkpoint
            if not gpt_path.exists():
                raise FileNotFoundError(f"未找到 GPT 权重文件：{gpt_path}，无法执行首次权重提取。")

            gpt_state_dict = torch.load(str(gpt_path), map_location="cpu")
            sub_state_dict = {}
            for k, v in gpt_state_dict.items():
                if (
                    k.startswith("emo_conditioning_encoder.")
                    or k.startswith("emo_perceiver_encoder.")
                    or k.startswith("emovec_layer.")
                    or k.startswith("emo_layer.")
                ):
                    sub_state_dict[k] = v

            self.emo_extractor.load_state_dict(sub_state_dict)
            self.logger.info(f"成功提取 {len(sub_state_dict)} 个情感相关权重，保存至本地缓存文件：{cache_file}")
            torch.save(sub_state_dict, str(cache_file))

            del gpt_state_dict, sub_state_dict
            gc.collect()

        self.emo_extractor = self.emo_extractor.to(self.device).eval()

    @torch.inference_mode()
    def _get_emb(self, input_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]
        return (feat - self.semantic_mean) / self.semantic_std

    @torch.inference_mode()
    def _find_most_similar_cosine(self, query_vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        similarities = F.cosine_similarity(query_vector.float(), matrix.float(), dim=1)
        return torch.argmax(similarities)

    def _get_max_audio_length(self) -> float:
        return self.infer_cfg.get("reference_encoder", {}).get("max_audio_length_seconds", 15.0)  # type: ignore[no-any-return]

    @torch.inference_mode()
    def __call__(
        self,
        audio_path: str,
        max_audio_length_seconds: float | None = None,
    ) -> dict[str, Any]:
        self.logger.info(f"正在编码参考音频: {audio_path}")

        if max_audio_length_seconds is None:
            max_audio_length_seconds = self._get_max_audio_length()

        def _load_raw(path: str) -> tuple[torch.Tensor, int]:
            info = torchaudio.info(path)
            sr = info.sample_rate
            max_frames = int(max_audio_length_seconds * sr)
            wav_t, sr = torchaudio.load(path, num_frames=max_frames)
            if wav_t.shape[0] > 1:
                wav_t = wav_t.mean(dim=0, keepdim=True)
            return wav_t, sr

        def _extract_w2v_emb(wav_16k: torch.Tensor) -> torch.Tensor:
            wav_np = wav_16k.squeeze(0).numpy()
            inputs = self.extract_features(wav_np, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            return self._get_emb(input_features, attention_mask)

        raw, raw_sr = _load_raw(audio_path)
        audio_22k = torchaudio.functional.resample(raw, raw_sr, 22050)
        audio_16k = torchaudio.functional.resample(raw, raw_sr, 16000)

        spk_cond_emb = _extract_w2v_emb(audio_16k)
        ref_mel = self.mel_fn(audio_22k.to(self.device).float())

        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(self.device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = self.campplus_model(feat.unsqueeze(0))

        return {
            "spk_cond_emb": spk_cond_emb,
            "spk_cond_length": int(spk_cond_emb.shape[1]),
            "style": style,
            "ref_mel": ref_mel,
        }

    @torch.inference_mode()
    def extract_emo_cond(
        self,
        audio_path: str,
        max_audio_length_seconds: float | None = None,
    ) -> tuple[torch.Tensor, int]:
        """
        从情感参考音频中提取 w2v-bert 语义嵌入

        参数:
            - audio_path (str): 情感参考音频文件路径。
            - max_audio_length_seconds (float): 最长截断秒数，默认 15。
        返回值:
            - Tuple[torch.Tensor, int]:
                emo_cond_emb    - [1, T, 1024]
                emo_cond_length - T
        """
        if max_audio_length_seconds is None:
            max_audio_length_seconds = self._get_max_audio_length()
        info = torchaudio.info(audio_path)
        wav_sr = info.sample_rate
        max_samples = int(max_audio_length_seconds * wav_sr)
        wav_t, _ = torchaudio.load(audio_path, num_frames=max_samples)
        if wav_t.shape[0] > 1:
            wav_t = wav_t.mean(dim=0, keepdim=True)
        audio_16k = torchaudio.functional.resample(wav_t, wav_sr, 16000)
        wav_np = audio_16k.squeeze(0).numpy()
        inputs = self.extract_features(wav_np, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        emo_cond_emb = self._get_emb(input_features, attention_mask)
        return emo_cond_emb, int(emo_cond_emb.shape[-1])

    @torch.inference_mode()
    def compute_emovec(
        self,
        spk_cond_emb: torch.Tensor,
        spk_cond_length: int,
        style: torch.Tensor,
        emo_cond_emb: torch.Tensor,
        emo_cond_length: int,
        emo_vector: list | None = None,
        emo_alpha: float = 1.0,
        use_random: bool = False,
    ) -> torch.Tensor:
        """
        情感向量计算：
            1) 基础插值：base_vec + alpha * (emo_vec - base_vec)
            2) 若提供 8 维情感向量，则叠加预训练情感矩阵的加权和。
        参数:
            - spk_cond_emb (torch.Tensor): 说话人 w2v-bert 特征，形状 [1, T, 1024]。
            - spk_cond_length (int): 说话人 w2v-bert 时间长度。
            - style (torch.Tensor): 说话人 CAMPPlus 声纹特征，形状 [1, 192]。
            - emo_cond_emb (torch.Tensor): 情感参考音频 w2v-bert 特征，形状 [1, T', 1024]。
            - emo_cond_length (int): 情感参考音频 w2v-bert 时间长度。
            - emo_vector (list | None): 8 维情感向量 [高兴, 愤怒, 悲伤, 恐惧, 反感, 低落, 惊讶, 自然]
            - emo_alpha (float): 情感插值强度，默认 1.0。
            - use_random (bool): 默认寻找和音色最接近的预训练情感，若为 True 则随机选择。
        返回值:
            - torch.Tensor: emovec 张量，形状 [1, 1280]
        """
        if self.emo_extractor is None:
            raise ValueError("compute_emovec 需要 EmotionExtractor 已载入")
        if self.emo_matrix is None or self.spk_matrix is None:
            raise ValueError("需先调用 load_emotion_matrices")

        spk_cond_emb_gpu = spk_cond_emb.to(device=self.device)
        emo_cond_gpu = emo_cond_emb.to(device=self.device)
        style_gpu = style.to(device=self.device)
        enabled = self.dtype is not None
        device_type = torch.device(self.device).type

        with autocast(device_type, enabled=enabled, dtype=self.dtype):
            emovec = self.emo_extractor.merge_emovec(
                spk_cond_emb_gpu,
                emo_cond_gpu,
                torch.tensor([spk_cond_length], device=self.device),
                torch.tensor([emo_cond_length], device=self.device),
                alpha=emo_alpha,
            )

            # 叠加预训练情感矩阵，最终音频的情感比例为
            #   音频情感：        1 - sum(emo_vector)
            #   情感向量预训练情感：   sum(emo_vector)
            if emo_vector is not None:
                emo_vector = normalize_emo_vec(emo_vector, apply_bias=True)
                weight_vector = torch.tensor(emo_vector, device=self.device)
                if use_random:
                    random_index = [random.randint(0, x - 1) for x in self.emo_num]
                else:
                    random_index = [self._find_most_similar_cosine(style_gpu, tmp) for tmp in self.spk_matrix]

                selected_emo_matrix = [
                    tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix, strict=False)
                ]
                selected_emo_matrix = torch.cat(selected_emo_matrix, 0)
                emovec_mat = weight_vector.unsqueeze(1) * selected_emo_matrix
                emovec_mat = torch.sum(emovec_mat, 0).unsqueeze(0)
                emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec
        return emovec

    def register_emotion_audio(self, audio_path: str) -> str:
        """缓存情感音频的 w2v-bert 特征"""
        if audio_path not in self.emo_audios:
            emo_cond_emb, emo_cond_len = self.extract_emo_cond(audio_path)
            self.emo_audios[audio_path] = (emo_cond_emb, emo_cond_len)
            self.logger.info(f"已缓存情感参考音频特征: {audio_path}")
        return audio_path

    @torch.inference_mode()
    def prepare_emotion(
        self,
        texts: list[str],
        speaker_audio: list[str] | str,
        speakers_bank: dict[str, dict],
        *,
        emo_mode: Literal["vector", "text", "audio"] = "vector",
        emo_vector: list[float] | None = None,
        emo_alpha: float = 1.0,
        emo_audio_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        提取说话人/情感音频特征，组装 items

        参数:
            - texts: 待生成文本列表
            - speaker_audio: 音频路径
            - speakers_bank: 说话人特征缓存
            - emo_mode: 情感获取方式（"vector" 直接使用 emo_vector；"text" 留空等待 pipeline 调用 Qwen 填入；"audio" 使用情感参考音频）
            - emo_vector: 8 维情感向量（仅 emo_mode="vector" 时有效）
            - emo_alpha: 情感强度
            - emo_audio_prompt: 情感参考音频路径
        """
        n = len(texts)

        # 归一化 speakers
        speakers: list[str]
        if isinstance(speaker_audio, str):
            speakers = [speaker_audio] * n
        elif isinstance(speaker_audio, list):
            if len(speaker_audio) != n:
                raise ValueError(f"一共有 {n} 个待生成的文本，但是只有 {len(speaker_audio)} 个参考音频")
            speakers = speaker_audio
        else:
            raise TypeError(f"speaker_audio 必须是 str 或 list[str]，但是是 {type(speaker_audio)}")

        # 2. 缓存说话人特征
        need_speaker_cache = set(speakers) - set(speakers_bank)
        if need_speaker_cache:
            for key in need_speaker_cache:
                features = self(key)
                for k, v in features.items():
                    if isinstance(v, torch.Tensor):
                        features[k] = v.cpu()
                speakers_bank[key] = features
            self.logger.info(f"已缓存 {len(need_speaker_cache)} 个参考音频特征")

        # 3. 根据 emo_mode 构建 per-item 情感参数
        emo_vector_per_item: list[Optional[list[float]]] = [None] * n
        emo_audio_per_item: list[str] = [""] * n
        emo_alpha_per_item: list[float] = [1.0] * n

        if emo_mode == "vector":
            if emo_vector is None:
                raise ValueError("emo_mode='vector' 时必须提供 emo_vector")
            scale = max(0.0, min(1.0, emo_alpha))
            if scale != 1.0:
                emo_vector = [int(x * scale * 10000) / 10000 for x in emo_vector]
            for i in range(n):
                emo_vector_per_item[i] = emo_vector
                emo_audio_per_item[i] = speakers[i]

        elif emo_mode == "text":
            # emo_vector 留空，由 pipeline 调用 Qwen 后填入
            for i in range(n):
                emo_audio_per_item[i] = speakers[i]

        elif emo_mode == "audio":
            for i in range(n):
                emo_audio_per_item[i] = emo_audio_prompt or speakers[i]
                emo_alpha_per_item[i] = emo_alpha
        else:
            raise ValueError(f"不支持的 emo_mode: {emo_mode}，可选 'vector' / 'text' / 'audio'")

        # 4. 缓存情感音频特征
        if emo_mode == "audio":
            emo_audio_set = set(emo_audio_per_item) - set(speakers)
            if emo_audio_set:
                for key in emo_audio_set:
                    if key not in self.emo_audios:
                        ec_emb, ec_len = self.extract_emo_cond(key)
                        self.emo_audios[key] = (ec_emb, ec_len)
                self.logger.info(f"已缓存 {len(emo_audio_set)} 个情感音频特征")

        # 5. 组装 items（emo_vector 可能在 pipeline 中由 Qwen 后续回填）
        items: list[dict[str, Any]] = []
        for i in range(n):
            items.append(
                {
                    "orig_idx": i,
                    "speaker_index": speakers[i],
                    "emo_audio_path": emo_audio_per_item[i],
                    "emo_vector": emo_vector_per_item[i],
                    "emo_alpha": emo_alpha_per_item[i],
                }
            )

        return items

    @torch.inference_mode()
    def compute_emovecs(
        self,
        items: list[dict[str, Any]],
        speakers_bank: dict[str, dict],
    ) -> list[dict[str, Any]]:
        emovec_cache: dict[tuple, torch.Tensor] = {}
        for item in items:
            speaker = item["speaker_index"]
            spk = speakers_bank[speaker]

            emo_audio_path = item["emo_audio_path"]
            if emo_audio_path == speaker:
                emo_cond_emb = spk["spk_cond_emb"]
                emo_cond_len = spk["spk_cond_length"]
            else:
                emo_cond_emb, emo_cond_len = self.emo_audios[emo_audio_path]

            ev = item["emo_vector"]
            emo_alpha = item["emo_alpha"]
            vec_tuple = tuple(ev) if ev is not None else None
            cache_key = (speaker, emo_audio_path, vec_tuple, emo_alpha)

            if cache_key in emovec_cache:
                emovec = emovec_cache[cache_key]
            else:
                emovec = self.compute_emovec(
                    spk_cond_emb=spk["spk_cond_emb"],
                    spk_cond_length=spk["spk_cond_length"],
                    style=spk["style"],
                    emo_cond_emb=emo_cond_emb.to(self.device),
                    emo_cond_length=emo_cond_len,
                    emo_vector=ev,
                    emo_alpha=emo_alpha,
                )
                emovec = emovec.cpu()
                emovec_cache[cache_key] = emovec

            item["emovec"] = emovec

        self.logger.info(f"已缓存 {len(emovec_cache)} 个情感向量")
        return items


# ──────── GPT 自回归解码 ─────────────────────────────────────────────────────
class GPTGenerator(BaseIndexModule):
    """
    GPT 自回归解码模块，生成离散的梅尔频谱编码（Codes）和隐变量（Latent）

    输入:
        spk_cond_emb
        text_tokens
        emovec
    输出：
        fixed_codes：梅尔频谱编码
        code_lens：编码长度
        latent：最后一层隐向量
    """

    def __init__(
        self,
        cfg_path: str,
        model_dir: str,
        device: str,
        dtype: torch.dtype,
        use_accel: bool,
        use_deepspeed: bool = False,
        infer_cfg: Any = None,
        pipeline: Any = None,
    ):
        super().__init__(cfg_path, model_dir, device, dtype, pipeline=pipeline)
        self.gpt: UnifiedVoice | None = None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        self.use_accel = use_accel
        self.use_deepspeed = use_deepspeed
        self.infer_cfg = infer_cfg or OmegaConf.create()

    @property
    def shared_dependencies(self) -> list:
        # 核心模型
        return [self.gpt]

    def load_model(self) -> None:
        start_mem = get_memory_allocated(self.device)
        start_time = time.perf_counter()
        self.logger.info("加载模型组件 (GPT)")

        self.gpt = UnifiedVoice(**self.cfg.gpt, use_accel=self.use_accel)
        gpt_path = self.model_dir / self.cfg.gpt_checkpoint
        load_checkpoint(self.gpt, str(gpt_path))
        self.gpt = self.gpt.to(self.device)

        if self.dtype == torch.float16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()

        # 配置自回归高频生成的 KV Cache 与推演模式
        self.gpt.post_init_gpt2_config(
            use_deepspeed=self.use_deepspeed,
            kv_cache=True,
            half=(self.dtype == torch.float16),
        )

        elapsed = time.perf_counter() - start_time
        mem_diff = (get_memory_allocated(self.device) - start_mem) / 1e9
        total_mem = get_memory_allocated(self.device) / 1e9
        self.logger.info(
            f"模型加载完成，占用显存：{mem_diff:.2f} GB, 当前已占用显存：{total_mem:.2f} GB, 耗时：{elapsed:.2f}秒"
        )

    @torch.inference_mode()
    def __call__(
        self,
        spk_cond_emb=None,
        emovec=None,
        text_tokens=None,
        speech_conditioning_latent=None,
        max_mel_tokens=None,
        **generation_kwargs,
    ) -> dict[str, torch.Tensor]:
        if self.gpt is None:
            raise RuntimeError("GPTGenerator 模型未加载")
        if text_tokens is None or emovec is None:
            raise ValueError("text_tokens 和 emovec 不能为 None")

        text_tokens = text_tokens.to(self.device)
        emovec = emovec.to(self.device)
        enabled = self.dtype is not None
        device_type = torch.device(self.device).type

        # 推理参数
        do_sample = generation_kwargs.pop("do_sample", self.infer_cfg.generation.do_sample)
        top_p = generation_kwargs.pop("top_p", self.infer_cfg.generation.top_p)
        top_k = generation_kwargs.pop("top_k", self.infer_cfg.generation.top_k)
        temperature = generation_kwargs.pop("temperature", self.infer_cfg.generation.temperature)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", self.infer_cfg.generation.length_penalty)
        num_beams = generation_kwargs.pop("num_beams", self.infer_cfg.generation.num_beams)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", self.infer_cfg.generation.repetition_penalty)
        typical_sampling = generation_kwargs.pop("typical_sampling", self.infer_cfg.generation.typical_sampling)
        typical_mass = generation_kwargs.pop("typical_mass", self.infer_cfg.generation.typical_mass)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", self.infer_cfg.generation.max_mel_tokens)

        with autocast(device_type, enabled=enabled, dtype=self.dtype):
            # GPT自回归生成
            if speech_conditioning_latent is not None:
                speech_conditioning_latent = speech_conditioning_latent.to(self.device)
                codes, _ = self.gpt.inference_speech(
                    speech_conditioning_latent=speech_conditioning_latent,
                    text_inputs=text_tokens,
                    emovec=emovec,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=autoregressive_batch_size,
                    length_penalty=length_penalty,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    typical_sampling=typical_sampling,
                    typical_mass=typical_mass,
                    max_generate_length=max_mel_tokens,
                    **generation_kwargs,
                )
            else:
                if spk_cond_emb is None:
                    raise ValueError("spk_cond_emb 不能为 None")
                spk_cond_emb = spk_cond_emb.to(self.device)
                codes, speech_conditioning_latent = self.gpt.inference_speech(
                    spk_cond_emb=spk_cond_emb,
                    text_inputs=text_tokens,
                    emo_cond_emb=spk_cond_emb,
                    cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                    emo_cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                    emovec=emovec,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=autoregressive_batch_size,
                    length_penalty=length_penalty,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    typical_sampling=typical_sampling,
                    typical_mass=typical_mass,
                    max_generate_length=max_mel_tokens,
                    **generation_kwargs,
                )  # [B, T] T的大小是GPT模型决定的

            # 对批量生成的结果进行对齐操作
            mask = codes == self.stop_mel_token
            has_stop = mask.any(dim=1)
            first_stop = mask.int().argmax(dim=1)
            code_lens = torch.where(has_stop, first_stop, codes.shape[1])
            if (code_lens == codes.shape[1]).any():
                self.logger.warning("部分样本在生成的梅尔频谱编码中未找到停止标记")
            max_code_len = code_lens.max().item()
            fixed_codes = codes[:, :max_code_len]

            # 单次批量提取隐变量
            batch_size = text_tokens.shape[0]
            stop_token = self.cfg.gpt.stop_text_token
            text_lengths = text_tokens.shape[1] - (text_tokens != stop_token).int().flip(dims=[1]).argmax(dim=1)
            dummy_cond = torch.zeros(batch_size, 1, 1024, device=self.device)
            latent = self.gpt(
                speech_conditioning_latent=speech_conditioning_latent,
                text_inputs=text_tokens,
                text_lengths=text_lengths,
                mel_codes=fixed_codes,
                mel_codes_lengths=code_lens,
                emo_speech_conditioning_latent=dummy_cond,
                cond_mel_lengths=torch.tensor([1024] * batch_size, device=self.device),
                emo_cond_mel_lengths=torch.tensor([1024] * batch_size, device=self.device),
                emo_vec=emovec,
                use_speed=torch.zeros(batch_size, dtype=torch.long, device=self.device),
            )

        return {
            "fixed_codes": fixed_codes,
            "code_lens": code_lens,
            "latent": latent,
        }


# ──────── 梅尔频谱扩散与波形重建 ──────────────────────────────────────────────
def _patch_cfm_solve_euler(cfm_model):
    """替换 CFM.solve_euler，仅在 estimator 调用时包裹 BF16 autocast，步进累保持 fp32"""

    def _solve_euler_amp(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
        t, _, _ = t_span[0], t_span[-1], t_span[1] - t_span[0]
        sol = []
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0

        use_amp = getattr(self, "use_amp", False)
        amp_dtype = getattr(self, "amp_dtype", torch.float32)

        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
                stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                stacked_x = torch.cat([x, x], dim=0)
                stacked_x_lens = torch.cat([x_lens, x_lens], dim=0)
                stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)
                with torch.autocast(device_type=x.device.type, enabled=use_amp, dtype=amp_dtype):
                    stacked_dphi_dt = self.estimator(
                        stacked_x,
                        stacked_prompt_x,
                        stacked_x_lens,
                        stacked_t,
                        stacked_style,
                        stacked_mu,
                    )
                stacked_dphi_dt = stacked_dphi_dt.float()
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                with torch.autocast(device_type=x.device.type, enabled=use_amp, dtype=amp_dtype):
                    dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)
                dphi_dt = dphi_dt.float()

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            x[:, :, :prompt_len] = 0

        return sol[-1]

    cfm_model.solve_euler = _solve_euler_amp.__get__(cfm_model, type(cfm_model))


class AudioGenerator(BaseIndexModule):
    """
    使用 CFM 模型进行梅尔频谱扩散重建，使用 BigVGAN 进行波形生成

    输入:
        latent, fixed_codes, code_lens (GPT 自回归输出)
        prompt_condition, ref_mel, style (参考音频特征)
    输出：
        wav (音频)
    """

    def __init__(
        self,
        cfg_path: str,
        model_dir: str,
        device: str,
        dtype: torch.dtype,
        use_cuda_kernel: bool = False,
        use_torch_compile: bool = False,
        infer_cfg: Any = None,
        pipeline: Any = None,
    ):
        super().__init__(cfg_path, model_dir, device, dtype, pipeline=pipeline)
        self.use_cuda_kernel = use_cuda_kernel and device.startswith("cuda")
        self.use_torch_compile = use_torch_compile and device.startswith("cuda")
        self.semantic_codec: Any = None
        self.s2mel: Any = None
        self.bigvgan: Any = None
        self.infer_cfg = infer_cfg or OmegaConf.create()

    def load_model(self) -> None:
        start_mem = get_memory_allocated(self.device)
        start_time = time.perf_counter()
        self.logger.info("加载模型组件 (Semantic Codec + s2mel + BigVGAN)")

        # 载入 s2mel
        s2mel_path = self.model_dir / self.cfg.s2mel_checkpoint
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            str(s2mel_path),
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device).eval()

        if self.use_torch_compile:
            self.logger.info("启用 torch.compile 优化")
            cfm = self.s2mel.models["cfm"]
            cfm.estimator.setup_caches(max_batch_size=64, max_seq_length=16384)
            cfm.estimator = torch.compile(cfm.estimator, fullgraph=False, dynamic=True)

        # CFM 估计器 BF16 混合精度（仅包裹 estimator，Euler 步进保持 fp32）
        cfm = self.s2mel.models["cfm"]
        accel_cfg = self.infer_cfg.get("acceleration", {})
        cfm.use_amp = accel_cfg.get("cfm_amp", False)
        cfm.amp_dtype = torch.bfloat16 if cfm.use_amp and not self.device.startswith("mps") else torch.float32
        if cfm.use_amp:
            _patch_cfm_solve_euler(cfm)
            self.logger.info("启用 CFM 估计器 BF16 混合精度进行加速")

        # 载入语义编码器 Semantic Codec（GPT Codes 反量化用）
        self.semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(self.semantic_codec, semantic_code_ckpt)
        self.semantic_codec = self.semantic_codec.to(self.device).eval()

        # 载入 BigVGAN 并剥离权重归一化层
        bigvgan_name = self.cfg.vocoder.name
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()

        elapsed = time.perf_counter() - start_time
        mem_diff = (get_memory_allocated(self.device) - start_mem) / 1e9
        total_mem = get_memory_allocated(self.device) / 1e9
        self.logger.info(
            f"模块加载完成，占用显存：{mem_diff:.2f} GB, 当前已占用显存：{total_mem:.2f} GB, 耗时：{elapsed:.2f}秒"
        )

    @torch.inference_mode()
    def __call__(
        self,
        latent: torch.Tensor,
        fixed_codes: torch.Tensor,
        code_lens: torch.Tensor,
        spk_cond_emb: torch.Tensor,
        ref_mel: torch.Tensor,
        style: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,
        diffusion_steps: int | None = None,
        inference_cfg_rate: float | None = None,
        temperature: float | None = None,
    ) -> dict[str, torch.Tensor | float]:
        audio_cfg = self.infer_cfg.get("audio", {})
        if diffusion_steps is None:
            diffusion_steps = audio_cfg.get("diffusion_steps", 25)
        if inference_cfg_rate is None:
            inference_cfg_rate = float(audio_cfg.get("inference_cfg_rate", 0.7))
        else:
            inference_cfg_rate = float(inference_cfg_rate)
        if temperature is None:
            temperature = audio_cfg.get("temperature", 1.0)

        latent = latent.to(self.device).float()
        fixed_codes = fixed_codes.to(self.device)
        code_lens = code_lens.to(self.device)
        spk_cond_emb = spk_cond_emb.to(self.device).float()
        ref_mel = ref_mel.to(self.device).float()
        style = style.to(self.device).float()

        _, S_ref = self.semantic_codec.quantize(spk_cond_emb)

        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(self.device)
        prompt_condition = self.s2mel.models["length_regulator"](
            S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None
        )[0]

        mel_scale = self.infer_cfg.s2mel.mel_scale
        hop_length = self.infer_cfg.s2mel.preprocess_params.hop_length
        target_length = int((code_lens[0] * mel_scale).item())
        p_len = int(prompt_lens[0].item()) if prompt_lens is not None else prompt_condition.size(1)
        cfm_length = p_len + target_length

        max_cache_len = max(8192, int(cfm_length))
        cfm_estimator = self.s2mel.models["cfm"].estimator
        orig_estimator = getattr(cfm_estimator, "_orig_mod", cfm_estimator)
        orig_estimator.setup_caches(
            max_batch_size=2 if inference_cfg_rate > 0 else 1,
            max_seq_length=max_cache_len,
        )

        device_type = torch.device(self.device).type
        with autocast(device_type, enabled=False):
            latent_mapped = self.s2mel.models["gpt_layer"](latent)
            clamped_codes = torch.clamp(fixed_codes, min=0, max=8191)

            S_infer = self.semantic_codec.quantizer.vq2emb(clamped_codes.unsqueeze(0)).transpose(1, 2)

            code_mask = sequence_mask(code_lens, max_length=fixed_codes.size(1)).unsqueeze(-1).to(S_infer.dtype)
            S_infer = (S_infer + latent_mapped) * code_mask

            S_infer = S_infer[:, : int(code_lens[0].item()), :]

            cond = self.s2mel.models["length_regulator"](
                S_infer,
                ylens=torch.tensor([target_length], device=self.device),
                n_quantizers=3,
                f0=None,
            )[0]

            t_len = target_length
            total_len = p_len + t_len
            cat_condition = torch.zeros(
                1,
                total_len,
                cond.shape[-1],
                device=cond.device,
                dtype=cond.dtype,
            )
            cat_condition[0, :p_len] = prompt_condition[0, :p_len]
            cat_condition[0, p_len : p_len + t_len] = cond[0, :t_len]

            t_cfm = time.perf_counter()
            vc_target = self.s2mel.models["cfm"].inference(
                cat_condition,
                torch.tensor([cfm_length], device=self.device),
                ref_mel,
                style,
                None,
                diffusion_steps,
                temperature=temperature,
                inference_cfg_rate=inference_cfg_rate,
            )
            t_cfm_end = time.perf_counter()

            gen_mel = vc_target[0, :, p_len : p_len + t_len]

            safety_tail_frames = audio_cfg.get("safety_tail_frames", 20)
            abs_trigger_frames = audio_cfg.get("abs_trigger_frames", 86)
            co_trigger_frames = audio_cfg.get("co_trigger_frames", 35)
            co_trigger_ratio = audio_cfg.get("co_trigger_ratio", 0.15)
            silence_energy_threshold = audio_cfg.get("silence_energy_threshold", 130.0)

            frame_energy = gen_mel.pow(2).mean(dim=0)
            non_silent_mask = frame_energy < silence_energy_threshold
            non_silent_indices = non_silent_mask.nonzero(as_tuple=True)[0]

            refined_target_length = target_length
            if len(non_silent_indices) > 0:
                last_vocal_frame = int(non_silent_indices[-1].item())
                new_len = min(t_len, last_vocal_frame + 1 + safety_tail_frames)
                if new_len < t_len:
                    silence_frames = t_len - new_len
                    silence_ratio = silence_frames / t_len
                    if (silence_frames >= abs_trigger_frames) or (
                        silence_frames >= co_trigger_frames and silence_ratio >= co_trigger_ratio
                    ):
                        ms_per_frame = hop_length * 1000.0 / self.infer_cfg.s2mel.preprocess_params.sr
                        self.logger.info(
                            f"触发尾部静音修剪，"
                            f"原始长度 {t_len} 帧，裁剪至 {new_len} 帧，"
                            f"裁剪静音帧数: {silence_frames} 帧 (~{silence_frames * ms_per_frame:.1f}ms)，"
                            f"裁剪比例: {silence_ratio:.2%}"
                        )
                        refined_target_length = new_len
                        gen_mel = gen_mel[:, :new_len]
                    else:
                        ms_per_frame = hop_length * 1000.0 / self.infer_cfg.s2mel.preprocess_params.sr
                        self.logger.debug(
                            f"检测到微量尾部静音 "
                            f"{silence_frames} 帧 (~{silence_frames * ms_per_frame:.1f}ms, {silence_ratio:.2%})，"
                            f"未达到安全判定阈值，已予以保护保留"
                        )

            mel_mask_target = (
                sequence_mask(
                    torch.tensor([refined_target_length], device=self.device),
                    max_length=gen_mel.size(-1),
                )
                .unsqueeze(1)
                .to(torch.bool)
            )
            gen_mel = torch.where(
                mel_mask_target,
                gen_mel.unsqueeze(0),
                torch.tensor(-11.5129, device=self.device, dtype=gen_mel.dtype),
            )

            t_voc = time.perf_counter()
            wav = self.bigvgan(gen_mel.float()).squeeze(1)
            wav = torch.clamp(32767.0 * wav, -32767.0, 32767.0)
            t_voc_end = time.perf_counter()

        wav = wav[:, : refined_target_length * hop_length]

        return {
            "wav": wav,
            "cfm_sec": t_cfm_end - t_cfm,
            "bigvgan_sec": t_voc_end - t_voc,
        }
