from __future__ import annotations

import gc
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal, cast

import sentencepiece as spm
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
)
from torch.amp.autocast_mode import autocast
from torch.nn.utils.rnn import pad_sequence

from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts_batch.device_wrapper import empty_cache, get_memory_allocated
from indextts_batch.modules import (
    AudioGenerator,
    GPTGenerator,
    QwenEmotionModule,
    ReferenceEncoder,
)
from indextts_batch.pronunciation import PronunciationModule
from indextts_batch.utils import GLOBAL_CONSOLE, PACKAGE_ROOT
from indextts_batch.utils import logger as _root_logger

logger = _root_logger.getChild(__name__)


# ──────── 显存管理器 ──────────────────────────────────────────────
class VRAMManager:
    """显存管理器，所有模块扁平注册，由调用者显式编排 load/unload"""

    def __init__(self, device: str, offload_device: str = "cpu"):
        self.device = device
        self.offload_device = offload_device
        self._modules: dict[str, Any] = {}
        self._loaded: set[str] = set()

    def register(self, name: str, module: Any) -> None:
        self._modules[name] = module

    def get(self, name: str) -> Any:
        if name not in self._modules:
            raise KeyError(f"模块 '{name}' 未注册")
        return self._modules[name]

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def load(self, *names: str) -> None:
        """加载指定模块到 GPU"""
        to_load = [n for n in names if n not in self._loaded]
        for name in to_load:
            mod = self._modules[name]
            if hasattr(mod, "ensure_loaded"):
                mod.ensure_loaded()
            if hasattr(mod, "to"):
                mod.to(self.device)
            self._loaded.add(name)
            logger.info(
                f"[VRAMManager] 将 {name} 加载至显存，当前显存占用: {(get_memory_allocated(self.device) / 1e9):.2f} GB"
            )

    def unload(self, *names: str, reason: str = "") -> None:
        """卸载指定模块到 CPU"""
        tag = f" ({reason})" if reason else ""
        for name in names:
            if name in self._loaded:
                self._modules[name].to(self.offload_device)
                self._loaded.discard(name)
                logger.info(
                    f"[VRAMManager] 将 {name} 卸载至内存{tag}，"
                    f"当前显存占用: {(get_memory_allocated(self.device) / 1e9):.2f} GB"
                )
        if names:
            gc.collect()
            empty_cache(self.device)

    def unload_all(self) -> None:
        self.unload(*list(self._loaded))

    def require_loaded(self, *names: str) -> None:
        """确保指定模块已加载到 GPU，未加载则自动加载"""
        to_load = [n for n in names if n not in self._loaded]
        if to_load:
            self.load(*to_load)


# ──────── IndexTTS 推理 ───────────────────────────────────────────
class IndexTTSBatch:
    def __init__(
        self,
        cfg_path: str = "checkpoints/config.yaml",
        model_dir: str = "checkpoints",
        device: str | None = None,
        cuda_memory_limit: float | None = None,
    ):
        # 确定推理设备
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda:0"
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        if device is None:
            logger.info(f"推理设备未指定, 使用: {self.device}")

        is_cuda = self.device.startswith("cuda")

        # 加载推理配置文件
        self._config_path = PACKAGE_ROOT / "config.yaml"
        self._config_mtime = 0.0
        self.infer_cfg: DictConfig = OmegaConf.create()
        self._load_config()

        accel_cfg = self.infer_cfg.get("acceleration", {})
        self.use_fp16 = accel_cfg.get("use_fp16", False) and is_cuda
        if not is_cuda:
            self.use_fp16 = False
        self.use_cuda_kernel = accel_cfg.get("use_cuda_kernel", False) and is_cuda
        self.use_deepspeed = accel_cfg.get("use_deepspeed", False) and is_cuda
        self.use_accel = False
        self.use_torch_compile = accel_cfg.get("use_torch_compile", False) and is_cuda
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        # 模型配置
        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = Path(model_dir)

        # 显存管理
        self.vram_manager = VRAMManager(device=self.device, offload_device="cpu")

        # 限制显存使用率
        if self.device.startswith("cuda") and cuda_memory_limit:
            device_idx = torch.device(self.device).index
            device_idx = device_idx if device_idx is not None else 0
            torch.cuda.set_per_process_memory_fraction(cuda_memory_limit, device=device_idx)
            logger.info(f"限制可申请显存为: {cuda_memory_limit * 100}%，避免windows使用共享GPU内存")

        # 注册模块到 VRAMManager
        # 提取参考音频特征的模块
        self.ref_encoder = ReferenceEncoder(
            cfg_path,
            model_dir,
            self.device,
            self.dtype,
            infer_cfg=self.infer_cfg,
            pipeline=self,
        )
        self.vram_manager.register("ref_encoder", self.ref_encoder)

        # 情感识别模块
        self.qwen_emo = QwenEmotionModule(
            cfg_path,
            model_dir,
            self.device,
            self.dtype,
            infer_cfg=self.infer_cfg,
            pipeline=self,
        )
        self.vram_manager.register("qwen_emo", self.qwen_emo)

        # GPT-2 生成模块
        self.gpt_gen = GPTGenerator(
            cfg_path,
            model_dir,
            self.device,
            self.dtype,
            use_accel=self.use_accel,
            use_deepspeed=self.use_deepspeed,
            infer_cfg=self.infer_cfg,
            pipeline=self,
        )
        self.vram_manager.register("gpt", self.gpt_gen)

        # 音频生成模块
        self.audio_gen = AudioGenerator(
            cfg_path,
            model_dir,
            self.device,
            self.dtype,
            use_cuda_kernel=self.use_cuda_kernel,
            use_torch_compile=self.use_torch_compile,
            infer_cfg=self.infer_cfg,
            pipeline=self,
        )
        self.vram_manager.register("audio", self.audio_gen)

        self.speakers: dict[str, dict] = {}
        self._bench_gpt_time = 0.0
        self._bench_audio_time = 0.0
        self._bench_audio_sec = 0.0
        self._bench_cfm_time = 0.0
        self._bench_bigvgan_time = 0.0
        self._bench_total_time = 0.0

        # 初始化文本分词器
        bpe_path = self.model_dir / self.cfg.dataset["bpe_model"]
        self.normalizer = TextNormalizer(enable_glossary=True)
        self.normalizer.load()
        self.tokenizer = TextTokenizer(str(bpe_path), self.normalizer)

        # 发音处理模块（多音字消歧等）
        self.pronunciation = None
        pron_cfg = self.infer_cfg.get("pronunciation", {})
        if pron_cfg.get("enable", False) and self.device.startswith("cuda"):
            g2pw_dir = str(PACKAGE_ROOT / pron_cfg.g2pw_model_dir)
            bert_dir = str(PACKAGE_ROOT / pron_cfg.bert_model_dir)
            poly_json = PACKAGE_ROOT / pron_cfg.polyphone_json_path
            sp = spm.SentencePieceProcessor()
            sp.Load(str(bpe_path))
            self.pronunciation = PronunciationModule(
                g2pw_dir,
                bert_dir,
                poly_json,
                batch_size=pron_cfg.get("batch_size", 512),
            )
            self.vram_manager.register("pronunciation", self.pronunciation)
            logger.info("初始化发音处理模块")
        else:
            logger.info("发音处理模块未启用")

    def _load_config(self) -> None:
        """读取配置文件并记录 mtime"""
        if not self._config_path.exists():
            raise FileNotFoundError(f"未找到推理配置文件 {self._config_path}")
        self.infer_cfg = cast(DictConfig, OmegaConf.load(str(self._config_path)))
        self._config_mtime = self._config_path.stat().st_mtime

    def _reload_config_if_needed(self) -> None:
        """检查配置文件是否变更，如有则热加载"""
        current_mtime = self._config_path.stat().st_mtime
        if current_mtime <= self._config_mtime:
            return
        new_cfg = cast(DictConfig, OmegaConf.load(str(self._config_path)))
        self.infer_cfg = new_cfg
        # 同步更新子模块的 config 引用
        self.gpt_gen.infer_cfg = new_cfg
        self.audio_gen.infer_cfg = new_cfg
        self._config_mtime = current_mtime
        logger.info("配置文件已热重载")

    # 保存说话人特征
    def save_speaker(self, audio_path: str, name: str | None = None) -> str:
        """提取说话人特征并缓存到 self.speakers"""
        name = name or audio_path
        if name in self.speakers:
            return name
        self.vram_manager.require_loaded("ref_encoder")
        features = self.ref_encoder(audio_path)
        for k, v in features.items():
            if isinstance(v, torch.Tensor):
                features[k] = v.cpu()
        self.speakers[name] = features
        return name

    # 构建后续输入给GPT推理的 items
    def build_items(
        self,
        texts: list[str],
        speaker: str,
        *,
        emo_vector: list[float] | None = None,
        emo_audio_prompt: str | None = None,
        emo_alpha: float = 1.0,
    ) -> list[dict]:
        if speaker not in self.speakers:
            raise ValueError(f"说话人 '{speaker}' 特征未缓存")
        self.vram_manager.require_loaded("ref_encoder")
        emo_mode = "vector" if emo_vector is not None else "audio"
        items = self.ref_encoder.prepare_emotion(
            texts,
            speaker,
            self.speakers,
            emo_mode=emo_mode,
            emo_vector=emo_vector or None,
            emo_alpha=emo_alpha,
            emo_audio_prompt=emo_audio_prompt,
        )
        items = self.ref_encoder.compute_emovecs(items, self.speakers)
        for item in items:
            item["text"] = texts[item["orig_idx"]]
        return items

    # 推理前校验 items
    def validate_items(self, items: list[dict]) -> list[str]:
        errors: list[str] = []
        for i, item in enumerate(items):
            if item.get("emovec") is None:
                errors.append(f"items[{i}]: 缺少 emovec")
            spk = item.get("speaker_index")
            if spk not in self.speakers:
                errors.append(f"items[{i}]: 说话人 '{spk}' 特征未缓存")
            else:
                for field in ("spk_cond_emb", "spk_cond_length", "style", "ref_mel"):
                    if field not in self.speakers[spk]:
                        errors.append(f"speakers['{spk}']: 缺少 {field}")
        return errors

    # 从 items 生成音频
    def infer_items(
        self,
        items: list[dict],
        output_dir: str | Path | None = None,
    ) -> dict[int, torch.Tensor]:
        errors = self.validate_items(items)
        if errors:
            raise ValueError("items 校验失败:\n  " + "\n  ".join(errors))
        self._reload_config_if_needed()
        self.vram_manager.require_loaded("gpt", "audio")
        if not self.device or not self.gpt_gen.gpt:
            raise RuntimeError("device 或 gpt_gen.gpt 未初始化")

        texts = [item["text"] for item in items]
        batching_cfg = self.infer_cfg.get("batching", {})
        max_tokens = int(batching_cfg.get("max_tokens_per_batch", 1500))
        max_text_tokens = int(batching_cfg.get("max_text_tokens_per_segment", 120))

        batches, _, _ = self._batch_items(items, texts, max_tokens, max_text_tokens)
        speaker_set = set(item["speaker_index"] for item in items)
        device_type = torch.device(self.device).type
        enabled = self.dtype is not None

        results_pool: dict[int, list[tuple[int, torch.Tensor]]] = {}
        total_items = len(items)
        sample_rate = self.cfg.s2mel.preprocess_params.sr

        with (
            torch.inference_mode(),
            autocast(device_type, enabled=enabled, dtype=self.dtype),
        ):
            for key in speaker_set:
                if "speech_conditioning_latent" not in self.speakers[key]:
                    feature = self.speakers[key]
                    spk_emb = feature["spk_cond_emb"].to(self.device)
                    cond_latent = self.gpt_gen.gpt.get_conditioning(
                        spk_emb.transpose(1, 2),
                        torch.tensor([spk_emb.shape[1]], device=self.device),
                    )
                    feature["speech_conditioning_latent"] = cond_latent.cpu()

            with Progress(
                TextColumn("推理进度"),
                BarColumn(complete_style="green", style="white"),
                TextColumn("{task.completed}/{task.total} 条"),
                TextColumn("{task.fields[stats]}"),
                console=GLOBAL_CONSOLE,
            ) as progress:
                task_id = progress.add_task("", total=total_items, stats="")
                for b_idx, batch in enumerate(batches):
                    progress.update(task_id, batch=f"{b_idx + 1}/{len(batches)}")
                    batch_result = self._safe_infer_batch(batch, progress, task_id)
                    for orig_idx, seg_list in batch_result.items():
                        if orig_idx not in results_pool:
                            results_pool[orig_idx] = []
                        results_pool[orig_idx].extend(seg_list)

        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            for orig_idx, seg_list in results_pool.items():
                seg_list.sort(key=lambda x: x[0])
                wav = torch.cat([seg[1] for seg in seg_list], dim=-1)
                sf.write(
                    str(output_path / f"{orig_idx}.wav"),
                    wav.type(torch.int16).cpu().numpy(),
                    sample_rate,
                )
            return {}

        final_results: dict[int, torch.Tensor] = {}
        for orig_idx, seg_list in results_pool.items():
            seg_list.sort(key=lambda x: x[0])
            wavs = [seg[1] for seg in seg_list]
            final_results[orig_idx] = torch.cat(wavs, dim=-1) if len(wavs) > 1 else wavs[0]
        return final_results

    # 分词 + 装箱
    def _batch_items(
        self,
        items: list[dict],
        texts: list[str],
        max_tokens_per_batch: int,
        max_text_tokens_per_segment: int,
    ) -> tuple[list[list[dict]], list[dict], list[dict]]:
        """对已填充 emovec 的 items 做 tokenize → 分段 → 按 token 排序装箱"""
        expanded_items: list[dict] = []
        segment_count = 0
        for item in items:
            tokens = self.tokenizer.tokenize(texts[item["orig_idx"]])
            if max_text_tokens_per_segment > 0 and len(tokens) > max_text_tokens_per_segment:
                segments = self.tokenizer.split_segments(tokens, max_text_tokens_per_segment)
                segment_count += 1
                logger.debug(f"单个文本过长，分段后: {''.join(segments[0])} | {''.join(segments[1])}")
                for seg_idx, seg_tokens in enumerate(segments):
                    new_item = dict(item)
                    ids = self.tokenizer.convert_tokens_to_ids(seg_tokens)
                    new_item["tensor"] = torch.tensor(ids, dtype=torch.long)
                    new_item["n_tokens"] = len(ids)
                    new_item["segment_index"] = seg_idx
                    new_item["n_segments"] = len(segments)
                    expanded_items.append(new_item)
            else:
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                item["tensor"] = torch.tensor(ids, dtype=torch.long)
                item["n_tokens"] = len(ids)
                item["segment_index"] = 0
                item["n_segments"] = 1
                expanded_items.append(item)
        logger.info(f"长文本分段完成，分段次数: {segment_count}，分段后文本条数: {len(expanded_items)}")

        # 按 token 长度排序 + 装箱
        expanded_items.sort(key=lambda x: x["n_tokens"])
        batches: list[list[dict]] = []
        current_batch: list[dict] = []
        current_batch_max_tensor_length = 0
        item_offset = 36  # 音频特征32 + 语速2 + 起止符2

        for item in expanded_items:
            item_len = item["n_tokens"]
            if item_len + item_offset > max_tokens_per_batch:
                raise ValueError(
                    f"当前文本占用token：{item_len + item_offset}，超过单批次最大token：{max_tokens_per_batch}"
                )
            current_batch_max_tensor_length = max(current_batch_max_tensor_length, item_len)
            test_cost = (len(current_batch) + 1) * (item_offset + current_batch_max_tensor_length)
            if test_cost <= max_tokens_per_batch:
                current_batch.append(item)
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [item]
                current_batch_max_tensor_length = item_len

        if current_batch:
            batches.append(current_batch)

        # 统计日志
        batch_stats = []
        for b_idx, batch in enumerate(batches):
            max_len = max(item["n_tokens"] for item in batch)
            item_count = len(batch)
            token_cost = item_count * (max_len + item_offset)
            batch_stats.append(
                {
                    "batch_idx": b_idx,
                    "item_count": item_count,
                    "max_token_len": max_len,
                    "total_token_cost": token_cost,
                }
            )
        logger.info("─" * 60)
        logger.info(f"总文本条数：{len(expanded_items)}, 单批次TOKEN额度：{max_tokens_per_batch}")
        logger.info(
            f"分为 {len(batches)} 个变长批次，平均 "
            f"{sum(s['total_token_cost'] for s in batch_stats) / len(batches):.1f}; "
            f"最大 {max(s['total_token_cost'] for s in batch_stats)}; "
            f"最小 {min(s['total_token_cost'] for s in batch_stats)}"
        )

        return batches, batch_stats, expanded_items

    # 单批次推理（GPT + 音频生成）
    def _infer_single_batch(
        self,
        batch: list[dict],
        progress: Progress,
        task_id: TaskID,
    ) -> dict[int, list[tuple[int, torch.Tensor]]]:
        speaker_audio_path = [item["speaker_index"] for item in batch]
        # 对齐文本 tensor
        batch_text_tokens = pad_sequence(
            [item["tensor"] for item in batch],
            batch_first=True,
            padding_value=self.cfg.gpt.stop_text_token,
        ).to(self.device)

        # 拼接 latent
        batch_speech_conditioning_latent = torch.cat(
            [self.speakers[key]["speech_conditioning_latent"] for key in speaker_audio_path],
            dim=0,
        ).to(self.device)

        # 拼接 emovec
        batch_emovec = torch.cat([item["emovec"] for item in batch], dim=0).to(self.device)

        t_gpt = time.perf_counter()
        try:
            gpt_output = self.gpt_gen(
                spk_cond_emb=None,
                emovec=batch_emovec,
                text_tokens=batch_text_tokens,
                speech_conditioning_latent=batch_speech_conditioning_latent,
            )
        finally:
            gpt = self.gpt_gen.gpt
            if gpt is not None and getattr(gpt, "inference_model", None) is not None:
                if hasattr(gpt.inference_model, "clear_mel_emb"):
                    gpt.inference_model.clear_mel_emb()
        t_gpt_end = time.perf_counter()

        # 音频阶段：按说话人组装原始特征 List
        spk_cond_embs = [self.speakers[key]["spk_cond_emb"] for key in speaker_audio_path]
        ref_mels = [self.speakers[key]["ref_mel"] for key in speaker_audio_path]
        styles = [self.speakers[key]["style"] for key in speaker_audio_path]

        t_audio = time.perf_counter()
        all_wavs: list[torch.Tensor | None] = [None] * len(batch)

        for i in range(len(batch)):
            single_spk_cond_emb = spk_cond_embs[i].to(self.device)
            single_ref_mel = ref_mels[i].to(self.device)
            single_style = styles[i].to(self.device)

            result = self.audio_gen(
                latent=gpt_output["latent"][[i]],
                fixed_codes=gpt_output["fixed_codes"][[i]],
                code_lens=gpt_output["code_lens"][[i]],
                spk_cond_emb=single_spk_cond_emb,
                ref_mel=single_ref_mel,
                style=single_style,
            )
            if not isinstance(result["wav"], torch.Tensor):
                raise RuntimeError(f"AudioGenerator 未返回合法的音频 Tensor, 得到 {type(result['wav'])}")
            self._bench_cfm_time += result["cfm_sec"]
            self._bench_bigvgan_time += result["bigvgan_sec"]
            mel_len = int(gpt_output["code_lens"][i].item() * self.infer_cfg.s2mel.mel_scale)
            wav_len = mel_len * self.infer_cfg.s2mel.preprocess_params.hop_length
            all_wavs[i] = result["wav"].cpu()[0, :wav_len]
            progress.update(task_id, advance=1)
        t_audio_end = time.perf_counter()
        code_lens = gpt_output["code_lens"].cpu()

        # 计算本批次音频总时长与 RTF
        total_wav_samples = 0
        for b_idx in range(len(batch)):
            mel_len = int(code_lens[b_idx].item() * self.infer_cfg.s2mel.mel_scale)
            total_wav_samples += mel_len * self.infer_cfg.s2mel.preprocess_params.hop_length
        audio_sec = total_wav_samples / self.infer_cfg.s2mel.preprocess_params.sr
        gpt_sec = t_gpt_end - t_gpt
        audio_gen_sec = t_audio_end - t_audio
        total_sec = t_audio_end - t_gpt
        rtf = total_sec / audio_sec if audio_sec > 0 else float("inf")
        progress.update(task_id)
        progress.refresh()
        if rtf > 0 and rtf != float("inf"):
            logger.info(
                f"[统计] "
                f"GPT解码耗时 {gpt_sec:6.2f}s | "
                f"音频生成耗时 {audio_gen_sec:6.2f}s | "
                f"生成音频总计 {audio_sec:7.2f}s | "
                f"RTF={rtf:5.3f}"
            )
        else:
            logger.info(f"[统计] GPT解码 {gpt_sec:.2f}s | 音频生成 {audio_gen_sec:.2f}s")

        # 累计全局统计
        self._bench_gpt_time += gpt_sec
        self._bench_audio_time += audio_gen_sec
        self._bench_audio_sec += audio_sec

        batch_results: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for b_idx, item in enumerate(batch):
            orig_idx = item["orig_idx"]
            if orig_idx not in batch_results:
                batch_results[orig_idx] = []
            wav = all_wavs[b_idx]
            if wav is None:
                raise RuntimeError(f"索引为 {b_idx} 的音频生成失败，wav 为 None")
            batch_results[orig_idx].append((item["segment_index"], wav))
        return batch_results

    # 带 OOM 恢复的批次推理
    def _safe_infer_batch(
        self,
        batch: list[dict],
        progress: Progress,
        task_id: TaskID,
        depth: int = 0,
    ) -> dict[int, list[tuple[int, torch.Tensor]]]:
        oom_error = False
        try:
            return self._infer_single_batch(batch, progress, task_id)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom_error = True
            else:
                raise e

        if oom_error:
            logger.warning(f"[pipeline] 显存溢出，当前层级：{depth}；桶大小：{len(batch)}")
            progress.update(task_id, stats="OOM!")
            gc.collect()
            empty_cache(self.device)

            if len(batch) <= 1:
                logger.error(
                    f"[pipeline] 单句推理也爆显存：{batch[0]['n_tokens']} tokens；索引：{batch[0]['orig_idx']}"
                )
                return {}

            mid = len(batch) // 2
            res: dict[int, list[tuple[int, torch.Tensor]]] = {}
            res1 = self._safe_infer_batch(batch[:mid], progress, task_id, depth + 1)
            res2 = self._safe_infer_batch(batch[mid:], progress, task_id, depth + 1)
            for k, v in res1.items():
                res.setdefault(k, []).extend(v)
            for k, v in res2.items():
                res.setdefault(k, []).extend(v)
            return res
        return {}

    def generate(
        self,
        texts: list[str],
        speaker_audio: list[str] | str,
        max_tokens_per_batch: int | None = None,
        *,
        max_text_tokens_per_segment: int | None = None,
        emo_mode: Literal["vector", "text", "audio"] = "audio",
        emo_alpha: float = 1.0,  # 情感权重
        emo_vector: list[float] | None = None,  # 情感向量
        emo_text: str | None = None,  # 情感文本
        emo_audio_prompt: str | None = None,  # 情感音频
        output_dir: str | Path | None = None,  # 输出目录，传入则逐批次保存 WAV 到磁盘
    ):
        if not texts:
            raise ValueError("未提供待生成的文本")
        # 热加载配置文件
        self._reload_config_if_needed()
        batching_cfg = self.infer_cfg.get("batching", {})
        max_tokens_per_batch = (
            batching_cfg.get("max_tokens_per_batch", 1500) if max_tokens_per_batch is None else max_tokens_per_batch
        )
        max_text_tokens_per_segment = (
            batching_cfg.get("max_text_tokens_per_segment", 120)
            if max_text_tokens_per_segment is None
            else max_text_tokens_per_segment
        )
        if max_text_tokens_per_segment is None:
            raise RuntimeError("max_text_tokens_per_segment 未能从配置文件或参数中确定")
        if max_tokens_per_batch is None:
            raise RuntimeError("max_tokens_per_batch 未能从配置文件或参数中确定")

        # Phase A: 多音字发音处理
        if self.pronunciation is not None:
            n_orig = len(texts)
            self.vram_manager.load("pronunciation")
            processed = self.pronunciation.process_all(texts)
            modified = sum(1 for a, b in zip(texts, processed) if a != b)
            texts = processed
            logger.info(f"发音处理完成：{n_orig} 条文本，{modified} 条被修改")
            self.vram_manager.unload("pronunciation", reason="Phase A 结束")

        # Phase B: 情感提取 → 分词 → 装箱
        # 提取说话人/情感音频特征
        self.vram_manager.load("ref_encoder")
        items = self.ref_encoder.prepare_emotion(
            texts,
            speaker_audio,
            self.speakers,
            emo_mode=emo_mode,
            emo_vector=emo_vector,
            emo_alpha=emo_alpha,
            emo_audio_prompt=emo_audio_prompt,
        )

        # Qwen 文本情感推理（可选）
        if emo_mode == "text":
            self.vram_manager.load("qwen_emo")
            qwen_needed = {emo_text} if emo_text else set(texts)
            qwen_results: dict[str, list[float]] = self.vram_manager.get("qwen_emo")(qwen_needed)
            for item in items:
                text_key = emo_text or texts[item["orig_idx"]]
                item["emo_vector"] = qwen_results[text_key]
            self.vram_manager.unload("qwen_emo", reason="Qwen 推理完成")

        # 计算 emovec
        items = self.ref_encoder.compute_emovecs(items, self.speakers)
        self.vram_manager.unload("ref_encoder", reason="Phase B 结束")

        # tokenize → 分段 → 装箱
        batches, batch_stats, indexed_items = self._batch_items(
            items,
            texts,
            max_tokens_per_batch,
            max_text_tokens_per_segment,
        )

        # Phase C: GPT 自回归 + 音频生成
        self.vram_manager.load("gpt", "audio")
        speaker_set = {item["speaker_index"] for item in indexed_items}
        if not self.device or not self.gpt_gen.gpt:
            raise RuntimeError("device 或 gpt_gen.gpt 未初始化")
        device_type = torch.device(self.device).type
        enabled = self.dtype is not None
        # 累计全局统计
        self._bench_gpt_time = 0.0
        self._bench_audio_time = 0.0
        self._bench_audio_sec = 0.0
        self._bench_cfm_time = 0.0
        self._bench_bigvgan_time = 0.0
        self._bench_total_time = 0.0
        t_total_start = time.perf_counter()
        results_pool: dict[int, list[tuple[int, torch.Tensor]]] = {}
        total_items = sum(s["item_count"] for s in batch_stats)

        # 输出目录准备
        sample_rate = self.cfg.s2mel.preprocess_params.sr
        total_segments = Counter(item["orig_idx"] for item in indexed_items)
        output_path: Path | None = None

        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        device_type = torch.device(self.device).type
        enabled = self.dtype is not None
        with (
            torch.inference_mode(),
            autocast(device_type, enabled=enabled, dtype=self.dtype),
        ):
            # 计算每个说话人的 speech_conditioning_latent
            for key in speaker_set:
                if "speech_conditioning_latent" not in self.speakers[key]:
                    feature = self.speakers[key]
                    spk_emb = feature["spk_cond_emb"].to(self.device)
                    cond_latent = self.gpt_gen.gpt.get_conditioning(
                        spk_emb.transpose(1, 2),
                        torch.tensor([spk_emb.shape[1]], device=self.device),
                    )
                    feature["speech_conditioning_latent"] = cond_latent.cpu()

            with Progress(
                TextColumn("批次 {task.fields[batch]}"),
                BarColumn(complete_style="green", style="white"),
                TextColumn("{task.completed}/{task.total} 条"),
                TextColumn("{task.fields[stats]}"),
                console=GLOBAL_CONSOLE,
            ) as progress:
                task_id = progress.add_task("", total=total_items, batch="", stats="")
                for b_idx, batch in enumerate(batches):
                    progress.update(task_id, batch=f"{b_idx + 1}/{len(batches)}")
                    batch_result = self._safe_infer_batch(
                        batch,
                        progress,
                        task_id,
                    )
                    for orig_idx, seg_list in batch_result.items():
                        if orig_idx not in results_pool:
                            results_pool[orig_idx] = []
                        results_pool[orig_idx].extend(seg_list)

                    # 逐批次保存已完成全部片段的结果
                    if output_path is not None:
                        completed = []
                        for orig_idx, seg_list in results_pool.items():
                            if len(seg_list) == total_segments[orig_idx]:
                                seg_list.sort(key=lambda x: x[0])
                                wav = torch.cat([seg[1] for seg in seg_list], dim=-1)
                                sf.write(
                                    str(output_path / f"{orig_idx}.wav"),
                                    wav.type(torch.int16).cpu().numpy(),
                                    sample_rate,
                                )
                                completed.append(orig_idx)
                        for orig_idx in completed:
                            del results_pool[orig_idx]

        t_total_end = time.perf_counter()
        self._bench_total_time = t_total_end - t_total_start
        total_time = self._bench_total_time
        overall_rtf = total_time / self._bench_audio_sec if self._bench_audio_sec > 0 else float("inf")
        logger.info(
            f"[全局统计] GPT解码总耗时 {self._bench_gpt_time:.2f}s | "
            f"音频生成总耗时 {self._bench_audio_time:.2f}s "
            f"(CFM={self._bench_cfm_time:.2f}s + "
            f"BigVGAN={self._bench_bigvgan_time:.2f}s) | "
            f"生成音频总计 {self._bench_audio_sec:.2f}s | "
            f"端到端耗时 {total_time:.2f}s | "
            f"整体RTF={overall_rtf:.4f}"
        )

        if output_path is not None:
            for orig_idx, seg_list in results_pool.items():
                seg_list.sort(key=lambda x: x[0])
                wav = torch.cat([seg[1] for seg in seg_list], dim=-1)
                sf.write(
                    str(output_path / f"{orig_idx}.wav"),
                    wav.type(torch.int16).cpu().numpy(),
                    sample_rate,
                )
            final_results: dict[int, torch.Tensor] = {}
        else:
            final_results: dict[int, torch.Tensor] = {}
            for orig_idx, seg_list in results_pool.items():
                seg_list.sort(key=lambda x: x[0])
                wavs = [seg[1] for seg in seg_list]
                if len(wavs) == 1:
                    final_results[orig_idx] = wavs[0]
                else:
                    final_results[orig_idx] = torch.cat(wavs, dim=-1)

        return final_results
