import json
import os

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

import random
from pathlib import Path

import numpy as np
import soundfile
import torch

from indextts_batch.pipeline import IndexTTSBatch
from indextts_batch.utils import PACKAGE_ROOT, setup_global_logger
from indextts_batch.utils import logger as _root_logger

logger = _root_logger.getChild(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _save_wavs(wavs: dict, output_dir: Path, sampling_rate: int = 22050, prefix: str = "sentence") -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    for orig_idx, wav in wavs.items():
        output_file = Path(output_dir, f"{prefix}_{orig_idx + 1}.wav")
        soundfile.write(output_file, wav.type(torch.int16).numpy(), sampling_rate)


if __name__ == "__main__":
    setup_global_logger()
    is_cuda = torch.cuda.is_available()
    is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    # set_seed(42)

    # 从电子书加载文本测试
    _books_dir = PACKAGE_ROOT / "books"
    with open(_books_dir / "text00000_sentences.json", "r", encoding="utf-8") as f:
        _book = json.load(f)
    texts = []
    for key, value in _book.items():
        texts += value
    texts = texts[:100]  # 只取前100条句子
    logger.info(f"已加载 {len(texts)} 条句子")

    pipeline = IndexTTSBatch(
        device="cuda:0" if is_cuda else ("mps" if is_mps else "cpu"),
        cuda_memory_limit=0.85 if is_cuda else None,
    )

    speakers = "examples/hanser.wav"
    # [喜、怒、哀、惧、厌恶、低落、惊喜、平静]
    emo_vec = [0, 0, 0, 0, 0.35, 0, 0, 0]

    output_dir = Path("output")
    stream_save = True
    if stream_save:
        pipeline.generate(
            texts,
            speaker_audio=speakers,
            emo_mode="vector",
            emo_vector=emo_vec,
            max_tokens_per_batch=1500,
            output_dir=output_dir,
        )
    else:
        wavs_base = pipeline.generate(
            texts,
            speaker_audio=speakers,
            emo_mode="vector",
            emo_vector=emo_vec,
            max_tokens_per_batch=1500,
        )
        sr = pipeline.cfg.s2mel["preprocess_params"]["sr"]
        _save_wavs(wavs_base, output_dir, sampling_rate=sr, prefix="sentence")
