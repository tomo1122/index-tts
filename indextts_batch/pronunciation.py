import gc
import json
import re
import shutil
import time
from pathlib import Path

from g2pw import G2PWConverter
from opencc import OpenCC

from indextts_batch.device_wrapper import (
    empty_cache,
    get_memory_info,
    get_onnx_providers,
)
from indextts_batch.utils import PACKAGE_ROOT, _RedirectStderr
from indextts_batch.utils import logger as _root_logger

logger = _root_logger.getChild(__name__)

# 台湾普通话发音差异修正词典
_TAIWAN_TO_MAINLAND_CORRECTIONS = {
    ("和", "HAN4"): "HE2",
    ("垃", "LE4"): "LA1",
    ("圾", "SE4"): "JI1",
    ("液", "YI4"): "YE4",
    ("企", "QI4"): "QI3",
    ("携", "XI1"): "XIE2",
    ("期", "QI2"): "QI1",
    ("括", "GUO1"): "KUO4",
    ("亚", "YA3"): "YA4",
    ("暂", "ZHAN4"): "ZAN4",
    ("微", "WEI2"): "WEI1",
    ("熟", "SHOU2"): "SHU2",
    ("框", "KUANG1"): "KUANG4",
    ("缩", "SU4"): "SUO1",
}


def _patch_g2pw_package(g2pw_model_dir: str | Path) -> None:
    """检测 pip 安装的 g2pw 包路径，并用本地大陆普通话字典覆盖其包内默认的台湾国语字典"""
    try:
        import g2pw

        g2pw_package_dir = Path(g2pw.__file__).parent
        local_model_dir = Path(g2pw_model_dir)

        target_files = [
            "char_bopomofo_dict.json",
            "bopomofo_to_pinyin_wo_tune_dict.json",
        ]

        patched = False
        for filename in target_files:
            src = local_model_dir / filename
            dst = g2pw_package_dir / filename

            if not src.exists():
                continue

            # 如果包内目标字典不存在，或者大小不一致，则进行覆盖替换
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
                patched = True

        if patched:
            logger.info("[Dirty Hack] 使用本地大陆普通话词表覆盖了全局 pip g2pw 包中的台湾国语词表")
    except Exception as e:
        logger.warning(f"[Dirty Hack] 自动替换全局 g2pw 包内词表失败: {e}")


class PronunciationModule:
    """发音处理模块，负责对原始文本应用发音层面的替换

    参数:
        g2pw_model_dir: g2pW ONNX 模型所在目录
        bert_model_dir: BERT 词元化模型所在目录
        polyphone_json_path: 多音字词典 JSON 路径
        batch_size: ONNX 推理批大小
    """

    def __init__(
        self,
        g2pw_model_dir: str,
        bert_model_dir: str,
        polyphone_json_path: str | Path,
        batch_size: int = 512,
    ):
        self.g2pw_model_dir = g2pw_model_dir
        self.bert_model_dir = bert_model_dir
        self.batch_size = batch_size
        self._root_dir = PACKAGE_ROOT  # indextts_batch/

        # 启动时完成全局 pip 环境的大陆词表对齐替换
        _patch_g2pw_package(self.g2pw_model_dir)

        # 加载多音字词典
        with open(polyphone_json_path, "r", encoding="utf-8") as f:
            poly_data = json.load(f)
        self.polyphone_chars = {item["char"] for item in poly_data}

        # 初始化 OpenCC 简繁转换器
        self.s2t_converter = OpenCC("s2t")

        self.conv = None  # g2pW 转换器，按需创建
        self._device = "cpu"  # 当前设备，to() 时更新

    def to(self, device: str) -> "PronunciationModule":
        """VRAMManager 接口：控制 ONNX 模型的资源生命周期

        参数:
            device: 目标设备（如 "cuda:0", "mps", "cpu"）
        """
        self._device = device
        if device.startswith("cpu"):
            self._unload()
        else:
            self._ensure_loaded()
        return self

    def _ensure_loaded(self) -> G2PWConverter:
        """创建 g2pW ONNX session（分配 GPU 显存）

        返回:
            已初始化的 G2PWConverter 实例
        """
        if self.conv is not None:
            return self.conv

        # 清空 PyTorch 缓存后获取基准显存，确保 ONNX 独占差值
        empty_cache(self._device)
        free_before, total = get_memory_info(self._device)
        logger.info("加载模型 (g2pW)")
        start_time = time.perf_counter()

        # 将 ONNX Runtime 的 C 级告警重定向到日志文件，避免污染终端
        with _RedirectStderr():
            self.conv = G2PWConverter(
                style="pinyin",
                enable_non_tradional_chinese=True,
                model_dir=self.g2pw_model_dir,
                model_source=self.bert_model_dir,
                batch_size=self.batch_size,
                num_workers=0,
                turnoff_tqdm=True,
            )
            providers = get_onnx_providers(self._device)
            self.conv.session_g2pw.set_providers(providers)

        elapsed = time.perf_counter() - start_time
        free_after, _ = get_memory_info(self._device)

        if total > 0:
            mem_used = max(0, free_before - free_after) / 1e9
            used_total = (total - free_after) / 1e9
            logger.info(
                f"模型加载完成，占用显存：{mem_used:.2f} GB, 当前已占用显存：{used_total:.2f} GB, 耗时：{elapsed:.2f}秒"
            )
        else:
            logger.info(f"模型加载完成（MPS 统一内存，无法统计显存），耗时：{elapsed:.2f}秒")
        return self.conv

    def _unload(self):
        """销毁 g2pW ONNX session（释放 GPU 显存）"""
        if self.conv is None:
            return
        del self.conv
        self.conv = None
        gc.collect()
        empty_cache(self._device)

    def process_all(self, texts: list[str]) -> list[str]:
        """对一批文本执行发音处理

        1. 预筛：对含多音字的文本进行高精度简繁转换后，送 GPU 推理
        2. 推理：g2pW 单次批处理
        3. 替换：将多音字替换为校准后的大写拼音 token

        参数:
            texts: 原始文本列表

        返回:
            处理后的文本列表，顺序与输入一致
        """
        # 步骤一：预筛
        logger.info("准备 g2pW 推理")
        t_start = time.perf_counter()
        filtered_indices: list[int] = []
        filtered_texts: list[str] = []
        for idx, text in enumerate(texts):
            if any(char in self.polyphone_chars for char in text):
                filtered_indices.append(idx)
                # 在送入推理前，将简体字转为繁体文本
                trad_text = self.s2t_converter.convert(text)
                filtered_texts.append(trad_text)

        if not filtered_texts:
            return list(texts)

        # 步骤二：g2pW 推理
        conv = self._ensure_loaded()
        preds = conv(filtered_texts)
        t_end = time.perf_counter()
        logger.info(f"g2pW 推理完成，共计 {len(filtered_texts)} 条文本, 耗时 {t_end - t_start:.2f}秒")

        # 步骤三：替换
        results = list(texts)
        debug_pairs = []
        for idx, sent_trad, sent_preds in zip(filtered_indices, filtered_texts, preds):
            modified = self._format_override(results[idx], sent_preds)
            results[idx] = modified
            if texts[idx] != modified:
                debug_pairs.append({"original": texts[idx], "modified": modified})

        if debug_pairs:
            debug_path = self._root_dir / "data" / "pronunciation_debug.json"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(debug_pairs, f, ensure_ascii=False, indent=2)
            logger.debug(f"发音处理调试信息已保存至 {debug_path}，共 {len(debug_pairs)} 条替换记录。")

        return results

    # ──────── 拼音校准 ────────────────────────────────────────
    @staticmethod
    def _correct_pinyin(pinyin: str) -> str:
        """jqx 后接 u/ü 转写为 v 形式并大写

        参数:
            pinyin: 原始拼音字符串（如 ju4, que3）

        返回:
            校准后的大写拼音字符串（如 JV4, QVE3）
        """
        if not pinyin:
            return ""
        if pinyin[0] not in "jqxJQX":
            return pinyin.upper()
        pattern = r"([jqx])[uü](n|e|an)*(\d)"
        repl = r"\g<1>v\g<2>\g<3>"
        pinyin = re.sub(pattern, repl, pinyin, flags=re.IGNORECASE)
        return pinyin.upper()

    def _format_override(self, sentence: str, pinyins: list[str | None]) -> str:
        """将句中的多音字替换为校准后的大写拼音 token

        参数:
            sentence: 原始汉游戏句子（简体）
            pinyins: g2pW 输出的 1-to-1 拼音列表（含 None）

        返回:
            替换后的文本，多音字位置为拼音 token
        """
        new_chars = []
        for char, py in zip(sentence, pinyins):
            if char in self.polyphone_chars and py is not None:
                py_upper: str = py.upper()
                corrected_py: str = _TAIWAN_TO_MAINLAND_CORRECTIONS.get((char, py_upper), py_upper)
                new_chars.append(f" {self._correct_pinyin(corrected_py)} ")
            else:
                new_chars.append(char)
        result = "".join(new_chars)
        return re.sub(r"\s+", " ", result).strip()
