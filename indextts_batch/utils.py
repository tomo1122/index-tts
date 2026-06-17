import atexit
import logging
import os
import re
import sys
import traceback
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

# 包根目录
PACKAGE_ROOT = Path(__file__).parent

# 创建日志目录
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file = open(LOG_DIR / "batch.log", "a", encoding="utf-8", buffering=1)

# 备份真实的 stdout 和 stderr
_real_stdout = sys.__stdout__ or sys.stdout
_real_stderr = sys.__stderr__ or sys.stderr

# 共享 Rich Console（Progress + 日志统一走 stderr，Rich 自动协调渲染顺序）
GLOBAL_CONSOLE = Console(file=_real_stderr)


# 终端 Formatter
class BeautifulConsoleFormatter(logging.Formatter):
    GRAY = "\033[90m"
    BLUE = "\033[1;34m"
    RESET = "\033[0m"

    # 日志级别的彩色配置
    LEVEL_COLORS = {
        logging.DEBUG: "\033[36m",  # 青色
        logging.INFO: "\033[32m",  # 绿色
        logging.WARNING: "\033[33m",  # 黄色
        logging.ERROR: "\033[31m",  # 红色
        logging.CRITICAL: "\033[1;31m",  # 粗体红
    }

    def format(self, record):
        # 格式化时间
        time_str = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        date_part, time_part = time_str.split(" ")
        colored_time = f"{self.GRAY}{date_part}{self.RESET} {self.BLUE}{time_part}{self.RESET}"

        # 格式化日志级别
        level_color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        colored_level = f"{level_color}{record.levelname.ljust(5)}{self.RESET}"

        # 暗灰色竖线分隔符
        divider = f" {self.GRAY}│{self.RESET} "

        # 拼接正文
        message = record.getMessage()
        formatted = f"{colored_time}{divider}{colored_level}{divider}{message}"

        # 异常与堆栈处理
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            formatted += f"\n{self.GRAY}{record.exc_text}{self.RESET}"
        if record.stack_info:
            formatted += f"\n{self.GRAY}{self.formatStack(record.stack_info)}{self.RESET}"

        return formatted


# 基础格式定义
_file_log_format = "%(asctime)s [%(levelname)s] %(message)s"
_file_datefmt = "%Y-%m-%d %H:%M:%S"
_file_fmt = logging.Formatter(_file_log_format, datefmt=_file_datefmt)


logger = logging.getLogger("batch")
logger.setLevel(logging.INFO)
logger.propagate = False


# 异常捕获钩子
def _custom_excepthook(exctype, value, tb):
    if issubclass(exctype, KeyboardInterrupt):
        sys.__excepthook__(exctype, value, tb)
        return

    err_msg = "".join(traceback.format_exception(exctype, value, tb))

    # 写入日志文件
    _log_file.write(f"\n────── CRITICAL UNHANDLED EXCEPTION ──────────────────────────────\n{err_msg}\n")
    _log_file.flush()

    # 写入物理屏幕
    _real_stderr.write(f"\n\033[1;31m [CRASH DETECTED]\033[0m\n\033[90m{err_msg}\033[0m")
    _real_stderr.flush()


# 注册资源释放与流恢复函数
def _cleanup():
    sys.stdout = _real_stdout
    sys.stderr = _real_stderr
    _log_file.close()


# ──────── C 级 stderr 重定向（用于 ONNX Runtime 等 C 扩展的告警捕获） ────────────
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class _RedirectStderr:
    """临时接管 C 级 stderr（fd 2），捕获 ONNX Runtime 的告警并写入日志文件"""

    def __init__(self, target_fd: int | None = None):
        self._target_fd = target_fd

    def __enter__(self):
        self._saved_fd = os.dup(2)

        if self._target_fd is not None:
            os.dup2(self._target_fd, 2)
            self._pipe_r = None
            return self

        self._pipe_r, pipe_w = os.pipe()
        os.dup2(pipe_w, 2)
        os.close(pipe_w)
        return self

    def __exit__(self, *args):
        os.dup2(self._saved_fd, 2)
        os.close(self._saved_fd)

        if self._pipe_r is None:
            return

        chunks = []
        while True:
            chunk = os.read(self._pipe_r, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        os.close(self._pipe_r)

        raw = b"".join(chunks)
        if raw:
            text = raw.decode("utf-16-le", errors="replace")
            clean = _ANSI_RE.sub("", text)
            _log_file.write(clean)
            _log_file.flush()


def setup_global_logger():
    """初始化日志系统：重定向控制台、配置 handler 与异常钩子"""
    global logger

    # 全局重定向，屏蔽控制台输出
    sys.stdout = _log_file
    sys.stderr = _log_file

    # Handler 1: 控制台（统一走 GLOBAL_CONSOLE，Rich 自动协调 Progress 与日志的输出顺序）
    _ch = RichHandler(
        console=GLOBAL_CONSOLE,
        show_time=True,
        show_path=False,
    )
    logger.addHandler(_ch)

    # Handler 2: 日志文件
    _fh = logging.FileHandler(LOG_DIR / "batch.log", encoding="utf-8")
    _fh.setFormatter(_file_fmt)
    logger.addHandler(_fh)

    # 配置 Root Logger（接收来自原始项目的 logging 输出，使其只去文件，不去屏幕）
    _root = logging.getLogger()
    _root.setLevel(logging.INFO)
    # 清理第三方库可能已经初始化好的控制台输出 Handler
    for h in _root.handlers[:]:
        _root.removeHandler(h)
    # 将全局 logging 仅导向日志文件
    _root_fh = logging.FileHandler(LOG_DIR / "batch.log", encoding="utf-8")
    _root_fh.setFormatter(_file_fmt)
    _root.addHandler(_root_fh)

    # 替换全局异常钩子
    sys.excepthook = _custom_excepthook

    # 注册资源释放与流恢复
    atexit.register(_cleanup)
