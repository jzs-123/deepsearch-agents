"""OCR 层:可插拔后端。

**为什么需要可插拔**:OCR 是「兜底」而不是主路径(见 docs/rag-ocr.md 的质量实测),
而且不同环境能用的引擎完全不同 ——
- `windows`:**零 pip 依赖**,走系统内置 `Windows.Media.Ocr`(方案 A,默认);
- `rapidocr`:装了 `rapidocr-onnxruntime` 时可用,跨平台;
- `none`:   显式关闭。

对上层只暴露两个东西:`available`(能不能用)与 `recognize(image_bytes) -> str`。
**后端不可用一律自动降级为 none 并 warning,绝不抛异常**——OCR 失败不该让整批入库失败。

调用方式与踩坑(均实测,详见 task 情报与 docs/rag-ocr.md):
- `OcrEngine.RecognizeAsync` / `BitmapDecoder.CreateAsync` / `DataWriter.StoreAsync` 都返回
  **IAsyncOperation<T>** → 用泛型 `AsTask<T>`;列表里的 `IAsyncAction` 重载不能混用;
- 走**内存流**比 `StorageFile.GetFileFromPathAsync` 稳(后者曾抛 AggregateException);
- 脚本经 `-EncodedCommand`(UTF-16LE base64)传入,规避 PS 5.1 把无 BOM .ps1 当 ANSI 读的问题;
- OCR 结果**汉字之间会插空格**(`normalize_text` 会清掉),但 `。`→`·`、`%`→`％` 这类
  字符混淆**不会被修**(属于识别误差,不是排版噪声)。
"""

from __future__ import annotations

import base64
import importlib.util
import logging
import subprocess
import sys
from dataclasses import dataclass, replace
from functools import lru_cache

from app.rag._powershell import powershell_executable, run_powershell_script

logger = logging.getLogger(__name__)

# 合法后端名(配置校验与 CLI 提示共用)
OCR_BACKENDS = ("windows", "rapidocr", "none")

# OCR 脚本:stdin = 图片的 base64;stdout = 识别出的纯文本(按行拼接)
_OCR_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$ProgressPreference = 'SilentlyContinue'
Add-Type -AssemblyName System.Runtime.WindowsRuntime

$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataWriter, Windows.Foundation, ContentType = WindowsRuntime]

# 只取泛型重载 AsTask<T>(IAsyncOperation<T>);把 IAsyncAction 当泛型转就是上一版的坑。
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
})[0]

function Await($op, $resultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($resultType)
    $netTask = $asTask.Invoke($null, @($op))
    $netTask.Wait(-1) | Out-Null
    return $netTask.Result
}

$lines = [Console]::In.ReadToEnd() -split "`n"
$b64 = $lines[0].Trim()
$langTag = if ($lines.Count -gt 1) { $lines[1].Trim() } else { 'zh-Hans-CN' }
$bytes = [Convert]::FromBase64String($b64)

$stream = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
$writer = New-Object Windows.Storage.Streams.DataWriter($stream)
$writer.WriteBytes($bytes)
Await ($writer.StoreAsync()) ([uint32]) | Out-Null
$writer.DetachStream() | Out-Null
$stream.Seek(0)

$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])

$engine = $null
try {
    $lang = New-Object Windows.Globalization.Language $langTag
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)
} catch {
    $engine = $null
}
if ($null -eq $engine) {
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
}
if ($null -eq $engine) {
    throw "no OCR engine available for language '$langTag' (Windows 可选功能里需要安装对应语言包)"
}

$res = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
$sb = New-Object System.Text.StringBuilder
foreach ($line in $res.Lines) { [void]$sb.AppendLine($line.Text) }
[Console]::Out.Write($sb.ToString())
"""


@dataclass(frozen=True)
class OcrOptions:
    """OCR 相关配置(与 app.core.config.Settings 一一对应,便于脱离全局单例测试)。"""

    enabled: bool = True
    backend: str = "windows"
    min_text_chars: int = 50
    language: str = "zh-Hans-CN"

    @classmethod
    def from_settings(cls) -> "OcrOptions":
        """从项目 Settings 读取;任何异常都回落到安全默认值(配置问题不该炸入库)。"""
        try:
            from app.core.config import settings

            return cls(
                enabled=bool(getattr(settings, "rag_ocr_enabled", True)),
                backend=str(getattr(settings, "rag_ocr_backend", "windows") or "windows"),
                min_text_chars=int(getattr(settings, "rag_ocr_min_text_chars", 50)),
                language=str(getattr(settings, "rag_ocr_language", "zh-Hans-CN") or "zh-Hans-CN"),
            )
        except Exception:  # pragma: no cover - 只在配置层异常时走到
            return cls()

    def with_overrides(self, **kwargs) -> "OcrOptions":
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean) if clean else self


class OcrEngine:
    """OCR 后端接口。

    :param calls: 被调用次数(用于「有文本层时不触发 OCR」这类断言取证)
    """

    name = "none"

    def __init__(self) -> None:
        self.calls = 0

    @property
    def available(self) -> bool:  # pragma: no cover - 接口默认值
        return False

    def recognize(self, image_bytes: bytes) -> str:
        """识别图片里的文字。实现约定:**失败返回空串并 warning,不抛业务异常**。"""
        raise NotImplementedError


class NoneOcrEngine(OcrEngine):
    """空后端:明确表示「不做 OCR」,所有调用返回空串。"""

    name = "none"

    @property
    def available(self) -> bool:
        return False

    def recognize(self, image_bytes: bytes) -> str:
        self.calls += 1
        return ""


@dataclass
class WindowsOcrEngine(OcrEngine):
    """Windows 内置 OCR(`Windows.Media.Ocr`),零 pip 依赖。

    :param language: BCP-47 语言标签,默认简体中文 `zh-Hans-CN`
    :param timeout: 单张图片识别超时秒数
    """

    language: str = "zh-Hans-CN"
    timeout: float = 120.0
    name = "windows"

    def __post_init__(self) -> None:
        OcrEngine.__init__(self)

    @property
    def available(self) -> bool:
        if sys.platform != "win32":
            return False
        return powershell_executable() is not None

    def recognize(self, image_bytes: bytes) -> str:
        self.calls += 1
        if not self.available:
            logger.warning("Windows OCR 不可用(仅 Windows + PowerShell),返回空文本")
            return ""
        if not image_bytes:
            return ""
        payload = f"{base64.b64encode(image_bytes).decode('ascii')}\n{self.language}\n"
        try:
            returncode, stdout, stderr = run_powershell_script(
                _OCR_SCRIPT, payload, timeout=self.timeout
            )
        except subprocess.TimeoutExpired:
            logger.warning("Windows OCR 超时(>%ss),该页按无文本处理", self.timeout)
            return ""
        except Exception as exc:  # pragma: no cover - 进程级异常
            logger.warning("Windows OCR 调用失败(%s),该页按无文本处理", exc)
            return ""
        if returncode != 0:
            logger.warning("Windows OCR 退出码 %s:%s", returncode, stderr.strip()[:300])
            return ""
        return stdout.strip()


@dataclass
class RapidOcrEngine(OcrEngine):
    """onnxruntime 版 RapidOCR(跨平台备选)。装了 `rapidocr-onnxruntime` 才可用。

    ⚠️ 本机未安装该包(方案 A 要求零依赖,且任务禁止装包),因此这条分支**未在本机实测**,
    只保证:未安装时 `available` 为 False、调用返回空串并 warning,不会影响链路。
    """

    language: str = "zh-Hans-CN"
    name = "rapidocr"

    def __post_init__(self) -> None:
        OcrEngine.__init__(self)
        self._engine = None

    @property
    def available(self) -> bool:
        return (
            importlib.util.find_spec("rapidocr_onnxruntime") is not None
            or importlib.util.find_spec("rapidocr") is not None
        )

    def _load(self):
        if self._engine is not None:
            return self._engine
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError:
            from rapidocr import RapidOCR  # type: ignore
        self._engine = RapidOCR()
        return self._engine

    def recognize(self, image_bytes: bytes) -> str:
        self.calls += 1
        if not self.available or not image_bytes:
            if not self.available:
                logger.warning("rapidocr 后端不可用(未安装 rapidocr-onnxruntime),返回空文本")
            return ""
        try:
            engine = self._load()
            output = engine(image_bytes)
            # rapidocr-onnxruntime 返回 (result, elapse);result 为 [[box, text, score], ...]
            result = output[0] if isinstance(output, tuple) else output
            if not result:
                return ""
            lines: list[str] = []
            for item in result:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    lines.append(str(item[1]))
            return "\n".join(lines).strip()
        except Exception as exc:
            logger.warning("rapidocr 识别失败(%s),返回空文本", exc)
            return ""


def build_ocr_engine(
    backend: str = "windows",
    language: str = "zh-Hans-CN",
    *,
    timeout: float = 120.0,
) -> OcrEngine:
    """按名字构造 OCR 后端;**不可用或未知时降级为 none 并 warning,不抛异常**。"""
    if backend == "windows":
        engine: OcrEngine = WindowsOcrEngine(language=language, timeout=timeout)
        if engine.available:
            return engine
        logger.warning("OCR 后端 windows 不可用(需要 Windows + PowerShell),降级为 none")
        return NoneOcrEngine()
    if backend == "rapidocr":
        engine = RapidOcrEngine(language=language)
        if engine.available:
            return engine
        logger.warning("OCR 后端 rapidocr 不可用(未安装 rapidocr-onnxruntime),降级为 none")
        return NoneOcrEngine()
    if backend in ("none", ""):
        return NoneOcrEngine()
    logger.warning("未知的 OCR 后端 %r,降级为 none", backend)
    return NoneOcrEngine()


@lru_cache(maxsize=8)
def _cached_engine(backend: str, language: str, enabled: bool) -> OcrEngine:
    if not enabled:
        return NoneOcrEngine()
    return build_ocr_engine(backend, language)


def default_ocr_engine(options: OcrOptions | None = None) -> OcrEngine:
    """按项目配置返回进程内共享的 OCR 后端(与 pipeline.get_pipeline 同风格)。"""
    opts = options or OcrOptions.from_settings()
    return _cached_engine(opts.backend, opts.language, opts.enabled)


def clear_ocr_engine_cache() -> None:
    """清掉进程内 OCR 后端缓存(测试与配置热变更用)。"""
    _cached_engine.cache_clear()


__all__ = [
    "OCR_BACKENDS",
    "OcrEngine",
    "NoneOcrEngine",
    "WindowsOcrEngine",
    "RapidOcrEngine",
    "OcrOptions",
    "build_ocr_engine",
    "default_ocr_engine",
    "clear_ocr_engine_cache",
]
