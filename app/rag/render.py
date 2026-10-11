"""PDF 页渲染层:把 PDF 的指定页渲染成 PNG 字节。

**为什么单独一层**:OCR 后端只关心「给我图片字节」,不该关心图片从哪来;
渲染只关心「产出图片字节」,不该关心拿它去做什么。两者解耦后,将来把渲染器
换成 pdfium / PyMuPDF,或把 OCR 换成 rapidocr,都只需要替换其中一侧。

当前唯一实现走 **Windows 内置的 `Windows.Data.Pdf`**(方案 A:零 pip 依赖),
因此只在 Windows 上可用;其它平台 `available` 为 False,调用方据此降级。

调用方式与踩坑(均实测):
- `PdfDocument.LoadFromStreamAsync` 返回 **IAsyncOperation<PdfDocument>** → 用泛型 AsTask<T>;
- `PdfPage.RenderToStreamAsync` 返回 **IAsyncAction**(不是泛型!) → 必须用非泛型 AsTask,
  把它当 IAsyncOperation<T> 转换是上一版失败的根因,勿重踩;
- 走**内存流**比 `StorageFile.GetFileFromPathAsync` 稳(后者曾抛 AggregateException);
- 脚本通过 `-EncodedCommand`(UTF-16LE base64)传入:PowerShell 5.1 会把**无 BOM 的 .ps1**
  当 ANSI 读,中文注释会直接把脚本解析搞崩,`-EncodedCommand` 从根上绕开编码问题。
"""

from __future__ import annotations

import base64
import logging
import sys
from dataclasses import dataclass

from app.rag._powershell import powershell_executable, run_powershell_script

logger = logging.getLogger(__name__)

# 渲染脚本:stdin 第 1 行 = 页索引(0-based),第 2 行 = PDF 的 base64;stdout = PNG 的 base64。
_RENDER_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$ProgressPreference = 'SilentlyContinue'
Add-Type -AssemblyName System.Runtime.WindowsRuntime

$null = [Windows.Data.Pdf.PdfDocument, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Data.Pdf.PdfPage, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Data.Pdf.PdfPageRenderOptions, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataWriter, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Foundation, ContentType = WindowsRuntime]

# 泛型重载 AsTask<T>(IAsyncOperation<T>)
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
})[0]
# 非泛型重载 AsTask(IAsyncAction) —— RenderToStreamAsync 走这个
$asTaskAction = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction'
})[0]

function Await($op, $resultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($resultType)
    $netTask = $asTask.Invoke($null, @($op))
    $netTask.Wait(-1) | Out-Null
    return $netTask.Result
}

function AwaitAction($op) {
    $netTask = $asTaskAction.Invoke($null, @($op))
    $netTask.Wait(-1) | Out-Null
}

$lines = [Console]::In.ReadToEnd() -split "`n"
$pageIndex = [int]$lines[0].Trim()
$pdfBytes = [Convert]::FromBase64String($lines[1].Trim())
$scale = if ($lines.Count -gt 2) { [double]$lines[2].Trim() } else { 2.0 }

$stream = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
$writer = New-Object Windows.Storage.Streams.DataWriter($stream)
$writer.WriteBytes($pdfBytes)
Await ($writer.StoreAsync()) ([uint32]) | Out-Null
$writer.DetachStream() | Out-Null
$stream.Seek(0)

$doc = Await ([Windows.Data.Pdf.PdfDocument]::LoadFromStreamAsync($stream)) ([Windows.Data.Pdf.PdfDocument])
$pageCount = [int]$doc.PageCount
if ($pageIndex -lt 0 -or $pageIndex -ge $pageCount) {
    throw "page index $pageIndex out of range (0..$($pageCount-1))"
}

$page = $doc.GetPage([uint32]$pageIndex)
$opts = New-Object Windows.Data.Pdf.PdfPageRenderOptions
$opts.DestinationWidth = [uint32][math]::Max(1, [math]::Round($page.Size.Width * $scale))

$outStream = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
AwaitAction ($page.RenderToStreamAsync($outStream, $opts))

$size = [int]$outStream.Size
$reader = New-Object Windows.Storage.Streams.DataReader($outStream.GetInputStreamAt(0))
Await ($reader.LoadAsync([uint32]$size)) ([uint32]) | Out-Null
$buf = New-Object byte[] $size
$reader.ReadBytes($buf)
[Console]::Error.Write("PAGES=$pageCount`n")
[Console]::Out.Write([Convert]::ToBase64String($buf))
"""


class PdfRenderer:
    """PDF 页渲染器接口。不可用时 `available` 为 False。"""

    name = "none"

    @property
    def available(self) -> bool:  # pragma: no cover - 接口默认值
        return False

    def render_page(self, pdf_bytes: bytes, page_index: int, *, scale: float = 2.0) -> bytes:
        raise NotImplementedError


@dataclass
class WindowsPdfRenderer(PdfRenderer):
    """Windows 内置渲染器(`Windows.Data.Pdf`),零第三方依赖。

    :param timeout: 单页渲染超时秒数;大页面渲染较慢,默认给足
    """

    name = "windows"
    timeout: float = 180.0

    @property
    def available(self) -> bool:
        if sys.platform != "win32":
            return False
        return powershell_executable() is not None

    def render_page(self, pdf_bytes: bytes, page_index: int, *, scale: float = 2.0) -> bytes:
        if not self.available:
            raise RuntimeError("Windows 内置 PDF 渲染器不可用(仅 Windows + PowerShell 可用)")
        payload = f"{int(page_index)}\n{base64.b64encode(pdf_bytes).decode('ascii')}\n{float(scale)}\n"
        returncode, stdout, stderr = run_powershell_script(
            _RENDER_SCRIPT, payload, timeout=self.timeout
        )
        if returncode != 0:
            raise RuntimeError(f"PDF 渲染失败(exit={returncode}):{stderr.strip()[:400]}")
        data = base64.b64decode(stdout.strip())
        if not data.startswith(b"\x89PNG"):
            raise RuntimeError("PDF 渲染返回的不是 PNG 数据")
        return data


@dataclass
class NonePdfRenderer(PdfRenderer):
    """空渲染器:明确表示「本环境不会渲染」,调用方据此降级而不是抛异常。"""

    name = "none"

    @property
    def available(self) -> bool:
        return False

    def render_page(self, pdf_bytes: bytes, page_index: int, *, scale: float = 2.0) -> bytes:
        raise RuntimeError("当前没有可用的 PDF 渲染器")


def build_pdf_renderer(backend: str = "windows", *, timeout: float = 180.0) -> PdfRenderer:
    """按名字构造渲染器;不可用时返回 NonePdfRenderer 并给出 warning,不抛异常。"""
    if backend == "windows":
        renderer = WindowsPdfRenderer(timeout=timeout)
        if renderer.available:
            return renderer
        logger.warning("PDF 渲染后端 windows 不可用(需要 Windows + PowerShell),降级为 none")
        return NonePdfRenderer()
    if backend in ("none", ""):
        return NonePdfRenderer()
    logger.warning("未知的 PDF 渲染后端 %r,降级为 none", backend)
    return NonePdfRenderer()


__all__ = [
    "PdfRenderer",
    "NonePdfRenderer",
    "WindowsPdfRenderer",
    "build_pdf_renderer",
]
