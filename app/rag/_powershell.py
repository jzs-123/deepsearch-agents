"""Windows PowerShell + WinRT 调用的共享基础设施(OCR 与 PDF 渲染共用)。

本模块存在有两个实测理由:

1. **用 `-EncodedCommand` 而不是 .ps1 文件。** PowerShell 5.1 会把**无 BOM 的 .ps1**
   当 ANSI 读,脚本里的中文注释会把解析搞崩(实测 `Unexpected token '}'`);
   执行策略还会额外拦 .ps1。把脚本编成 UTF-16LE base64 交给 `-EncodedCommand`
   一次同时解决编码与策略两个问题,调用方也不必落地临时脚本。

2. **子进程输出捕获可能被沙箱拒绝。** 本沙箱禁止程序打开命名管道,而 piped stdio 走的
   就是命名管道(`capture_output=True` 可能直接 `OSError`)。
   实测在本会话里 `powershell.exe` 的管道捕获是**可用**的,但为稳妥起见:
   `run_powershell_script` 先走管道,一旦抛 `OSError` 就**自动回落到
   「stdin/stdout/stderr 全部重定向到临时文件」**——完全不走管道,结果一样能拿到。
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile

# 不让每次调用都闪一个控制台黑框
CREATE_NO_WINDOW = 0x08000000


def encode_ps_command(script: str) -> str:
    """把 PowerShell 脚本编成 `-EncodedCommand` 需要的 UTF-16LE base64。"""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def powershell_executable() -> str | None:
    """找到 PowerShell 可执行文件(优先 Windows PowerShell 5.1)。"""
    return shutil.which("powershell.exe") or shutil.which("powershell")


def _mkstemp_path(suffix: str) -> str:
    """在系统临时目录建一个文件并**立刻关掉 fd**;临时目录不可用时回落到当前目录。"""
    try:
        fd, path = tempfile.mkstemp(suffix=suffix)
    except OSError:
        fd, path = tempfile.mkstemp(suffix=suffix, dir=os.getcwd())
    os.close(fd)
    return path


def run_powershell_script(
    script: str,
    stdin_text: str = "",
    *,
    timeout: float = 120.0,
) -> tuple[int, str, str]:
    """执行 PowerShell 脚本,返回 `(returncode, stdout, stderr)`。

    先尝试管道捕获;若沙箱拒绝管道(``OSError``),自动改用临时文件重定向重试。
    ``OSError`` 之外的异常(例如超时)不在此处理,交由调用方决定降级方式。
    """
    exe = powershell_executable()
    if exe is None:
        raise RuntimeError("找不到 powershell.exe(本功能仅 Windows 可用)")
    args = [exe, "-NoProfile", "-NonInteractive", "-EncodedCommand", encode_ps_command(script)]
    try:
        proc = subprocess.run(
            args,
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except OSError:
        # 沙箱拒绝命名管道 → 改用文件重定向(不碰管道)
        return _run_via_files(args, stdin_text, timeout)


def _run_via_files(args: list[str], stdin_text: str, timeout: float) -> tuple[int, str, str]:
    """不使用管道的执行路径:stdin/stdout/stderr 全部走临时文件。"""
    in_path = _mkstemp_path(".stdin.txt")
    out_path = _mkstemp_path(".stdout.txt")
    err_path = _mkstemp_path(".stderr.txt")
    try:
        with open(in_path, "w", encoding="utf-8") as fh:
            fh.write(stdin_text)
        with open(in_path, "r", encoding="utf-8") as fin, open(
            out_path, "w", encoding="utf-8"
        ) as fout, open(err_path, "w", encoding="utf-8") as ferr:
            proc = subprocess.run(
                args,
                stdin=fin,
                stdout=fout,
                stderr=ferr,
                timeout=timeout,
                creationflags=CREATE_NO_WINDOW,
            )
        with open(out_path, "r", encoding="utf-8", errors="replace") as fh:
            out = fh.read()
        with open(err_path, "r", encoding="utf-8", errors="replace") as fh:
            err = fh.read()
        return proc.returncode, out, err
    finally:
        for path in (in_path, out_path, err_path):
            try:
                os.unlink(path)
            except OSError:  # pragma: no cover - 清理失败不影响结果
                pass


__all__ = [
    "CREATE_NO_WINDOW",
    "encode_ps_command",
    "powershell_executable",
    "run_powershell_script",
]
