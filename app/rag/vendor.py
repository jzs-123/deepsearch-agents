"""
依赖兜底路径(vendor)

用途:当运行环境**无法使用 pip** 时,可以把纯 Python 依赖的 wheel 解压到项目根的
`vendor/` 目录,本模块负责把它加进 sys.path,让 `import pypdf` 之类的语句仍能成功。

为什么要单独做一个模块而不是在代码里到处 sys.path.insert:
1. 只有一个地方处理路径,行为可预期;
2. 只在真正需要时才插入(惰性),不污染不需要这些依赖的场景(如只用 hash 向量时);
3. 显式记录「这里有非标准的依赖加载方式」,避免后来者误以为依赖已正常安装。

正式环境仍然推荐用 pip/uv 安装依赖。vendor 目录是为「pip 不可用」准备的逃生通道。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# 项目根目录下的 vendor 目录:app/rag/vendor.py -> parents[2]
VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor"


def ensure_vendor_path() -> bool:
    """把 vendor 目录加入 sys.path(若存在且尚未加入)。

    :return: vendor 目录是否可用
    """
    if not VENDOR_DIR.is_dir():
        return False
    vendor_str = str(VENDOR_DIR)
    if vendor_str not in sys.path:
        sys.path.append(vendor_str)
        logger.debug("已启用 vendor 依赖目录:%s", vendor_str)
    return True


__all__ = ["ensure_vendor_path", "VENDOR_DIR"]
