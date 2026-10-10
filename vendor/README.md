# vendor —— 依赖逃生通道（非标准安装方式）

这个目录用于**pip 不可用的环境**：把纯 Python 依赖的 wheel 解压到这里，
`app/rag/vendor.py` 会在 `import` 失败时惰性把本目录加入 `sys.path`。

## 当前内容

| 包 | 版本 | 用途 |
| --- | --- | --- |
| `pypdf` | 6.20.0 | PDF 文本抽取（`app/rag/loaders.py::load_pdf`） |

## 正式环境请优先用 pip / uv

```bash
pip install -r requirements.txt
# 或
uv sync
```

装了正式依赖后，本目录不会被优先使用（`sys.path.append` 追加在末尾，标准 site-packages 优先级更高）。

## 什么时候才用它

本项目开发环境里遇到过两种「依赖装不上」的情况：

1. 系统 `site-packages` 目录对当前进程不可写；
2. `pip install` 进程空转（CPU 持续占用但没有任何网络连接），而 `urllib` 直连 PyPI 正常。

此时可以把 wheel 下载后解压到本目录：

```python
import json, urllib.request, zipfile, sysconfig
from pathlib import Path

pkg = "pypdf"
meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=30))
ver = meta["info"]["version"]
url = next(i["url"] for i in meta["releases"][ver] if i["filename"].endswith("-py3-none-any.whl"))
wheel = Path(".tmp_wheels") / url.split("/")[-1]
wheel.parent.mkdir(exist_ok=True)
urllib.request.urlretrieve(url, wheel)
with zipfile.ZipFile(wheel) as zf:
    zf.extractall("vendor")
```

## 限制

- **只放纯 Python 包**（`py3-none-any` 的 wheel）。带编译扩展的包（如 `torch`、`sentence-transformers`
  依赖链、`faiss-cpu`）不要这样装，容易出现二进制不匹配的问题。
- 不参与任何构建流程，纯粹是运行时 `sys.path` 兜底。
- 如果不需要 PDF 解析，可以直接删除整个目录，不影响其它格式（`.md` / `.txt` / `.docx` / `.csv`）的使用。
