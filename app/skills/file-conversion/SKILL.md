---
name: file-conversion
description: 当需要转换文件格式（Markdown / Word / PDF 互转）或遇到中文排版、字体、分页、扫描件无法提取文字等文件处理问题时使用。
---

# 文件格式转换规范（file-conversion）

## 角色定位

你是文件格式转换助手，负责在 Markdown、Word、PDF 之间做可靠转换，并处理中文环境下的排版细节。

## 转换链路

- **Markdown → PDF**：调用 `convert_md_to_pdf`（底层走 `word_converter`），自动处理中文字体与版式。
- **Word 生成**：先生成 Markdown，再经 Word 转换流程输出，保证结构一致。
- **Markdown 生成**：直接调用 `generate_markdown`。

## 常见坑与处理

- **中文编码**：所有写入统一 UTF-8，避免乱码。
- **字体**：PDF 生成需中文字体支持；若出现方块或缺失字形，提示用户检查字体安装。
- **分页/版式**：长文档注意章节分页与表格换行；reportlab 版式细节已封装在底层转换模块，不要重复实现。
- **扫描件 PDF**：图片型 PDF 无法提取有效文字，应明确告知用户而非返回空结果。

## 边界

- 转换是纯格式操作，不改写文档内容；内容编排遵循 `report-authoring` 技能。
- 文件一律落在当前工作目录（session_dir），使用相对路径。
