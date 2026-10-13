"""有界的数据库查询：固定口径、校验 SQL、执行一次并保存原始结果。"""

import asyncio
import csv
import json
import re
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field
from sqlglot import exp, parse, parse_one
from sqlglot.optimizer.qualify import qualify

from app.api.context import get_session_context
from app.api.monitor import monitor
from app.tools.db_tools import execute_database_query, get_database_schema
from app.utils.sql_guard import assert_readonly_sql


class QueryOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str
    direction: Literal["asc", "desc"]


class QueryJoin(BaseModel):
    model_config = ConfigDict(extra="forbid")
    condition: str
    kind: Literal["inner", "left", "right"] = "inner"


class QueryContract(BaseModel):
    """从原始问题提取一次，SQL 纠错时不能修改此口径。表达式使用真实表名.列名。"""
    model_config = ConfigDict(extra="forbid")
    tables: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    joins: list[QueryJoin] = Field(default_factory=list)
    where: str = ""
    group_by: list[str] = Field(default_factory=list)
    having: str = ""
    order_by: list[QueryOrder] = Field(default_factory=list)
    limit: int | None = Field(default=None, gt=0, le=10000)
    distinct: bool = False
    response_mode: Literal["table", "analysis"] = "table"
    clarification: str = ""


class SQLDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sql: str


def validate_contract(question: str, contract: QueryContract, schema: dict):
    """药品库存明确口径独立核对原文，不能把模型误读的阈值固定下来。"""
    if "quantity_on_hand" not in schema.get("inventory", {}):
        return
    match = re.search(r"库存(?:量)?\s*(大于等于|小于等于|不小于|不大于|至少|至多|大于|小于|超过|等于|>=|<=|>|<|=)\s*(-?\d+(?:\.\d+)?)", question)
    operators = {"大于等于": ">=", "不小于": ">=", "至少": ">=", "小于等于": "<=", "不大于": "<=", "至多": "<=", "大于": ">", "超过": ">", "小于": "<", "等于": "="}
    aggregate = any(word in question for word in ("总库存", "汇总", "合计", "求和"))
    if match and not aggregate:
        expected = parse_one(f"inventory.quantity_on_hand {operators.get(match[1], match[1])} {match[2]}", read="mysql")
        actual = parse_one(contract.where, read="mysql") if contract.where else None
        parts = list(actual.flatten()) if isinstance(actual, exp.And) else [actual]
        if not any(part == expected for part in parts):
            raise ValueError("查询口径未保留原始库存阈值和比较符")
    order = re.search(r"按\s*库存(?:量)?\s*(升序|降序|从小到大|从大到小)", question)
    if order and not aggregate:
        direction = "asc" if order[1] in ("升序", "从小到大") else "desc"
        if not contract.order_by or contract.order_by[0].expression != "inventory.quantity_on_hand" or contract.order_by[0].direction != direction:
            raise ValueError("查询口径未保留原始库存排序")
    requested = re.search(r"(?:列出|返回|展示|显示)(.*)", question)
    if requested:
        fields = {"药品名称": "drugs.generic_name", "批次号": "inventory.batch_number", "仓库位置": "inventory.warehouse_location", "过期日期": "inventory.expiry_date"}
        for label, column in fields.items():
            if label in requested[1] and column not in contract.columns:
                raise ValueError(f"查询口径遗漏要求字段：{label}")
        if "批次号" in requested[1] and not aggregate and contract.group_by:
            raise ValueError("要求批次明细，不得改成药品汇总")


def validate_query(sql: str, contract: QueryContract, schema: dict) -> str:
    """比较语法树，避免条件、字段、排序或统计粒度在 SQL 生成阶段漂移。"""
    try:
        if not contract.tables or not contract.columns:
            raise ValueError("查询口径必须包含真实表和返回字段")
        trees = parse(assert_readonly_sql(sql), read="mysql")
        if len(trees) != 1 or not isinstance(trees[0], exp.Select):
            raise ValueError("只能执行一条 SELECT 查询")
        tree = trees[0]
        # shortcut: 单层 SELECT/JOIN/GROUP BY；复杂子查询须单独扩展口径校验后放行。
        if len(list(tree.find_all(exp.Select))) != 1 or tree.args.get("with_"):
            raise ValueError("暂不支持子查询或 CTE，请拆分需求")
        if tree.args.get("into") or tree.args.get("locks"):
            raise ValueError("禁止导出服务器文件或加锁")
        if any(t.db or t.catalog or t.name not in schema for t in tree.find_all(exp.Table)):
            raise ValueError("表不存在或不属于当前业务库")
        if any(isinstance(n, (exp.Anonymous, exp.Window)) for n in tree.walk()):
            raise ValueError("暂不支持自定义函数或窗口函数")
        tree = qualify(tree, dialect="mysql", schema=schema, quote_identifiers=False, identify=False)
        aliases = {t.alias_or_name.lower(): t.name.lower() for t in tree.find_all(exp.Table)}
        output_aliases = {n.alias: n.this for n in tree.expressions if isinstance(n, exp.Alias)}

        def canonical(node):
            if node is None:
                return ""
            if isinstance(node, str):
                node = parse_one(node, read="mysql")
            node = node.copy()
            if isinstance(node, exp.Alias):
                node = node.this
            node = node.transform(lambda n: output_aliases[n.name].copy() if isinstance(n, exp.Column) and not n.table and n.name in output_aliases else n)
            for column in node.find_all(exp.Column):
                column.set("table", exp.to_identifier(aliases.get(column.table.lower(), column.table.lower())))
            return node.sql(dialect="mysql", normalize=True)

        def conditions(node):
            if node is None or node == "":
                return []
            if isinstance(node, str):
                node = parse_one(node, read="mysql")
            if isinstance(node, (exp.Where, exp.Having)):
                node = node.this
            parts = node.flatten() if isinstance(node, exp.And) else [node]
            return sorted(canonical(part) for part in parts)

        if sorted(t.name for t in tree.find_all(exp.Table)) != sorted(contract.tables):
            raise ValueError("查询表与原始口径不一致")
        projections = {canonical(n) for n in tree.expressions}
        if any(canonical(column) not in projections for column in contract.columns):
            raise ValueError("缺少要求返回的字段或聚合表达式")
        if conditions(tree.args.get("where")) != conditions(contract.where):
            raise ValueError("WHERE 筛选条件与原始口径不一致")
        if conditions(tree.args.get("having")) != conditions(contract.having):
            raise ValueError("HAVING 聚合筛选与原始口径不一致")
        groups = tree.args.get("group")
        if sorted(canonical(n) for n in (groups.expressions if groups else [])) != sorted(canonical(n) for n in contract.group_by):
            raise ValueError("分组粒度与原始口径不一致")
        joins = tree.args.get("joins") or []
        if any(not j.args.get("on") for j in joins) or sorted((canonical(j.args.get("on")), (j.args.get("side") or "inner").lower()) for j in joins) != sorted((canonical(n.condition), n.kind) for n in contract.joins):
            raise ValueError("表关联条件与原始口径不一致")
        order = tree.args.get("order")
        actual_order = order.expressions if order else []
        for index, required in enumerate(contract.order_by):
            if index >= len(actual_order) or canonical(actual_order[index].this) != canonical(required.expression) or bool(actual_order[index].args.get("desc")) != (required.direction == "desc"):
                raise ValueError("排序字段或方向与原始口径不一致")
        limit = tree.args.get("limit")
        if (int(limit.expression.this) if limit else None) != contract.limit or tree.args.get("offset"):
            raise ValueError("不得自行截断或跳过记录")
        if bool(tree.args.get("distinct")) != contract.distinct:
            raise ValueError("去重口径不一致")
        if not contract.group_by and any(isinstance(n, exp.AggFunc) for n in tree.walk()) and not any(isinstance(n, exp.AggFunc) for c in contract.columns for n in parse_one(c, read="mysql").walk()):
            raise ValueError("明细查询不得改成聚合查询")
        return tree.sql(dialect="mysql")
    except Exception as exc:
        raise ValueError(f"SQL 校验失败：{exc}") from exc


@dataclass
class DatabaseRun:
    question: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    result: str | None = None
    answer: str = ""
    artifact: str = ""
    row_count: int = 0
    status: str = "failed"
    response_mode: str = "table"


database_run: ContextVar[DatabaseRun | None] = ContextVar("database_run", default=None)


async def _structured(schema, messages):
    from app.agent.llm import model
    from app.core.config import settings
    if urlparse(settings.openai_base_url).hostname == "api.deepseek.com":
        model = model.model_copy(update={"extra_body": {**(model.extra_body or {}), "thinking": {"type": "disabled"}}})
    return await model.with_structured_output(schema, method="function_calling").ainvoke(messages)


def _result_files(scope, sql, columns, rows, truncated):
    session_dir = get_session_context()
    if not session_dir:
        raise ValueError("缺少会话目录，无法保存数据库结果")
    filename = f"database-{uuid.uuid4().hex[:8]}.csv"
    with (Path(session_dir) / filename).open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)
    scope.artifact = filename
    scope.row_count = len(rows)
    scope.status = "partial" if truncated else "success"

    def cell(value):
        return ("" if value is None else str(value)).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    def table(records):
        return "\n".join(["| " + " | ".join(map(cell, columns)) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"] + ["| " + " | ".join(map(cell, row)) + " |" for row in records])

    count = f"查询返回 {len(rows)} 条记录。" if not truncated else f"结果超过 10000 条，仅导出前 {len(rows)} 条，尚未核验总数。请缩小筛选范围。"
    scope.answer = count + "\n\n" + table(rows) + f"\n\n已导出结果：`{filename}`。"
    return json.dumps({"status": scope.status, "row_count": len(rows), "truncated": truncated,
        "sql": sql, "columns": columns, "artifact": filename,
        "preview": table(rows[:10]), "preview_rows": min(10, len(rows)),
        "instruction": "查询已完成。预览不是全部数据；完整数据在 CSV 文件中，不要重新查询剩余记录或重数预览。使用核验条数，引用现有文件。"}, ensure_ascii=False)


async def run_database_query(state):
    scope = database_run.get() or DatabaseRun(state["messages"][-1].text)
    async with scope.lock:
        if scope.result is None:
            try:
                schema = await asyncio.to_thread(get_database_schema)
                contract = await _structured(QueryContract, [SystemMessage(content=
                    "从原始需求提取不可变查询口径，不生成 SQL。表/列必须来自结构信息；表达式必须用真实表名.列名，不用别名。"
                    "WHERE/HAVING 填完整 SQL 条件；joins 填 ON 条件及 inner/left/right 类型。用户未要求汇总时禁止 SUM/GROUP BY。"
                    "要求批次号、仓库位置或过期日期表示批次明细。保留比较符、阈值和排序方向，加入排序字段便于核验。"
                    "limit 仅在用户明确指定条数时填写，否则为 null。直接取数 response_mode=table；要求分析或报告时为 analysis。"
                    "条件有歧义、涉及写操作或需要多个独立查询时，在 clarification 中说明并请求澄清。"
                    f"\n结构：{json.dumps(schema, ensure_ascii=False)}"), HumanMessage(content=scope.question)])
                scope.response_mode = contract.response_mode
                if not contract.clarification:
                    validate_contract(scope.question, contract, schema)
                monitor._emit("database_contract", "已固定数据库查询口径", {"question": scope.question, "contract": contract.model_dump()})
                if contract.clarification:
                    scope.status = "refused"
                    scope.answer = contract.clarification
                    scope.result = json.dumps({"status": "refused", "clarification": contract.clarification}, ensure_ascii=False)
                else:
                    messages = [SystemMessage(content="生成一条 MySQL SELECT，严格遵守固定口径，禁止改变筛选、分组或排序；只返回 SQLDraft。"
                        f"\n结构：{json.dumps(schema, ensure_ascii=False)}\n固定口径：{contract.model_dump_json()}"), HumanMessage(content=scope.question)]
                    for attempt in range(2):
                        draft = await _structured(SQLDraft, messages)
                        try:
                            sql = validate_query(draft.sql, contract, schema)
                            break
                        except ValueError as exc:
                            if attempt == 1:
                                raise
                            messages.extend([AIMessage(content=draft.sql), HumanMessage(content=str(exc) + "。只修正 SQL，不得改变固定口径。")])
                    monitor.report_tool("数据库表数据查询工具：execute_sql_query", {"query": sql})
                    columns, rows, truncated = await asyncio.to_thread(execute_database_query, sql)
                    scope.result = _result_files(scope, sql, columns, rows, truncated)
            except Exception as exc:
                scope.answer = f"数据库查询失败：{exc}"
                scope.result = json.dumps({"status": "failed", "error": str(exc), "instruction": "本次查询已停止，不要反复委派或声称查到数据。"}, ensure_ascii=False)
    return {"messages": [AIMessage(content=scope.result)]}
