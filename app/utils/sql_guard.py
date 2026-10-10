"""
SQL 安全防护模块

为数据库查询工具提供两道防线:
1. validate_identifier:表名/列名等标识符白名单校验,杜绝拼接注入
2. assert_readonly_sql:仅放行只读查询语句,拦截 DML/DDL/管理类语句
"""

import re

# 标识符只允许字母、数字、下划线,兼容反引号包裹的合法表名
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_]+$")

# 允许的只读语句前缀
_READONLY_PREFIXES = ("SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN", "WITH")

# 危险关键字(命中即拒绝),覆盖写操作与管理操作
_DANGEROUS_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE",
    "GRANT", "REVOKE", "REPLACE", "RENAME", "SET", "CALL", "EXEC", "EXECUTE",
    "ATTACH", "DETACH", "PRAGMA", "BEGIN", "COMMIT", "ROLLBACK", "LOCK", "UNLOCK",
)

_MAX_SQL_LENGTH = 4000


def validate_identifier(name: str) -> str:
    """
    校验并清洗数据库标识符(表名/列名)。

    :param name: 原始标识符,可带反引号
    :return: 清洗后的标识符
    :raises ValueError: 标识符非法时抛出
    """
    cleaned = str(name).strip().strip("`").strip()
    if not _IDENTIFIER_RE.match(cleaned):
        raise ValueError(f"非法标识符: {name!r},仅允许字母、数字、下划线")
    return cleaned


def assert_readonly_sql(query: str) -> str:
    """
    校验 SQL 语句为只读查询。

    :param query: 待执行的 SQL
    :return: 清洗后的 SQL(去除行尾分号)
    :raises ValueError: 非只读或超长时抛出
    """
    sql = (query or "").strip()
    if not sql:
        raise ValueError("SQL 语句不能为空")
    if len(sql) > _MAX_SQL_LENGTH:
        raise ValueError(f"SQL 语句过长(>{_MAX_SQL_LENGTH} 字符),已拒绝执行")

    # 去掉注释与行尾分号,便于提取首关键字
    normalized = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    normalized = re.sub(r"/\*.*?\*/", "", normalized, flags=re.DOTALL)
    normalized = normalized.strip().rstrip(";").strip()

    upper = normalized.upper()

    # 先做危险关键字拦截,再做首关键字白名单
    for keyword in _DANGEROUS_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            raise ValueError(f"已拦截非只读 SQL(命中关键字 {keyword})")

    first_word = upper.split(None, 1)[0] if upper else ""
    if not first_word.startswith(_READONLY_PREFIXES):
        raise ValueError(f"仅允许只读查询(SELECT/SHOW/DESCRIBE/EXPLAIN),当前首关键字: {first_word}")

    return normalized


def safe_identifier_or_error(name: str) -> str:
    """
    兼容旧调用风格的校验:非法时返回错误信息字符串而非抛异常。
    供 db_tools 直接拼进返回结果使用。
    """
    try:
        return validate_identifier(name)
    except ValueError as e:
        return f"错误: {e}"
