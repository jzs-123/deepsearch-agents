import asyncio
import csv
import sqlite3

import pytest
from langchain_core.messages import HumanMessage

from app.agent.sql_query import QueryContract, QueryOrder, validate_contract, validate_query


SCHEMA = {
    'inventory': {'inventory_id': 'INT', 'drug_id': 'INT', 'quantity_on_hand': 'INT',
                  'batch_number': 'TEXT', 'warehouse_location': 'TEXT', 'expiry_date': 'DATE'},
    'drugs': {'drug_id': 'INT', 'generic_name': 'TEXT'},
}
SQL = '''SELECT d.generic_name, i.quantity_on_hand, i.batch_number,
    i.warehouse_location, i.expiry_date FROM inventory i
    LEFT JOIN drugs d ON i.drug_id = d.drug_id
    WHERE i.quantity_on_hand > 100 ORDER BY i.quantity_on_hand ASC, i.inventory_id ASC'''


@pytest.fixture
def contract():
    return QueryContract(tables=['inventory', 'drugs'],
        columns=['drugs.generic_name', 'inventory.quantity_on_hand', 'inventory.batch_number',
                 'inventory.warehouse_location', 'inventory.expiry_date'],
        joins=[{'condition':'inventory.drug_id = drugs.drug_id', 'kind':'left'}],
        where='inventory.quantity_on_hand > 100',
        order_by=[QueryOrder(expression='inventory.quantity_on_hand', direction='asc')])


@pytest.mark.parametrize('sql', [
    SQL.replace('WHERE i.quantity_on_hand > 100', ''),
    SQL.replace('> 100', '> 0'),
    SQL.replace('> 100', '>= 100'),
    SQL.replace('i.quantity_on_hand ASC', 'i.quantity_on_hand DESC'),
    SQL.replace('i.quantity_on_hand ASC', 'i.inventory_id ASC'),
    SQL.replace('i.expiry_date', 'i.production_date'),
    SQL + ' LIMIT 10',
    SQL + '; SELECT * FROM drugs',
    SQL.replace('WHERE i.quantity_on_hand > 100', 'WHERE i.quantity_on_hand > 100 OR 1=1'),
    SQL.replace('i.drug_id = d.drug_id', 'i.inventory_id = d.drug_id'),
    SQL.replace('LEFT JOIN', 'JOIN'),
    '''SELECT d.generic_name, SUM(i.quantity_on_hand) FROM inventory i
       JOIN drugs d ON i.drug_id=d.drug_id GROUP BY d.generic_name
       HAVING SUM(i.quantity_on_hand)>100 ORDER BY SUM(i.quantity_on_hand) DESC''',
])
def test_rejects_sql_that_changes_the_request(contract, sql):
    with pytest.raises(ValueError):
        validate_query(sql, contract, SCHEMA)


def test_valid_query_keeps_boundary_filter_and_ascending_order(contract):
    sql = validate_query(SQL, contract, SCHEMA)
    db = sqlite3.connect(':memory:')
    db.executescript('''CREATE TABLE drugs(drug_id INTEGER, generic_name TEXT);
        CREATE TABLE inventory(inventory_id INTEGER, drug_id INTEGER, quantity_on_hand INTEGER,
        batch_number TEXT, warehouse_location TEXT, expiry_date TEXT);
        INSERT INTO drugs VALUES(1,'测试药品');''')
    for row_id, qty in enumerate([101, 100, 99, 500, 102], start=1):
        db.execute('INSERT INTO inventory VALUES(?,?,?,?,?,?)',
                   (row_id, 1, qty, str(row_id), 'A', '2027-01-01'))
    rows = db.execute(sql).fetchall()
    assert [row[1] for row in rows] == [101, 102, 500]
    assert all(row[4] == '2027-01-01' for row in rows)
    db.close()


def test_aggregate_query_preserves_having_and_order():
    contract = QueryContract(tables=['inventory'], columns=['inventory.drug_id', 'SUM(inventory.quantity_on_hand)'],
        group_by=['inventory.drug_id'], having='SUM(inventory.quantity_on_hand)>100',
        order_by=[QueryOrder(expression='SUM(inventory.quantity_on_hand)', direction='desc')])
    sql = 'SELECT drug_id,SUM(quantity_on_hand) AS total FROM inventory GROUP BY drug_id HAVING SUM(quantity_on_hand)>100 ORDER BY total DESC'
    assert 'HAVING' in validate_query(sql, contract, SCHEMA)


def test_aliases_do_not_change_requirements(contract):
    assert validate_query(SQL.replace('d.', 'drug.').replace('drugs d ', 'drugs drug ')
                          .replace('i.', 'stock.').replace('inventory i', 'inventory stock'), contract, SCHEMA)


@pytest.mark.parametrize('changes', [
    {'where': 'inventory.quantity_on_hand > 0'},
    {'where': 'inventory.quantity_on_hand >= 100'},
    {'order_by': [QueryOrder(expression='inventory.quantity_on_hand',direction='desc')]},
    {'columns': ['drugs.generic_name']},
])
def test_wrong_contract_is_rejected_against_original_question(contract, changes):
    with pytest.raises(ValueError):
        validate_contract('查询库存大于100的药品，按库存量升序列出药品名称、批次号、仓库位置和过期日期',
                          contract.model_copy(update=changes), SCHEMA)


def test_pipeline_repairs_once_and_reuses_result_without_losing_original_request(contract, tmp_path, monkeypatch):
    import app.agent.sql_query as query
    from app.api.context import set_session_context, reset_session_context
    requests = []
    sqls = iter([SQL.replace('WHERE i.quantity_on_hand > 100', ''), SQL])
    executions = []
    async def structured(schema, messages):
        requests.append((schema, messages))
        return contract if schema is QueryContract else query.SQLDraft(sql=next(sqls))
    monkeypatch.setattr(query, '_structured', structured)
    monkeypatch.setattr(query, 'get_database_schema', lambda: SCHEMA)
    def execute(sql):
        executions.append(sql)
        return ['generic_name','quantity_on_hand','batch_number','warehouse_location','expiry_date'], [
            ('含,逗号的药品',101,'B1','A','2027-01-01'), ('药品二',102,'B2','B','2027-02-01')], False
    monkeypatch.setattr(query, 'execute_database_query', execute)
    scope = query.DatabaseRun('库存大于100，升序，列出药品、批次、仓库和过期日期')
    session_token = set_session_context(str(tmp_path))
    token = query.database_run.set(scope)
    async def run():
        return await asyncio.gather(*[query.run_database_query({'messages':[HumanMessage(content='查询最后几条')]}) for _ in range(2)])
    try:
        results = asyncio.run(run())
        assert results[0]['messages'][-1].content == results[1]['messages'][-1].content
        assert len(requests) == 3  # 固定口径一次、SQL 两次；无 Agent 工具循环。
        assert scope.question in requests[0][1][-1].content
        assert len(executions) == 1
        assert scope.row_count == 2
        with (tmp_path / scope.artifact).open(encoding='utf-8-sig', newline='') as f:
            assert list(csv.reader(f))[1][0] == '含,逗号的药品'
        assert '2027-01-01' in scope.answer
        assert '含,逗号的药品' in scope.answer
    finally:
        query.database_run.reset(token)
        reset_session_context(session_token)


def test_exhausted_repair_never_executes_bad_sql(contract, tmp_path, monkeypatch):
    import app.agent.sql_query as query
    from app.api.context import set_session_context, reset_session_context
    calls=[]
    async def structured(schema, messages):
        calls.append(schema)
        return contract if schema is QueryContract else query.SQLDraft(sql=SQL.replace('> 100','> 0'))
    monkeypatch.setattr(query, '_structured', structured)
    monkeypatch.setattr(query, 'get_database_schema', lambda: SCHEMA)
    monkeypatch.setattr(query, 'execute_database_query', lambda _: pytest.fail('错误 SQL 不得执行'))
    token = set_session_context(str(tmp_path))
    try:
        result = asyncio.run(query.run_database_query({'messages':[HumanMessage(content='库存>100')]}))
        assert '失败' in result['messages'][-1].content
        assert len(calls) == 3
    finally:
        reset_session_context(token)


def test_schema_cache_expires_and_does_not_reuse_other_connection(monkeypatch):
    import app.tools.db_tools as db
    connects = []
    now = [0]
    config = {'database':'business','host':'local','user':'one'}
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, sql, params): assert params == (config['database'],)
        def fetchall(self): return [('inventory','quantity_on_hand','int')]
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def cursor(self): return Cursor()
    def connect(**kwargs):
        connects.append(kwargs)
        return Connection()
    monkeypatch.setattr(db, 'get_db_config', lambda: config.copy())
    monkeypatch.setattr(db, 'connect', connect)
    monkeypatch.setattr(db, '_schema_cache', {})
    monkeypatch.setattr(db.time, 'monotonic', lambda: now[0])
    assert db.get_database_schema() == {'inventory': {'quantity_on_hand':'int'}}
    db.get_database_schema()
    assert len(connects) == 1
    now[0] = 301
    db.get_database_schema()
    config['user'] = 'two'
    db.get_database_schema()
    assert len(connects) == 3


def test_execution_cap_keeps_order_and_marks_partial(monkeypatch):
    import app.tools.db_tools as db
    statements=[]
    class Cursor:
        description=[('quantity_on_hand',)]
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, sql): statements.append(sql)
        def fetchall(self): return [(i,) for i in range(10001)]
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def cursor(self): return Cursor()
    monkeypatch.setattr(db, 'connect', lambda **_: Connection())
    monkeypatch.setattr(db, 'get_db_config', lambda: {})
    columns, rows, truncated = db.execute_database_query('SELECT quantity_on_hand FROM inventory ORDER BY quantity_on_hand ASC')
    assert statements[0] == 'SET SESSION max_execution_time = 10000'
    assert statements[1].endswith('ORDER BY quantity_on_hand ASC LIMIT 10001')
    assert columns == ['quantity_on_hand']
    assert len(rows) == 10000 and truncated


@pytest.mark.parametrize('artifact_path', [None, '/database-cafebabe.csv'])
def test_main_graph_keeps_original_query_caches_redelegation_and_presents_raw_rows(contract, tmp_path, monkeypatch, artifact_path):
    import importlib.util
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    import app.agent.sql_query as query
    from app.utils import history_manager as history
    class FakeModel(FakeMessagesListChatModel):
        def bind_tools(self, *args, **kwargs): return self
    monkeypatch.setattr(query, 'uuid', SimpleNamespace(uuid4=lambda: SimpleNamespace(hex='cafebabe')))
    fake = FakeModel(responses=[
        AIMessage(content='', tool_calls=[{'name':'task','id':'db1','args':{'subagent_type':'数据库查询助手','description':'查询剩余记录'}}]),
        AIMessage(content='', tool_calls=[{'name':'task','id':'db2','args':{'subagent_type':'数据库查询助手','description':'查询最后几条'}}]),
        AIMessage(content='', tool_calls=[{'name':'AgentResult','id':'final','args':{'status':'success','answer':'总计138条，过期日期字段不存在',
            'artifacts': [] if artifact_path is None else [{'path':artifact_path,'title':'查询结果'}]}}]),
    ])
    monkeypatch.setitem(sys.modules, 'app.agent.llm', SimpleNamespace(model=fake))
    requests=[]
    executions=[]
    async def structured(schema, messages):
        requests.append(messages)
        return contract if schema is QueryContract else query.SQLDraft(sql=SQL)
    monkeypatch.setattr(query, '_structured', structured)
    monkeypatch.setattr(query, 'get_database_schema', lambda: SCHEMA)
    def execute(sql):
        executions.append(sql)
        return ['药品名称','库存量','批次号','仓库位置','过期日期'], [('A',101,'B1','W1','2027-01-01')], False
    monkeypatch.setattr(query, 'execute_database_query', execute)
    path=Path(__file__).resolve().parents[1]/'app/agent/main_agent.py'
    spec=importlib.util.spec_from_file_location('test_sql_main_agent',path)
    agent=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    (tmp_path/'skills').mkdir()
    monkeypatch.setattr(agent, 'project_root_path', tmp_path)
    conn=sqlite3.connect(tmp_path/'sessions.db',check_same_thread=False)
    conn.row_factory=sqlite3.Row
    history._init_db(conn)
    monkeypatch.setattr(history,'_conn',conn)
    try:
        original='库存大于100，按库存升序列出药品、批次、仓库和过期日期'
        result=asyncio.run(agent.run_deep_agent(original,'sql-test'))
        assert requests[0][-1].content == original
        assert len(requests)==2 and len(executions)==1
        assert '查询返回 1 条记录' in result.answer
        assert '2027-01-01' in result.answer and '138' not in result.answer
        assert len(result.artifacts)==1 and Path(result.artifacts[0].path).is_file()
        assert query.database_run.get() is None
    finally:
        conn.close()
