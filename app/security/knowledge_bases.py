"""知识库访问范围：本地匿名库、租户库与显式共享只读库。"""

from dataclasses import replace
import hashlib

from app.core.config import settings
from app.api.knowledge_bases import normalize_kb_name
from app.rag.config import load_rag_config
from app.rag.vectorstore import list_knowledge_bases
from app.security.auth import AuthError, Identity, get_identity_context


def knowledge_config(identity: Identity | None = None):
    config = load_rag_config()
    if not settings.auth_enabled:
        return config
    identity = identity or get_identity_context()
    if not identity or not identity.authenticated:
        raise AuthError("知识库访问需要已验证的身份")
    tenant_key = hashlib.sha256(identity.tenant_id.encode()).hexdigest()
    return replace(config, data_dir=config.data_path / ".tenants" / tenant_key)


def visible_knowledge_bases(identity: Identity | None = None) -> list[dict]:
    own = list_knowledge_bases(knowledge_config(identity).data_path)
    if not settings.auth_enabled:
        return own
    names = {item["name"] for item in own}
    shared = [
        item for item in list_knowledge_bases(load_rag_config().data_path)
        if item["name"] in settings.rag_shared_kbs and item["name"] not in names
    ]
    return own + shared


def query_knowledge_config(kb_name: str, identity: Identity | None = None):
    config = knowledge_config(identity)
    name = normalize_kb_name(kb_name or config.default_kb)
    if settings.auth_enabled and not config.kb_dir(name).exists() and name in settings.rag_shared_kbs:
        return load_rag_config()
    return config
