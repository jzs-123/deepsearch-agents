from types import SimpleNamespace

import pytest

from app.security import auth


@pytest.fixture
def configured_auth(monkeypatch):
    config = SimpleNamespace(
        auth_enabled=True, auth_api_key="legacy-key",
        auth_users={"alice-key": {"user_id": "alice", "tenant_id": "team-a"}},
        cors_origins=["http://localhost:5173"],
    )
    monkeypatch.setattr(auth, "settings", config)
    return config


def test_identity_is_bound_to_key_not_client_headers(configured_auth):
    identity = auth.resolve_identity({
        "Authorization": "Bearer alice-key", "X-User-Id": "bob",
        "X-Tenant-Id": "team-b", "X-Roles": "admin",
    })
    assert (identity.user_id, identity.tenant_id, identity.roles) == ("alice", "team-a", ())


def test_cookie_auth_uses_same_verified_identity(configured_auth):
    assert auth.resolve_identity({"cookie": "deepsearch_auth=alice-key"}).user_id == "alice"


@pytest.mark.parametrize("headers", [{}, {"authorization": "Bearer wrong"}, {"cookie": "deepsearch_auth=wrong"}])
def test_missing_or_invalid_credential_is_rejected(configured_auth, headers):
    with pytest.raises(auth.AuthError):
        auth.resolve_identity(headers)


def test_legacy_key_cannot_choose_another_user(configured_auth):
    identity = auth.resolve_identity({"authorization": "Bearer legacy-key", "x-user-id": "alice"})
    assert (identity.user_id, identity.tenant_id) == ("anonymous", "default")


def test_auth_disabled_is_single_anonymous_identity(configured_auth):
    configured_auth.auth_enabled = False
    identity = auth.resolve_identity({"x-user-id": "alice", "x-tenant-id": "team-a"})
    assert (identity.user_id, identity.tenant_id, identity.authenticated) == ("anonymous", "default", False)


def test_memory_namespace_includes_tenant():
    assert auth.user_namespace(auth.Identity("alice", "a")) != auth.user_namespace(auth.Identity("alice", "b"))


def test_foreign_browser_origin_is_rejected(configured_auth):
    with pytest.raises(auth.AuthError):
        auth.resolve_identity({"cookie": "deepsearch_auth=alice-key", "origin": "https://evil.example"})
