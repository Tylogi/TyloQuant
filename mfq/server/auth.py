"""Scoped API key issuance and authentication."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

from mfq.server.models import ApiKeyResource, ApiKeyScope, ApiKeySecretResource, CreateApiKeyRequest
from mfq.server.storage import SessionStore

ROLE_SCOPES = {
    "viewer": frozenset({"inference"}),
    "operator": frozenset({"inference", "models", "jobs"}),
    "administrator": frozenset({"admin"}),
}


def hash_api_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuthenticatedKey:
    resource: ApiKeyResource | None
    scopes: frozenset[ApiKeyScope]


class ApiKeyManager:
    def __init__(self, store: SessionStore, root_key: str = "") -> None:
        self.store = store
        self.root_key = root_key

    def create(self, request: CreateApiKeyRequest) -> ApiKeySecretResource:
        token = f"mfq_{secrets.token_urlsafe(32)}"
        resource = self.store.create_api_key(
            request,
            hash_api_key(token),
            token[:12],
        )
        return ApiKeySecretResource(key=resource, token=token)

    def authenticate(self, token: str) -> AuthenticatedKey | None:
        if self.root_key and secrets.compare_digest(token, self.root_key):
            return AuthenticatedKey(resource=None, scopes=frozenset({"admin"}))
        resource = self.store.authenticate_api_key(hash_api_key(token))
        if resource is None:
            return None
        scopes = frozenset(resource.scopes)
        if resource.role is not None:
            scopes = scopes | ROLE_SCOPES[resource.role]
        return AuthenticatedKey(resource=resource, scopes=scopes)

    def rotate(self, key_id) -> ApiKeySecretResource:
        token = f"mfq_{secrets.token_urlsafe(32)}"
        resource = self.store.rotate_api_key(
            key_id,
            hash_api_key(token),
            token[:12],
        )
        return ApiKeySecretResource(key=resource, token=token)

    @staticmethod
    def permits(authenticated: AuthenticatedKey, required: ApiKeyScope) -> bool:
        return "admin" in authenticated.scopes or required in authenticated.scopes


def required_scope(method: str, path: str) -> ApiKeyScope:
    if path.startswith("/api/v1/auth/"):
        return "admin"
    if path.startswith("/api/v1/cluster/"):
        return "admin"
    if path.startswith("/api/v1/models/directories"):
        return "models"
    if path.startswith(("/api/v1/models", "/api/v1/runtime", "/api/v1/hub")):
        return "models" if method != "GET" else "inference"
    if path.startswith(
        ("/api/v1/jobs", "/api/v1/artifacts", "/api/v1/datasets", "/api/v1/evaluations")
    ):
        return "jobs"
    if path.startswith("/api/v1/mcp/") and method != "GET":
        return "admin"
    return "inference"
