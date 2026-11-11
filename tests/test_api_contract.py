"""FastAPI REST API contract and workspace path-containment tests."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from symposium_studio.api.app import create_app
from symposium_studio.config import StudioConfig


def test_api_endpoints_and_path_security(tmp_path: Path) -> None:
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "s.db")
    app = create_app(cfg)
    client = TestClient(app)

    h = client.get("/api/health")
    assert h.status_code == 200
    assert h.json()["agents_count"] == 18

    created = client.post("/api/projects", json={"name": "API Demo", "creator_niche": "programming"})
    assert created.status_code == 200
    pid = created.json()["project"]["project_id"]

    # Path traversal protection on workspace file server
    esc = client.get(f"/api/projects/{pid}/files/../../etc/passwd")
    assert esc.status_code in (400, 404)

    # Effects & viral research endpoints
    fx = client.get("/api/effects?category=transition")
    assert fx.status_code == 200 and len(fx.json()) >= 5

    vi = client.get("/api/research/items?category=gaming")
    assert vi.status_code == 200 and len(vi.json()) >= 1
