"""FastAPI REST API contract and workspace path-containment tests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from symposium_studio.api.app import create_app
from symposium_studio.config import StudioConfig
from symposium_studio.engines.research import YouTubeOfficialConnector
from symposium_studio.models.research import ContentCategory, PublicEngagementMetrics, ResearchPlatform, ViralContentItem
from symposium_studio.models.social import FunnelLink


def test_api_endpoints_and_path_security(tmp_path: Path) -> None:
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "s.db")
    app = create_app(cfg)
    client = TestClient(app)

    h = client.get("/api/health")
    assert h.status_code == 200
    body_h = h.json()
    assert body_h["agents_count"] == 18
    assert "llm_backend" in body_h

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


def test_api_new_research_style_growth_and_funnel_routes(monkeypatch: Any, tmp_path: Path) -> None:
    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "s.db")
    app = create_app(cfg)
    client = TestClient(app)

    pid = client.post("/api/projects", json={"name": "Route Coverage", "creator_niche": "science"}).json()["project"]["project_id"]

    # 1. Research discover + inspect-url (mocked connector methods so test stays hermetic)
    studio = app.state.studio
    mock_item = ViralContentItem(
        item_id="yt_rss_mock1",
        platform=ResearchPlatform.YOUTUBE,
        creator="Science Demo",
        original_url="https://www.youtube.com/watch?v=mock1",
        title="Mocked Discovery Outlier",
        published_at="2026-10-12T10:00:00+00:00",
        category=ContentCategory.SCIENCE,
        topic="electromagnetism",
        opening_hook_structure="Immediate anomaly.",
        editing_style="tight",
        narrative_structure="hook -> proof",
        metrics=PublicEngagementMetrics(
            platform=ResearchPlatform.YOUTUBE,
            metric_source="official_public_rss",
            views=95000,
            likes=4200,
            channel_baseline_views=19000,
            audience_normalized_ratio=5.0,
        ),
        evidence="Official Atom RSS statistics: 95,000 views vs 19,000 baseline.",
        confidence=0.9,
        platform_relative_score=86.0,
        score_basis="audience_normalized_breakout",
        provenance_type="official_api",
    )
    monkeypatch.setattr(YouTubeOfficialConnector, "discover_multi_niche_corpus", lambda self, **kw: [mock_item])
    disc = client.post("/api/research/discover", json={"target_count": 20})
    assert disc.status_code == 200
    assert disc.json()["discovered_count"] == 1

    monkeypatch.setattr(YouTubeOfficialConnector, "fetch_from_url_oembed", lambda self, url, **kw: mock_item)
    insp = client.post("/api/research/inspect-url", json={"url": "https://www.youtube.com/watch?v=mock1"})
    assert insp.status_code == 200
    assert insp.json()["item_id"] == "yt_rss_mock1"

    # 2. Reference video style cloning
    ref_mp4 = tmp_path / "ref_clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=2.5:size=640x360:rate=24",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(ref_mp4),
        ],
        check=True,
    )
    cloned = client.post(f"/api/projects/{pid}/style-from-reference", json={"reference_video_path": str(ref_mp4)})
    assert cloned.status_code == 200
    assert "editing_style" in cloned.json() and "profile" in cloned.json()

    # 3. Growth import + Telegram sync + funnel redirect safety
    imp = client.post(
        f"/api/growth/{pid}/import",
        json={"records": [{"platform": "youtube", "views": 3200, "likes": 190, "external_post_id": "v1"}]},
    )
    assert imp.status_code == 200 and imp.json()["imported_count"] == 1

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot:secret")

    class _OkTg:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {"ok": True, "result": 900}

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _OkTg())
    sync_r = client.post(f"/api/growth/{pid}/telegram-sync", json={"chat_id": "@demo_channel"})
    assert sync_r.status_code == 200 and sync_r.json()["subscriber_or_member_count"] == 900

    # Funnel link redirect & event tracking + open-redirect protection
    studio.db.upsert_funnel_link(
        pid,
        FunnelLink(
            link_id="lnk_safe",
            source_platform="youtube",
            campaign_slug="yt_main",
            target_telegram_channel="@demo_channel",
            tagged_url="https://t.me/demo_channel?start=yt_main",
        ),
    )
    studio.db.upsert_funnel_link(
        pid,
        FunnelLink(
            link_id="lnk_evil",
            source_platform="youtube",
            campaign_slug="evil",
            target_telegram_channel="@demo_channel",
            tagged_url="https://evil.example/phish",
        ),
    )
    ev = client.post("/api/funnel/lnk_safe/event?clicks_delta=2&conversions_delta=1")
    assert ev.status_code == 200 and ev.json()["clicks"] == 2 and ev.json()["conversions"] == 1

    redir = client.get("/r/lnk_safe", follow_redirects=False)
    assert redir.status_code == 302
    assert redir.headers["location"] == "https://t.me/demo_channel?start=yt_main"

    blocked = client.get("/r/lnk_evil", follow_redirects=False)
    assert blocked.status_code == 400


def test_api_growth_experiments_and_evaluation(tmp_path: Path) -> None:
    from tests.test_research_and_symposium import _sample_analysis

    cfg = StudioConfig(repo_root=tmp_path, workspaces_root=tmp_path / "ws", db_path=tmp_path / "ws" / "exp_api.db")
    app = create_app(cfg)
    client = TestClient(app)
    studio = app.state.studio

    pid1 = client.post("/api/projects", json={"name": "Growth Exp One", "creator_niche": "programming"}).json()["project"]["project_id"]
    pid2 = client.post("/api/projects", json={"name": "Growth Exp Two", "creator_niche": "science_engineering"}).json()["project"]["project_id"]

    rep = studio.research.generate_research_report("content_proposal", "science_engineering", ["faraday"])
    s1 = studio.symposium.run_symposium(pid1, _sample_analysis(pid1), rep)
    s2 = studio.symposium.run_symposium(pid2, _sample_analysis(pid2), rep)
    studio.growth.record_experiment_from_production(pid1, "API Baseline Exp", "programming", s1, rep)
    studio.growth.record_experiment_from_production(pid2, "API Stretch Exp", "science_engineering", s2, rep)

    client.post(
        f"/api/projects/{pid1}/growth",
        json={"platform": "youtube", "views_or_impressions": 1000, "first_7d_views": 1000, "external_post_id": "yt1"},
    )
    client.post(
        f"/api/projects/{pid2}/growth",
        json={
            "platform": "youtube",
            "views_or_impressions": 2200,
            "first_7d_views": 2200,
            "authorized_ctr_pct": 6.2,
            "authorized_retention_pct": 55.0,
            "external_post_id": "yt2",
        },
    )
    ev_resp = client.post(f"/api/growth/{pid2}/evaluate")
    assert ev_resp.status_code == 200
    ev_data = ev_resp.json()
    yt_exp = next(e for e in ev_data["experiments"] if e["platform"] == "youtube")
    assert yt_exp["ratio_vs_creator_median"] == 2.2
    assert yt_exp["growth_tier_reached"] == "stretch_target_2.0x"

    search_resp = client.get("/api/growth/experiments?q=Stretch&platform=youtube")
    assert search_resp.status_code == 200
    found = search_resp.json()
    assert any(item["project_id"] == pid2 for item in found)
