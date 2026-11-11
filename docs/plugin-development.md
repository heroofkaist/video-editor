# Plugin Development Guide

Symposium Studio is structured around clean extension contracts so contributors can add animation engines, research platform connectors, agents, and effects without editing core orchestration.

## 1. Adding an Animation Adapter
Subclass `AnimationAdapter` in `src/symposium_studio/engines/animation/base.py`:
1. Implement `capability() -> AnimationAdapterCapability` declaring version, availability check, license tier (`permissive_open`, `copyleft_isolated`, `commercial_restricted`), alpha support, and best domains.
2. Implement `render(scene_spec: SceneSpec, output_dir: Path) -> AnimationRenderResult` writing the editable `.spec.json` and producing ProRes4444 alpha `.mov` or H.264 `.mp4`.
3. Register the adapter in `AnimationEngineRegistry.adapters` (`engines/animation/registry.py`).
- **License rule**: Keep GPL/commercial tools strictly behind external subprocess calls or opt-in environment flags.

## 2. Adding a Platform Research Connector
Subclass `PlatformConnector` in `src/symposium_studio/engines/research/connectors.py`:
1. Report accurate auth eligibility and rate limits in `status()`.
2. Only consume official APIs, public RSS/oEmbed feeds, or explicit user imports. Never scrape behind login walls or bypass CAPTCHAs.

## 3. Adding an Editing Style or Effect Preset
- Add `EffectPreset` entries in `src/symposium_studio/engines/effects/library.py` with explicit `editorial_purpose` and configurable FFmpeg filter parameters.
- Add `EditingStyleProfile` configurations in `src/symposium_studio/engines/analysis/styles.py`.

## 4. Adding a Symposium Agent
Register an `AgentDefinition` in `AGENT_ROSTER` (`engines/symposium/orchestrator.py`) specifying `team`, `min_mode`, and evidence-grounded recommendation logic.
