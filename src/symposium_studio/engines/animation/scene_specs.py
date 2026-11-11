"""Editable SceneSpec presets across physics, electrical engineering, programming, and storytelling."""

from __future__ import annotations

import uuid

from symposium_studio.models.animation import (
    AnimationDomain,
    FactualityMode,
    KeyframeValue,
    SceneElementSpec,
    SceneSpec,
)


def build_scene_spec(
    template_id: str,
    title: str | None = None,
    duration_sec: float = 7.0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    transparent_background: bool = True,
    preferred_adapter: str | None = None,
    parameters: dict | None = None,
) -> SceneSpec:
    params = dict(parameters or {})
    scene_id = f"anim_{uuid.uuid4().hex[:8]}"

    if template_id == "em_induction_faraday":
        return SceneSpec(
            scene_id=scene_id,
            title=title or "Electromagnetic Induction — Faraday's Law",
            domain=AnimationDomain.PHYSICS,
            template_id=template_id,
            factuality_mode=FactualityMode.FACTUAL_SCIENTIFIC,
            duration_sec=duration_sec,
            width=width,
            height=height,
            fps=fps,
            transparent_background=transparent_background,
            preferred_adapter=preferred_adapter,
            scientific_notes=[
                "Analytical Faraday law model: EMF = -N * dPhi_B/dt.",
                "Lenz's law sign convention: induced current opposes magnetic flux change.",
                "Waveform plotted directly from numerical derivative of Phi_B(t) = B0 * cos(w*t) * coupling(x(t)).",
            ],
            parameters={"turns_N": 6, "oscillations": 2.0, "show_equation": True, **params},
            elements=[
                SceneElementSpec(
                    element_id="bar_magnet",
                    kind="magnet",
                    label="Bar Magnet (N / S)",
                    keyframes={
                        "x_offset": [
                            KeyframeValue(time_sec=0.0, value=-180, easing="ease_in_out"),
                            KeyframeValue(time_sec=duration_sec * 0.25, value=40, easing="ease_in_out"),
                            KeyframeValue(time_sec=duration_sec * 0.5, value=-160, easing="ease_in_out"),
                            KeyframeValue(time_sec=duration_sec * 0.75, value=55, easing="ease_in_out"),
                            KeyframeValue(time_sec=duration_sec, value=-140, easing="ease_in_out"),
                        ]
                    },
                ),
                SceneElementSpec(element_id="copper_coil", kind="coil", label="N-Turn Copper Coil"),
                SceneElementSpec(element_id="b_field_lines", kind="field_lines", label="Magnetic Flux Lines B"),
                SceneElementSpec(element_id="faraday_eq", kind="equation", label="EMF = -N dPhi_B/dt"),
            ],
        )

    if template_id == "rlc_signal_flow":
        return SceneSpec(
            scene_id=scene_id,
            title=title or "Circuit Signal Flow & Transient Response",
            domain=AnimationDomain.ELECTRICAL_ENGINEERING,
            template_id=template_id,
            factuality_mode=FactualityMode.FACTUAL_SCIENTIFIC,
            duration_sec=duration_sec,
            width=width,
            height=height,
            fps=fps,
            transparent_background=transparent_background,
            preferred_adapter=preferred_adapter,
            scientific_notes=[
                "Underdamped second-order RLC response: v(t) = exp(-alpha*t) * cos(omega_d*t).",
                "Electron drift arrows pulse proportionally to instantaneous i(t).",
            ],
            parameters={"damping_alpha": 0.45, "freq_hz": 1.6, **params},
            elements=[
                SceneElementSpec(element_id="rlc_loop", kind="circuit_loop", label="R-L-C Series Loop"),
                SceneElementSpec(element_id="scope_trace", kind="waveform", label="V_out(t) & I(t) Scope"),
            ],
        )

    if template_id == "binary_tree_trace":
        return SceneSpec(
            scene_id=scene_id,
            title=title or "Binary Search Tree Lookup & O(log n) Trace",
            domain=AnimationDomain.PROGRAMMING,
            template_id=template_id,
            factuality_mode=FactualityMode.FACTUAL_SCIENTIFIC,
            duration_sec=duration_sec,
            width=width,
            height=height,
            fps=fps,
            transparent_background=transparent_background,
            preferred_adapter=preferred_adapter,
            scientific_notes=[
                "Exact BST search path comparison trace for target key=42.",
                "Complexity curves plot exact n vs log2(n) step counts.",
            ],
            parameters={"target_key": 42, "values": [32, 16, 48, 8, 24, 40, 56, 42], **params},
            elements=[
                SceneElementSpec(element_id="bst_nodes", kind="tree_node", label="Balanced BST Traversal"),
                SceneElementSpec(element_id="array_view", kind="array_node", label="Sorted Array Index Trace"),
            ],
        )

    if template_id == "system_architecture_flow":
        return SceneSpec(
            scene_id=scene_id,
            title=title or "Distributed Ingestion & Streaming Pipeline Flow",
            domain=AnimationDomain.PROGRAMMING,
            template_id=template_id,
            factuality_mode=FactualityMode.FACTUAL_SCIENTIFIC,
            duration_sec=duration_sec,
            width=width,
            height=height,
            fps=fps,
            transparent_background=transparent_background,
            preferred_adapter=preferred_adapter,
            scientific_notes=[
                "Stage-by-stage packet flow illustrating queue buffering, worker compute, and cold storage.",
                "Throughput & stage latency bars reflect configured simulation parameters.",
            ],
            parameters={
                "stages": params.get(
                    "stages",
                    [
                        {"name": "Ingest Edge", "metric": "1.2k req/s", "color": "#38BDF8"},
                        {"name": "Queue Buffer", "metric": "p99 4.2 ms", "color": "#F59E0B"},
                        {"name": "Worker Pool", "metric": "6x parallel", "color": "#34D399"},
                        {"name": "Vector Index", "metric": "HNSW persist", "color": "#A78BFA"},
                    ],
                ),
                "packet_rate_hz": 1.8,
                **params,
            },
            elements=[
                SceneElementSpec(element_id="pipeline_nodes", kind="node_stage", label="Pipeline Stages"),
                SceneElementSpec(element_id="packet_bus", kind="data_packet", label="Animated Packet Flow"),
            ],
        )

    if template_id == "wave_interference_sim":
        return SceneSpec(
            scene_id=scene_id,
            title=title or "Acoustic & Electromagnetic Wave Superposition",
            domain=AnimationDomain.PHYSICS,
            template_id=template_id,
            factuality_mode=FactualityMode.FACTUAL_SCIENTIFIC,
            duration_sec=duration_sec,
            width=width,
            height=height,
            fps=fps,
            transparent_background=transparent_background,
            preferred_adapter=preferred_adapter,
            scientific_notes=[
                "Linear superposition principle: y_sum(x,t) = A1*sin(k1*x - w1*t) + A2*sin(k2*x - w2*t + phi).",
                "Demonstrates constructive vs destructive beat envelope cleanly.",
            ],
            parameters={
                "k1": 3.0,
                "k2": 3.7,
                "omega1": 4.2,
                "omega2": 4.9,
                "phase_rad": 0.0,
                **params,
            },
            elements=[
                SceneElementSpec(element_id="wave_a", kind="sinusoid", label="Carrier Wave y1(x,t)"),
                SceneElementSpec(element_id="wave_b", kind="sinusoid", label="Detuned Wave y2(x,t)"),
                SceneElementSpec(element_id="wave_sum", kind="superposition", label="Superposition y1 + y2"),
            ],
        )

    if template_id == "narrative_timeline":
        return SceneSpec(
            scene_id=scene_id,
            title=title or "Engineering & Story Timeline",
            domain=AnimationDomain.STORYTELLING,
            template_id=template_id,
            factuality_mode=FactualityMode.ARTISTIC_ILLUSTRATION,
            duration_sec=duration_sec,
            width=width,
            height=height,
            fps=fps,
            transparent_background=transparent_background,
            preferred_adapter=preferred_adapter,
            parameters={
                "milestones": params.get(
                    "milestones",
                    [
                        {"tag": "01", "title": "Hypothesis", "detail": "Initial coil setup"},
                        {"tag": "02", "title": "Failure", "detail": "Static magnet = 0 V"},
                        {"tag": "03", "title": "Breakthrough", "detail": "Changing flux induces EMF"},
                    ],
                ),
                **params,
            },
            elements=[
                SceneElementSpec(element_id="timeline_rail", kind="timeline_card", label="Chronological Rail"),
            ],
        )

    if template_id == "meme_reaction_card":
        return SceneSpec(
            scene_id=scene_id,
            title=title or "Trending Meme Reaction",
            domain=AnimationDomain.STORYTELLING,
            template_id=template_id,
            factuality_mode=FactualityMode.ARTISTIC_ILLUSTRATION,
            duration_sec=duration_sec,
            width=width,
            height=height,
            fps=fps,
            transparent_background=transparent_background,
            preferred_adapter=preferred_adapter,
            parameters={
                "top_label": params.get("top_label", "TRENDING MEME"),
                "meme_name": params.get("meme_name", "Live Trend"),
                "reaction_text": params.get("reaction_text", title or "Wait for it"),
                "image_path": params.get("image_path", ""),
                "image_url": params.get("image_url", ""),
                "accent_color": params.get("accent_color", "#FACC15"),
                **params,
            },
            elements=[
                SceneElementSpec(element_id="meme_card", kind="meme_popup", label="Live Trending Meme Pop-up"),
            ],
        )

    return SceneSpec(
        scene_id=scene_id,
        title=title or "Key Insight Callout",
        domain=AnimationDomain.STORYTELLING,
        template_id="kinetic_callout",
        factuality_mode=FactualityMode.ARTISTIC_ILLUSTRATION,
        duration_sec=duration_sec,
        width=width,
        height=height,
        fps=fps,
        transparent_background=transparent_background,
        preferred_adapter=preferred_adapter,
        parameters={"headline": title or "Key Insight", "subtitle": params.get("subtitle", ""), **params},
        elements=[SceneElementSpec(element_id="callout_card", kind="badge", label="Kinetic Callout")],
    )
