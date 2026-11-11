"""Shot detection (PySceneDetect), face tracking (OpenCV), motion/sharpness, and local OCR."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scenedetect import ContentDetector, detect

from symposium_studio.models.media import FaceDetectionBox, SceneCut, VisualContextSample


def detect_scene_cuts(media_path: Path, duration_sec: float, threshold: float = 27.0) -> list[SceneCut]:
    scene_list = detect(str(media_path), ContentDetector(threshold=threshold))
    cuts: list[SceneCut] = []
    if not scene_list:
        return [
            SceneCut(
                scene_index=0,
                start_sec=0.0,
                end_sec=round(duration_sec, 3),
                duration_sec=round(duration_sec, 3),
            )
        ]
    for idx, (start_tc, end_tc) in enumerate(scene_list):
        s = round(start_tc.get_seconds(), 3)
        e = round(end_tc.get_seconds(), 3)
        cuts.append(
            SceneCut(
                scene_index=idx,
                start_sec=s,
                end_sec=e,
                duration_sec=round(max(0.05, e - s), 3),
            )
        )
    return cuts


def _try_apple_vision_ocr(frame_bgr: np.ndarray) -> str:
    """Optional macOS Vision OCR when pyobjc-framework-Vision is importable; returns '' otherwise."""
    try:
        import Quartz  # type: ignore
        import Vision  # type: ignore
    except Exception:
        return ""

    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        return ""
    data = buf.tobytes()
    provider = Quartz.CGDataProviderCreateWithData(None, data, len(data), None)
    cg_img = Quartz.CGImageCreateWithPNGDataProvider(provider, None, True, Quartz.kCGRenderingIntentDefault)
    if cg_img is None:
        return ""
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_img, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast)
    success, _ = handler.performRequests_error_([req], None)
    if not success:
        return ""
    lines = []
    for obs in req.results() or []:
        cands = obs.topCandidates_(1)
        if cands:
            lines.append(str(cands[0].string()))
    return " | ".join(lines[:6])


def _estimate_subject_center_x(frame_bgr: np.ndarray) -> float | None:
    """Estimate primary presenter center X using YCrCb skin-region segmentation when Haar misses."""
    h, w = frame_bgr.shape[:2]
    upper = frame_bgr[: int(h * 0.78), :]
    ycrcb = cv2.cvtColor(upper, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, np.array([45, 133, 77], dtype=np.uint8), np.array([255, 178, 130], dtype=np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < (w * h * 0.005):
        return None
    x, _, bw, _ = cv2.boundingRect(largest)
    return round(min(0.85, max(0.15, (x + bw * 0.5) / max(1, w))), 4)


def analyze_visual_context(
    media_path: Path,
    sample_interval_sec: float = 2.0,
    run_ocr: bool = True,
) -> tuple[list[VisualContextSample], float]:
    """Sample frames for face tracking (for dynamic 9:16 crop), motion energy, sharpness, and OCR."""
    cap = cv2.VideoCapture(str(media_path))
    if not cap.isOpened():
        return [], 0.5

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0

    cascades = []
    for fname in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"):
        cp = Path(cv2.data.haarcascades) / fname
        if cp.exists():
            cascades.append(cv2.CascadeClassifier(str(cp)))

    samples: list[VisualContextSample] = []
    prev_gray: np.ndarray | None = None
    step_frames = max(1, int(round(sample_interval_sec * fps)))

    frame_idx = 0
    crop_xs: list[float] = []

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t_sec = round(frame_idx / fps, 3)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Motion energy relative to previous sampled frame
        motion = 0.0
        if prev_gray is not None and prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, prev_gray)
            motion = round(float(np.mean(diff)) / 255.0, 4)
        prev_gray = gray

        # Focus / sharpness via variance of Laplacian
        sharpness = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)

        faces_out: list[FaceDetectionBox] = []
        crop_cx = 0.5
        if cascades:
            scale = 640.0 / max(w, 1)
            small = cv2.resize(gray, (640, max(1, int(h * scale)))) if w > 640 else gray
            sh, sw = small.shape[:2]
            rects: list[tuple[int, int, int, int]] = []
            for fc in cascades:
                found = fc.detectMultiScale(small, scaleFactor=1.12, minNeighbors=3, minSize=(28, 28))
                if len(found) > 0:
                    rects = list(found)
                    break
            for (fx, fy, fw, fh) in rects:
                cx = (fx + fw * 0.5) / sw
                cy = (fy + fh * 0.5) / sh
                faces_out.append(
                    FaceDetectionBox(
                        time_sec=t_sec,
                        x_norm=round(fx / sw, 4),
                        y_norm=round(fy / sh, 4),
                        w_norm=round(fw / sw, 4),
                        h_norm=round(fh / sh, 4),
                        center_x_norm=round(cx, 4),
                        center_y_norm=round(cy, 4),
                    )
                )
            if faces_out:
                primary = max(faces_out, key=lambda b: b.w_norm * b.h_norm)
                crop_cx = primary.center_x_norm
                crop_xs.append(crop_cx)

        if not faces_out:
            skin_cx = _estimate_subject_center_x(frame)
            if skin_cx is not None:
                crop_cx = skin_cx
                crop_xs.append(skin_cx)

        ocr_txt = ""
        if run_ocr and (frame_idx == 0 or (frame_idx // step_frames) % 5 == 0):
            ocr_txt = _try_apple_vision_ocr(frame)

        samples.append(
            VisualContextSample(
                time_sec=t_sec,
                faces=faces_out,
                ocr_text=ocr_txt,
                motion_score=motion,
                sharpness_score=sharpness,
                suggested_crop_center_x=crop_cx,
            )
        )
        frame_idx += step_frames

    cap.release()
    median_crop = round(float(np.median(crop_xs)), 4) if crop_xs else 0.5
    # Fallback safeguard if duration was empty
    if not samples and duration >= 0:
        samples.append(VisualContextSample(time_sec=0.0, suggested_crop_center_x=0.5))
    return samples, median_crop
