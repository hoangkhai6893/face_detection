#!/usr/bin/env python3
"""
validate_backends.py — A/B accuracy + speed comparison between YOLO+dlib and YuNet+SFace.

This script is the GATE for Phase 3 migration:
  - Run BOTH backends on the same held-out test images
  - Compare accuracy, false-unknown rate, confusion rate, and latency
  - Print a clear PASS / FAIL / NEEDS_TUNING decision

Usage:
    python scripts/validate_backends.py \\
        --test-dir test_data/ \\
        --encodings model/face_encodings_hybrid.pkl \\
        [--report validation_report.json]

Test data layout:
    test_data/
        Khai/       # 10-20 images NOT used in training
        MaiAnh/
        Unknown/    # images of people NOT in the family

The encoding .pkl MUST have been built with the same backend you want to test.
For a fair A/B test, rebuild encodings with each backend and run separately:
    FACE_BACKEND=dlib  → rebuild → validate_backends.py --encodings model/enc_dlib.pkl
    FACE_BACKEND=sface → rebuild → validate_backends.py --encodings model/enc_sface.pkl
"""

import argparse
import json
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Make src/ importable from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(
    results: List[Tuple[str, str, float]],   # [(true_label, predicted, ms)]
    known_names: set,
) -> dict:
    """Compute accuracy metrics from (true_label, predicted_name, latency_ms) triples."""
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "false_unknown_rate": 0.0,
                "confusion_rate": 0.0, "avg_ms": 0.0, "total": 0}

    correct = 0
    false_unknown = 0   # known person → "Unknown"
    confused = 0        # known A → known B (wrong person)
    total_ms = 0.0

    for true_label, predicted, ms in results:
        total_ms += ms
        is_known = true_label in known_names

        pred_name = predicted.split(" (")[0].strip()   # strip "(0.85)" confidence suffix

        if true_label == "Unknown":
            if pred_name == "Unknown":
                correct += 1
        else:
            if pred_name == true_label:
                correct += 1
            elif pred_name == "Unknown" and is_known:
                false_unknown += 1
            elif pred_name in known_names and pred_name != true_label:
                confused += 1

    known_total = sum(1 for t, _, _ in results if t != "Unknown")
    return {
        "accuracy":           correct / total,
        "false_unknown_rate": false_unknown / max(1, known_total),
        "confusion_rate":     confused / max(1, known_total),
        "avg_ms":             total_ms / total,
        "total":              total,
    }


# ---------------------------------------------------------------------------
# Recognition helper
# ---------------------------------------------------------------------------

def _recognize(
    frame_bgr: np.ndarray,
    detector,
    embedder,
    known_encodings: List[np.ndarray],
    known_names: List[str],
    tolerance: float,
    confusion_margin: float,
    top_k: int,
) -> str:
    """Run one face through the detector+embedder and return the name."""
    detections = detector.detect(frame_bgr)
    if not detections:
        return "Unknown"

    best = max(detections, key=lambda d: d.confidence)
    enc = embedder.encode(frame_bgr, best, padding=config.RECOGNITION_PADDING)
    if enc is None:
        return "Unknown"

    distances = embedder.batch_distance(known_encodings, enc)

    k = min(top_k, len(distances))
    top_k_idx = np.argsort(distances)[:k]

    votes: dict = {}
    for idx in top_k_idx:
        if distances[idx] <= tolerance:
            person = known_names[idx]
            votes[person] = votes.get(person, 0.0) + (1.0 - distances[idx])

    if not votes:
        return "Unknown"

    winner = max(votes, key=lambda p: votes[p])
    best_dist = min(distances[i] for i in top_k_idx if known_names[i] == winner)

    if len(set(known_names)) > 1:
        second = min(
            (distances[i] for i, n in enumerate(known_names) if n != winner),
            default=float("inf"),
        )
        if second - best_dist < confusion_margin:
            return "Unknown"

    return f"{winner} ({1.0 - best_dist:.2f})"


# ---------------------------------------------------------------------------
# Validate one backend
# ---------------------------------------------------------------------------

def _validate_backend(
    backend: str,
    test_dir: str,
    encodings_path: str,
    tolerance: float,
    confusion_margin: float,
) -> Tuple[dict, List[Tuple[str, str, float]]]:
    """Run validation for one backend. Returns (metrics, raw_results)."""
    from core.detector import YoloDetector, YuNetDetector
    from core.embedder import DlibEmbedder, SFaceEmbedder

    print(f"\n{'='*60}")
    print(f"  Validating backend: {backend}")
    print(f"  Encodings : {encodings_path}")

    if backend == "sface":
        detector = YuNetDetector(model_path=config.YUNET_MODEL_PATH)
        embedder = SFaceEmbedder(model_path=config.SFACE_MODEL_PATH)
        tol = config.SFACE_TOLERANCE
        margin = config.SFACE_CONFUSION_MARGIN
    else:
        detector = YoloDetector(
            model_path=config.MODEL_PATH,
            detection_confidence=config.DETECTION_CONFIDENCE,
            input_width=config.YOLO_INPUT_WIDTH,
        )
        embedder = DlibEmbedder()
        tol = tolerance
        margin = confusion_margin

    # Safe: encodings_path is a local file generated by our own encoder.py
    # (never loaded from network or untrusted sources).
    with open(encodings_path, "rb") as f:
        data = pickle.load(f)
    known_encodings: List[np.ndarray] = data["encodings"]
    known_names:     List[str]         = data["names"]
    known_set = set(known_names)

    raw_results: List[Tuple[str, str, float]] = []

    for person_name in sorted(os.listdir(test_dir)):
        person_dir = os.path.join(test_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            img = cv2.imread(os.path.join(person_dir, fname))
            if img is None:
                continue

            t0 = time.time()
            predicted = _recognize(
                img, detector, embedder, known_encodings, known_names,
                tol, margin, config.RECOGNITION_TOP_K,
            )
            ms = (time.time() - t0) * 1000

            raw_results.append((person_name, predicted, ms))
            pred_short = predicted.split(" (")[0]
            ok = "✓" if (
                (person_name == "Unknown" and pred_short == "Unknown") or
                (person_name != "Unknown" and pred_short == person_name)
            ) else "✗"
            print(f"  {ok} {person_name:12s} → {pred_short:12s}  ({ms:.0f}ms)")

    metrics = _compute_metrics(raw_results, known_set)
    return metrics, raw_results


# ---------------------------------------------------------------------------
# Gate decision
# ---------------------------------------------------------------------------

def _decide(a: dict, b: dict) -> str:
    """
    PASS        → proceed to Phase 3 migration
    FAIL        → keep stack A (YOLO+dlib), accuracy regression
    NEEDS_TUNING → B is close but not quite — try tuning SFACE_TOLERANCE
    """
    # Zero tolerance on confusion: mistaking one family member for another
    if b["confusion_rate"] > a["confusion_rate"]:
        return "FAIL"

    # Core accuracy gate: B must match or beat A
    if b["accuracy"] >= a["accuracy"] and b["false_unknown_rate"] <= a["false_unknown_rate"] + 0.05:
        return "PASS"

    # Within 5% margin: try tuning thresholds
    if b["accuracy"] >= a["accuracy"] - 0.05:
        return "NEEDS_TUNING"

    return "FAIL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate YuNet+SFace vs YOLO+dlib on held-out family images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-dir", required=True,
                        help="Test data root (one folder per person + Unknown/)")
    parser.add_argument("--encodings-dlib", default=None,
                        help="Encodings .pkl built with dlib (default: model/face_encodings_hybrid.pkl)")
    parser.add_argument("--encodings-sface", default=None,
                        help="Encodings .pkl built with sface (default: same as --encodings-dlib)")
    parser.add_argument("--report", default=None,
                        help="Write JSON report to this file (optional)")
    parser.add_argument("--tolerance", type=float, default=config.TOLERANCE)
    parser.add_argument("--confusion-margin", type=float, default=config.CONFUSION_MARGIN)
    args = parser.parse_args()

    default_enc = os.path.join(config.ENCODINGS_DIR, "face_encodings_hybrid.pkl")
    enc_dlib  = args.encodings_dlib  or default_enc
    enc_sface = args.encodings_sface or enc_dlib

    if not os.path.isdir(args.test_dir):
        print(f"ERROR: test-dir not found: {args.test_dir}")
        sys.exit(1)

    # --- Validate both backends ---
    metrics_a, raw_a = _validate_backend(
        "dlib", args.test_dir, enc_dlib,
        args.tolerance, args.confusion_margin,
    )
    metrics_b, raw_b = _validate_backend(
        "sface", args.test_dir, enc_sface,
        args.tolerance, args.confusion_margin,
    )

    # --- Decision ---
    decision = _decide(metrics_a, metrics_b)

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Metric':<25s} {'YOLO+dlib':>12s} {'YuNet+SFace':>12s}")
    print(f"  {'-'*50}")
    for key in ("accuracy", "false_unknown_rate", "confusion_rate", "avg_ms"):
        a_val = metrics_a[key]
        b_val = metrics_b[key]
        unit  = "ms" if key == "avg_ms" else "%"
        fmt_a = f"{a_val*100:.1f}{unit}" if unit == "%" else f"{a_val:.1f}{unit}"
        fmt_b = f"{b_val*100:.1f}{unit}" if unit == "%" else f"{b_val:.1f}{unit}"
        print(f"  {key:<25s} {fmt_a:>12s} {fmt_b:>12s}")

    print(f"\n  {'DECISION':<25s} → {decision}")
    if decision == "PASS":
        print("  ✅ YuNet+SFace accuracy is acceptable. Proceed to Phase 3.")
    elif decision == "NEEDS_TUNING":
        print("  ⚠️  YuNet+SFace is close. Try tuning SFACE_TOLERANCE in .env.")
    else:
        print("  ❌ YuNet+SFace accuracy worse. Keep YOLO+dlib. Do NOT migrate.")

    # --- JSON report ---
    report = {
        "backend_a": {"name": "dlib", **metrics_a},
        "backend_b": {"name": "sface", **metrics_b},
        "decision":  decision,
    }
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report written → {args.report}")

    sys.exit(0 if decision == "PASS" else 1)


if __name__ == "__main__":
    main()
