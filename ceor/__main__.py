"""Command-line entry points for the three stages of CEOR."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_parser():
    parser = argparse.ArgumentParser(description="CEOR: OSD, grouped probes, frozen-answer occlusion")
    sub = parser.add_subparsers(dest="command", required=True)
    osd = sub.add_parser("osd", help="Generate auxiliary draws and recover OSD labels")
    osd.add_argument("--input", type=Path, required=True, help="JSONL with GT and fixed target draws")
    osd.add_argument("--output", type=Path, required=True)
    osd.add_argument("--reference-config", type=Path, help="Omit to use input auxiliary_predictions")
    osd.add_argument("--target-model", help="Target model ID/path; required for live auxiliaries")
    osd.add_argument("--cache-dir", type=Path, default=Path("experiments/cache/osd"))
    osd.add_argument("--num-samples", type=int, default=10)
    osd.add_argument("--temperature", type=float, default=0.8)
    osd.add_argument("--seed", type=int, default=10)
    osd.add_argument("--min-models", type=int, default=2)
    osd.add_argument("--iou-threshold", type=float, default=0.5)
    osd.add_argument("--smoothing", type=float, default=0.01)
    for name, help_text in (
        ("prepare-labels", "Select auxiliary models on train; freeze and recompute labels"),
        ("collect-heads", "Capture prompt-final pre-o_proj head states from generation records"),
    ):
        sub.add_parser(name, help=help_text, add_help=False)
    probe = sub.add_parser("fit-probes", help="Grouped OOF Ridge probes and all-head Key1 selection")
    probe.add_argument("--features-dir", type=Path, required=True)
    probe.add_argument("--output-dir", type=Path, required=True)
    probe.add_argument("--num-folds", type=int, default=5)
    probe.add_argument("--ridge-alpha", type=float, default=10.0)
    probe.add_argument("--jobs", type=int, default=1)
    probe.add_argument("--exclude-validation-group-overlap", action="store_true")
    score = sub.add_parser("score", help="Score saved answers without regenerating them")
    source = score.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSONL: sample_id, image_path, question, frozen_answer")
    source.add_argument("--run-dir", type=Path, help="generate_answers.py run directory or its files/")
    score.add_argument("--split", choices=("train", "validation"), default="validation")
    score.add_argument("--model", required=True, help="Same Qwen3-VL model ID/path used by the probe")
    score.add_argument("--key-layer", type=Path, required=True, help="fit-probes output key_layer.json")
    score.add_argument("--features-dir", type=Path, help="Reuse verified clean features; saves one prefill")
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--limit", type=int, default=0, help="0 uses all inputs")
    score.add_argument("--mask-method", choices=("image_mean", "black", "white", "gaussian_blur"),
                       default="image_mean")
    score.add_argument("--save-head-responses", action="store_true", help="Include all per-head ablation features")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "collect-heads":
        from experiments.collect_qwen_uncertainty_heads import build_parser as collect_parser, main as collect
        return collect(collect_parser().parse_args(argv[1:]))
    if argv and argv[0] == "prepare-labels":
        from experiments.prepare_lgd_quality_filtered_labels import main as prepare
        return prepare(argv[1:])
    args = build_parser().parse_args(argv)
    if args.command == "osd":
        from ceor.osd import generate_osd
        return generate_osd(args)
    if args.command == "fit-probes":
        from ceor.probe import fit_probes
        return fit_probes(args)
    if args.command == "score":
        if args.limit < 0:
            raise ValueError("--limit must be non-negative.")
        from ceor.inference import score
        return score(args)


def cli():
    # Console-script launchers pass the return value to sys.exit().
    main()


if __name__ == "__main__":
    main()
