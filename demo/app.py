"""Gradio-powered local demo for RevUtil review analysis."""

from __future__ import annotations

import argparse
from functools import lru_cache
from typing import Dict, List, Tuple

import gradio as gr
import pandas as pd

from inference_engine import ASPECTS, DemoConfig, RevUtilEngine, split_review_text


DEFAULT_SAMPLE = """The submission proposes a diffusion-based reviewer assistant. The idea is novel, but the experiments do not cover large-scale settings and the writing sometimes hides the actual failure cases."""

HERO_MARKDOWN = f"""
# The Good, the Bad & the Constructive

Explore how reviewer comments score across **Actionability**, **Grounding & Specificity**, **Verifiability**, and **Helpfulness**—the four aspects highlighted in the RevUtil paper for measuring author-facing utility.

> Providing constructive feedback to paper authors is a core component of peer review. RevUtil pairs 1.4k human-labeled comments with 10k GPT-4o rationales to benchmark models that rival GPT-4o consistency while helping authors focus on actionable fixes.

**Quick links**

- 📄 Paper: [RevUtil on arXiv](https://www.arxiv.org/abs/2509.04484)
- 👥 Human dataset: [boda/RevUtil_human](https://huggingface.co/datasets/boda/RevUtil_human)
- 🤖 Synthetic dataset: [boda/RevUtil_synthetic](https://huggingface.co/datasets/boda/RevUtil_synthetic)
- 🧠 Models: [Score only](https://huggingface.co/boda/RevUtil_Llama-3.1-8B-Instruct_score_only) · [Score + rationale](https://huggingface.co/boda/RevUtil_Llama-3.1-8B-Instruct_score_rationale)
- 📎 License: [CC-BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the RevUtil Gradio demo.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for Gradio.")
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to bind the demo server."
    )
    parser.add_argument(
        "--share", action="store_true", help="Enable Gradio public sharing tunnel."
    )
    parser.add_argument(
        "--adapter",
        default=DemoConfig.adapter_name,
        help="Hugging Face LoRA adapter to load.",
    )
    parser.add_argument(
        "--base-model",
        default=DemoConfig.base_model_name,
        help="Base LLM to pair with the adapter.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Bypass model loading and return deterministic mock outputs.",
    )
    return parser.parse_args()


@lru_cache(maxsize=4)
def get_engine(mock: bool, adapter: str, base_model: str) -> RevUtilEngine:
    """Lazily construct (and cache) the RevUtil engine."""

    config = DemoConfig(adapter_name=adapter, base_model_name=base_model)
    engine = RevUtilEngine(config=config, mock=mock)
    if not mock:
        engine.warm_up(DEFAULT_SAMPLE)
    return engine


TABLE_COLUMNS = ["Point #", "Point excerpt", "Aspect", "Score", "Rationale"]


ASPECT_DISPLAY_NAMES = {
    "actionability": "Actionability",
    "grounding_specificity": "Grounding & Specificity",
    "verifiability": "Verifiability",
    "helpfulness": "Helpfulness",
}


SCORE_COLORS = {
    1: "#e74c3c",
    2: "#e67e22",
    3: "#f1c40f",
    4: "#27ae60",
    5: "#1e8449",
}


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame(columns=TABLE_COLUMNS)


def score_to_color(score) -> str:
    if score is None:
        return "#95a5a6"
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "#95a5a6"
    value = max(1.0, min(5.0, value))
    rounded = int(round(value))
    return SCORE_COLORS.get(rounded, "#95a5a6")


def rows_from_point_results(point_results: List[dict]) -> List[dict]:
    rows = []
    for result in point_results:
        idx = result["index"]
        text = result["text"].strip()
        excerpt = text if len(text) <= 160 else text[:157] + "..."
        for aspect in ASPECTS:
            aspect_data = result["aspects"].get(aspect, {})
            rows.append(
                {
                    "Point #": idx,
                    "Point excerpt": excerpt,
                    "Aspect": ASPECT_DISPLAY_NAMES.get(aspect, aspect.title()),
                    "Score": aspect_data.get("score") or "–",
                    "Rationale": (
                        aspect_data.get("rationale") or "No rationale provided."
                    ).strip(),
                }
            )
    return rows


def compute_aggregates(point_results: List[dict]) -> Dict[str, Dict[str, float]]:
    aggregates: Dict[str, Dict[str, float]] = {}
    overall_values: List[float] = []

    def to_float(score):
        if score is None:
            return None
        score_str = str(score).strip()
        normalized = score_str.upper()
        if not normalized:
            return None
        if normalized.startswith("X"):
            # X indicates "No Claim" or similar sentinel; exclude from averages.
            return None
        try:
            return float(score_str)
        except ValueError:
            return None

    for aspect in ASPECTS:
        values: List[float] = []
        for result in point_results:
            aspect_score = result["aspects"].get(aspect, {}).get("score")
            numeric = to_float(aspect_score)
            if numeric is not None:
                values.append(numeric)
                overall_values.append(numeric)
        avg = sum(values) / len(values) if values else None
        aggregates[aspect] = {"average": avg, "count": len(values)}

    overall_avg = sum(overall_values) / len(overall_values) if overall_values else None
    aggregates["overall"] = {"average": overall_avg, "count": len(overall_values)}
    return aggregates


def render_aggregate_cards(
    aggregates: Dict[str, Dict[str, float]], point_count: int
) -> str:
    cards = []
    for aspect in ASPECTS:
        data = aggregates.get(aspect, {})
        avg = data.get("average")
        count = data.get("count", 0)
        color = score_to_color(avg)
        display_score = "–" if avg is None else f"{avg:.2f}"
        cards.append(
            f"<div style='flex:1; min-width:220px; background:{color}; color:white; padding:16px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.1);'>"
            f"<div style='font-size:0.9rem; text-transform:uppercase; opacity:0.85;'>{ASPECT_DISPLAY_NAMES.get(aspect, aspect.title())}</div>"
            f"<div style='font-size:2rem; font-weight:600;'>{display_score}</div>"
            f"<div style='font-size:0.85rem;'>Based on {count} score(s)</div>"
            "</div>"
        )

    overall = aggregates.get("overall", {})
    overall_color = score_to_color(overall.get("average"))
    overall_display = (
        "–" if overall.get("average") is None else f"{overall['average']:.2f}"
    )
    cards.append(
        f"<div style='flex:1; min-width:220px; background:{overall_color}; color:white; padding:16px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.1);'>"
        "<div style='font-size:0.9rem; text-transform:uppercase; opacity:0.85;'>Overall</div>"
        f"<div style='font-size:2rem; font-weight:600;'>{overall_display}</div>"
        f"<div style='font-size:0.85rem;'>Across {point_count} point(s)</div>"
        "</div>"
    )

    container = (
        "<div style='display:flex; gap:16px; flex-wrap:wrap;'>"
        + "".join(cards)
        + "</div>"
    )
    return container


def summarize_aggregates(
    aggregates: Dict[str, Dict[str, float]], point_count: int
) -> str:
    lines = [f"Processed **{point_count}** review point(s)."]

    scored_aspects: List[Tuple[str, float]] = [
        (aspect, data["average"])
        for aspect, data in aggregates.items()
        if aspect in ASPECTS and data.get("average") is not None
    ]

    if scored_aspects:
        best_aspect, best_score = max(scored_aspects, key=lambda item: item[1])
        worst_aspect, worst_score = min(scored_aspects, key=lambda item: item[1])
        lines.append(
            f"- **{ASPECT_DISPLAY_NAMES.get(best_aspect, best_aspect.title())}** leads with an average score of {best_score:.2f}."
        )
        if best_aspect != worst_aspect:
            lines.append(
                f"- **{ASPECT_DISPLAY_NAMES.get(worst_aspect, worst_aspect.title())}** trails at {worst_score:.2f}; consider prioritizing improvements here."
            )

    overall = aggregates.get("overall", {})
    if overall.get("average") is not None:
        lines.append(f"- Overall weighted score: **{overall['average']:.2f}**.")
    else:
        lines.append("- Overall score unavailable (no numeric labels returned).")

    return "\n".join(lines)


def format_prompts(point_results: List[dict]) -> str:
    sections = []
    for result in point_results:
        sections.append(f"### Point {result['index']}\n{result['prompt']}")
    return "\n\n".join(sections)


def build_analyzer(args: argparse.Namespace):
    def _analyze(review_blob: str, mock_toggle: bool):
        mock_mode = args.mock or mock_toggle
        points = split_review_text(review_blob)
        if not points:
            warning = "### Please enter at least one review point."
            return "", _empty_table(), warning, ""

        try:
            engine = get_engine(mock_mode, args.adapter, args.base_model)
            point_results = engine.analyze_points(points)
            table = pd.DataFrame(rows_from_point_results(point_results))
            aggregates = compute_aggregates(point_results)
            agg_cards = render_aggregate_cards(aggregates, len(points))
            summary = summarize_aggregates(aggregates, len(points))
            prompts = format_prompts(point_results)
            return agg_cards, table, summary, prompts
        except Exception as exc:  # pylint: disable=broad-except
            warning = f"### ⚠️ Could not analyze review\n{exc}"
            return "", _empty_table(), warning, ""

    return _analyze


def build_interface(args: argparse.Namespace) -> gr.Blocks:
    analyze_fn = build_analyzer(args)

    with gr.Blocks(title="RevUtil Demo") as demo:
        gr.Markdown(HERO_MARKDOWN)

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                review_input = gr.Textbox(
                    label="Paste a review comment",
                    placeholder="Summarize the tone, strengths, weaknesses, and actionable feedback...",
                    lines=10,
                    value=DEFAULT_SAMPLE,
                )
                mock_checkbox = gr.Checkbox(
                    value=args.mock,
                    label="Use mock output (for CPU-only preview)",
                )
                analyze_button = gr.Button("Analyze Review", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown(
                    """
### How to read the scores

- **Actionability** – Scores how clearly the comment spells out concrete actions; 1 means the authors have no idea what to do next, while 5 pairs explicit actions with concrete guidance.
- **Grounding & Specificity** – Measures whether the reviewer pinpoints a part of the paper and explains what’s wrong, ranging from unguided remarks (1) to fully grounded, specific pointers (5).
- **Verifiability** – Rates how well each claim is justified with reasoning, common knowledge, or citations. Use 1–5 when a claim exists; "X" appears when there’s no claim to verify.
- **Helpfulness** – Captures the overall usefulness of the feedback: low scores are vague or non-actionable, whereas a 5 delivers detailed, constructive suggestions that materially improve the draft.
"""
                )

        aggregate_html = gr.HTML(label="Aggregate scores")
        results_table = gr.Dataframe(
            headers=TABLE_COLUMNS,
            label="Per-point breakdown",
            wrap=True,
            interactive=False,
        )
        summary_md = gr.Markdown(label="Insights", value="Waiting for input...")
        with gr.Accordion("Show generated prompt", open=False):
            prompt_box = gr.Textbox(
                lines=8, interactive=False, label="Instruction Prompt"
            )

        analyze_button.click(
            analyze_fn,
            inputs=[review_input, mock_checkbox],
            outputs=[aggregate_html, results_table, summary_md, prompt_box],
        )

    return demo


def main():
    args = parse_args()
    demo = build_interface(args)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
