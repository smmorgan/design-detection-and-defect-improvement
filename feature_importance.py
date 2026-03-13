#!/usr/bin/env python3
"""
Feature importance analysis using Integrated Gradients (Sundararajan et al., 2017).

Computes token-level attributions for the design classifier, then aggregates
to find the most influential words for design vs non-design predictions.

Usage:
    python feature_importance.py
    python feature_importance.py --model gcp_results/roberta_conservative_0309_0738
    python feature_importance.py --n_samples 200 --n_steps 50
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PRETRAINED_MODEL = "gcp_results/roberta_conservative_0309_0738"
TAWOS_MANUAL = "./output/all_manually_labelled.csv"
MAX_LENGTH = 512


def load_model(model_path, device):
    """Load model and tokenizer from checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
    model.to(device)
    model.eval()
    model.zero_grad()
    return model, tokenizer


def make_forward_fn(model):
    """Create a forward function that takes input_ids and attention_mask."""
    def forward_fn(input_ids, attention_mask):
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits
    return forward_fn


def compute_attributions(model, tokenizer, text, target_class, device, n_steps=50):
    """Compute Integrated Gradients attributions for a single text."""
    # Tokenize
    inputs = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Get embedding layer
    if hasattr(model, "roberta"):
        embed_layer = model.roberta.embeddings
    elif hasattr(model, "bert"):
        embed_layer = model.bert.embeddings
    elif hasattr(model, "distilbert"):
        embed_layer = model.distilbert.embeddings
    else:
        raise ValueError("Unknown model architecture")

    # Baseline: PAD token ids
    pad_id = tokenizer.pad_token_id
    baseline_ids = torch.full_like(input_ids, pad_id)

    # Set up Layer Integrated Gradients on the embedding layer
    lig = LayerIntegratedGradients(
        make_forward_fn(model),
        embed_layer,
    )

    # Compute attributions (internal_batch_size limits memory for interpolation steps)
    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=target_class,
        n_steps=n_steps,
        internal_batch_size=5,
        return_convergence_delta=True,
    )

    # Sum attribution across embedding dimensions to get per-token score
    attr_scores = attributions.sum(dim=-1).squeeze(0)  # (seq_len,)
    norm = torch.norm(attr_scores)
    if norm > 0:
        attr_scores = attr_scores / norm  # normalize

    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    # Get actual token count (non-padding)
    actual_len = int(attention_mask.sum().item())

    return tokens[:actual_len], attr_scores[:actual_len].detach().cpu().numpy()


def aggregate_word_importance(all_token_attrs, tokenizer):
    """Aggregate token-level attributions to word level.

    Uses the tokenizer to decode full words, avoiding subword fragment issues.
    Returns dict mapping words to their mean attribution scores.
    """
    word_scores = defaultdict(list)
    special_tokens = {"<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"}

    for tokens, scores in all_token_attrs:
        # Group consecutive tokens into words
        # In RoBERTa: Ġ prefix = word start, no prefix = continuation
        current_word_tokens = []
        current_word_scores = []

        def flush_word():
            if current_word_tokens:
                # Reconstruct word from tokens
                word = "".join(current_word_tokens).strip().lower()
                # Filter: must be alphabetic and len > 2
                if word and len(word) > 2 and any(c.isalpha() for c in word):
                    word_scores[word].append(float(np.mean(current_word_scores)))

        for token, score in zip(tokens, scores):
            if token in special_tokens:
                flush_word()
                current_word_tokens = []
                current_word_scores = []
                continue

            # RoBERTa: Ġ prefix means new word
            if token.startswith("Ġ"):
                flush_word()
                current_word_tokens = [token[1:]]  # strip Ġ
                current_word_scores = [score]
            # BERT: ## prefix means continuation
            elif token.startswith("##"):
                current_word_tokens.append(token[2:])
                current_word_scores.append(score)
            else:
                # No prefix: continuation of current word (RoBERTa subword)
                # OR first token in sequence
                if not current_word_tokens:
                    current_word_tokens = [token]
                    current_word_scores = [score]
                else:
                    current_word_tokens.append(token)
                    current_word_scores.append(score)

        flush_word()

    return word_scores


def generate_heatmap_html(sample_records, out_path, n_examples=3):
    """Generate an HTML file with token-level attribution heatmaps.

    Selects representative examples: best TP, best TN, a FN, and a FP
    (where available), plus high-confidence correct predictions.
    """
    # Select examples: pick best from each category
    examples = []
    for cat in ["TP", "TN", "FP", "FN"]:
        candidates = [r for r in sample_records if r["category"] == cat]
        if candidates:
            # Pick highest confidence for TP/TN, lowest for FP/FN
            if cat in ("TP", "TN"):
                best = max(candidates, key=lambda r: r["confidence"])
            else:
                best = min(candidates, key=lambda r: r["confidence"])
            examples.append(best)

    # If we have room, add more diverse examples
    for cat in ["TP", "TN"]:
        candidates = sorted(
            [r for r in sample_records if r["category"] == cat and r not in examples],
            key=lambda r: r["confidence"], reverse=True,
        )
        for c in candidates[:max(0, n_examples - len([e for e in examples if e["category"] == cat]))]:
            examples.append(c)

    # Generate HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>Feature Attribution Heatmaps</title>",
        "<style>",
        "  body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; }",
        "  h1 { color: #333; border-bottom: 2px solid #666; padding-bottom: 10px; }",
        "  .example { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "  .meta { font-size: 14px; color: #666; margin-bottom: 12px; }",
        "  .meta span { margin-right: 20px; }",
        "  .category { font-weight: bold; padding: 2px 8px; border-radius: 4px; color: white; }",
        "  .category-TP { background: #28a745; }",
        "  .category-TN { background: #007bff; }",
        "  .category-FP { background: #dc3545; }",
        "  .category-FN { background: #fd7e14; }",
        "  .heatmap { line-height: 2.0; font-size: 15px; word-wrap: break-word; }",
        "  .token { padding: 2px 1px; border-radius: 3px; }",
        "  .legend { display: flex; align-items: center; gap: 10px; margin: 20px 0; font-size: 14px; }",
        "  .legend-bar { width: 300px; height: 20px; border-radius: 4px; ",
        "    background: linear-gradient(to right, rgba(0,100,255,0.5), rgba(255,255,255,0), rgba(255,50,0,0.5)); }",
        "  .method-note { font-size: 13px; color: #888; margin-top: 30px; border-top: 1px solid #ddd; padding-top: 10px; }",
        "</style>",
        "</head><body>",
        "<h1>Token Attribution Heatmaps</h1>",
        "<p>Integrated Gradients attributions showing which tokens most influence the model's output.</p>",
        "<div class='legend'>",
        "  <span>Negative (against class)</span>",
        "  <div class='legend-bar'></div>",
        "  <span>Positive (supports class)</span>",
        "</div>",
    ]

    for rec in examples:
        tokens = rec["tokens"]
        scores = rec["scores"]
        true_str = "Design" if rec["true_label"] == 1 else "Non-Design"
        pred_str = "Design" if rec["pred"] == 1 else "Non-Design"
        cat = rec["category"]

        html_parts.append(f"<div class='example'>")
        html_parts.append(f"<div class='meta'>")
        html_parts.append(f"  <span><b>{rec['issue_key']}</b> ({rec['project']})</span>")
        html_parts.append(f"  <span>True: <b>{true_str}</b></span>")
        html_parts.append(f"  <span>Predicted: <b>{pred_str}</b> ({rec['confidence']:.1%})</span>")
        html_parts.append(f"  <span class='category category-{cat}'>{cat}</span>")
        html_parts.append(f"</div>")

        # Build token spans with color
        html_parts.append("<div class='heatmap'>")
        # Use scores to color tokens; clamp to [-1, 1] range
        max_abs = max(abs(scores.max()), abs(scores.min()), 1e-6)
        for token, score in zip(tokens, scores):
            # Skip special tokens in display
            if token in ("<s>", "</s>", "<pad>", "[CLS]", "[PAD]"):
                continue

            # Clean token for display
            display = token
            if display.startswith("Ġ"):
                display = " " + display[1:]
            elif display.startswith("##"):
                display = display[2:]
            if display == "[SEP]":
                display = " | "

            # Normalize score to [-1, 1]
            norm_score = float(score) / max_abs

            # Color: positive = red (supports class), negative = blue (against class)
            if norm_score > 0:
                r, g, b = 255, int(255 * (1 - norm_score)), int(255 * (1 - norm_score))
                alpha = min(0.7, abs(norm_score))
            else:
                r, g, b = int(255 * (1 + norm_score)), int(255 * (1 + norm_score)), 255
                alpha = min(0.7, abs(norm_score))

            html_parts.append(
                f"<span class='token' style='background: rgba({r},{g},{b},{alpha:.2f})'>"
                f"{display}</span>"
            )
        html_parts.append("</div></div>")

    html_parts.append("<div class='method-note'>")
    html_parts.append("Method: Layer Integrated Gradients (Sundararajan et al., 2017) on RoBERTa embedding layer.<br>")
    html_parts.append("Red = token supports the attributed class. Blue = token opposes it.<br>")
    html_parts.append("Attributions are normalized per-sample.")
    html_parts.append("</div>")
    html_parts.append("</body></html>")

    with open(out_path, "w") as f:
        f.write("\n".join(html_parts))
    logger.info(f"Heatmap HTML saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=PRETRAINED_MODEL)
    parser.add_argument("--data", default=TAWOS_MANUAL)
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of samples to analyze (stratified)")
    parser.add_argument("--n_steps", type=int, default=50,
                        help="Integration steps for IG (higher = more accurate)")
    parser.add_argument("--out", default="gcp_results/feature_importance.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model, tokenizer = load_model(args.model, device)
    logger.info(f"Model loaded from {args.model}")

    # Load data and stratified sample
    df = pd.read_csv(args.data)
    df["label_int"] = (df["label"] == "design").astype(int)

    n_per_class = args.n_samples // 2
    design = df[df["label_int"] == 1].sample(n=min(n_per_class, len(df[df["label_int"] == 1])),
                                              random_state=args.seed)
    nondesign = df[df["label_int"] == 0].sample(n=min(n_per_class, len(df[df["label_int"] == 0])),
                                                 random_state=args.seed)
    sample = pd.concat([design, nondesign]).sample(frac=1, random_state=args.seed)
    logger.info(f"Analyzing {len(sample)} samples ({len(design)} design, {len(nondesign)} non-design)")

    # Compute attributions
    design_attrs = []  # attributions for design-labelled samples
    nondesign_attrs = []  # attributions for non-design-labelled samples
    correct_design = 0
    correct_nondesign = 0
    # Store per-sample info for example selection
    sample_records = []

    for idx, (_, row) in enumerate(sample.iterrows()):
        text = row["text"]
        true_label = row["label_int"]
        issue_key = row.get("issue_key", f"sample_{idx}")
        project = row.get("project", "unknown")

        # Get model prediction first
        inputs = tokenizer(text, truncation=True, padding="max_length",
                           max_length=MAX_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            confidence = probs[0, pred].item()

        # Attribute toward TRUE label — shows what features the model associates
        # with each class, regardless of whether it got the prediction right.
        # This gives balanced coverage of both classes.
        tokens, scores = compute_attributions(model, tokenizer, text, true_label, device, args.n_steps)

        # Categorize: TP, TN, FP, FN
        if true_label == 1 and pred == 1:
            category = "TP"
        elif true_label == 0 and pred == 0:
            category = "TN"
        elif true_label == 0 and pred == 1:
            category = "FP"
        else:
            category = "FN"

        sample_records.append({
            "issue_key": issue_key,
            "project": project,
            "true_label": true_label,
            "pred": pred,
            "confidence": confidence,
            "category": category,
            "tokens": tokens,
            "scores": scores,
            "summary": row.get("summary", text[:100]),
        })

        if true_label == 1:
            design_attrs.append((tokens, scores))
            if pred == 1:
                correct_design += 1
        else:
            nondesign_attrs.append((tokens, scores))
            if pred == 0:
                correct_nondesign += 1

        if (idx + 1) % 25 == 0:
            logger.info(f"  Processed {idx + 1}/{len(sample)} samples")

    logger.info(f"True labels: {len(design_attrs)} design, {len(nondesign_attrs)} non-design")
    logger.info(f"Model accuracy on sample: design={correct_design}/{len(design_attrs)}, "
                f"non-design={correct_nondesign}/{len(nondesign_attrs)}")

    # Aggregate to word level
    design_words = aggregate_word_importance(design_attrs, tokenizer)
    nondesign_words = aggregate_word_importance(nondesign_attrs, tokenizer)

    # Filter: require word appears in at least 3 samples
    min_count = 3

    def top_words(word_scores, n=50):
        filtered = {w: scores for w, scores in word_scores.items()
                    if len(scores) >= min_count and len(w) > 1}
        ranked = sorted(filtered.items(), key=lambda x: np.mean(x[1]), reverse=True)
        return [
            {"word": w, "mean_attribution": round(float(np.mean(s)), 4),
             "std_attribution": round(float(np.std(s)), 4),
             "count": len(s)}
            for w, s in ranked[:n]
        ]

    top_design = top_words(design_words)
    top_nondesign = top_words(nondesign_words)

    # Print results
    print(f"\n{'='*70}")
    print("TOP WORDS DRIVING DESIGN PREDICTIONS")
    print(f"{'='*70}")
    print(f"{'Word':<25} {'Mean Attr':>12} {'Std':>10} {'Count':>8}")
    print("-" * 55)
    for entry in top_design[:30]:
        print(f"{entry['word']:<25} {entry['mean_attribution']:>12.4f} {entry['std_attribution']:>10.4f} {entry['count']:>8}")

    print(f"\n{'='*70}")
    print("TOP WORDS DRIVING NON-DESIGN PREDICTIONS")
    print(f"{'='*70}")
    print(f"{'Word':<25} {'Mean Attr':>12} {'Std':>10} {'Count':>8}")
    print("-" * 55)
    for entry in top_nondesign[:30]:
        print(f"{entry['word']:<25} {entry['mean_attribution']:>12.4f} {entry['std_attribution']:>10.4f} {entry['count']:>8}")

    # Generate per-sample heatmap visualizations
    heatmap_path = Path(args.out).with_suffix(".html")
    generate_heatmap_html(sample_records, heatmap_path)

    # Save results
    output = {
        "model": args.model,
        "method": "Integrated Gradients (Sundararajan et al., 2017)",
        "n_samples": len(sample),
        "attribution_target": "true_label",
        "n_design_samples": len(design_attrs),
        "n_nondesign_samples": len(nondesign_attrs),
        "n_steps": args.n_steps,
        "min_word_count": min_count,
        "top_design_words": top_design,
        "top_nondesign_words": top_nondesign,
    }
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
