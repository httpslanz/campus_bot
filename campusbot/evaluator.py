"""
ml_evaluator.py
---------------
Comprehensive Evaluation Module for Hybrid Chatbot Pipeline
Run:  python evaluator.py   (from your project root)
"""

# ── matplotlib MUST be configured before any other import ────────────────────
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import seaborn as sns
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pickle
import os
import csv
import json
from datetime import datetime
from collections import Counter

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_score
import django
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "campusbot.settings")
django.setup()

from chatbot.models import TrainingData, ModelVersion, Intent
from chatbot.ml_hybridpipeline import HybridChatbotPipeline


class ChatbotEvaluator:

    def __init__(self, k_folds: int = 5):
        self.k_folds    = k_folds
        self.pipeline   = HybridChatbotPipeline()
        self.model_data = self._load_model()
        self.timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_model(self):
        try:
            active = ModelVersion.objects.filter(is_active=True).latest("trained_at")
            if os.path.exists(active.model_path):
                with open(active.model_path, "rb") as f:
                    return pickle.load(f)
        except ModelVersion.DoesNotExist:
            pass
        raise RuntimeError("No active model found. Train the model first.")

    def _load_all_data(self):
        data = TrainingData.objects.filter(is_active=True).select_related("intent")
        questions_pre, questions_raw, intents = [], [], []
        for item in data:
            for q in item.get_questions():
                questions_pre.append(self.pipeline.preprocess_text(q))
                questions_raw.append(q)
                intents.append(item.intent.name)
        return questions_pre, questions_raw, intents

    def _vectorize(self, questions):
        return self.model_data["vectorizer"].transform(questions)

    def _svm_predict(self, X_vec):
        reverse   = self.model_data["reverse_mapping"]
        raw_preds = self.model_data["svm_model"].predict(X_vec)
        return [reverse[p] for p in raw_preds]

    def _svm_confidence(self, X_vec):
        scores = self.model_data["svm_model"].decision_function(X_vec)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        max_scores  = np.max(np.abs(scores), axis=1)
        confidences = (max_scores / (max_scores + 0.5)) * 100
        return np.clip(confidences, 0, 100)

    # ── 1. Classification metrics ─────────────────────────────────────────────

    def evaluate_classification(self, questions_pre, true_intents):
        X          = self._vectorize(questions_pre)
        pred_intents = self._svm_predict(X)
        labels     = sorted(set(true_intents))
        report_dict = classification_report(
            true_intents, pred_intents, labels=labels,
            output_dict=True, zero_division=0,
        )
        overall = {
            "accuracy":        accuracy_score(true_intents, pred_intents),
            "macro_precision": precision_score(true_intents, pred_intents, average="macro", zero_division=0),
            "macro_recall":    recall_score(true_intents, pred_intents, average="macro", zero_division=0),
            "macro_f1":        f1_score(true_intents, pred_intents, average="macro", zero_division=0),
        }
        per_intent = {
            intent: {
                "precision": report_dict[intent]["precision"],
                "recall":    report_dict[intent]["recall"],
                "f1":        report_dict[intent]["f1-score"],
                "support":   int(report_dict[intent]["support"]),
            }
            for intent in labels if intent in report_dict
        }
        cm = confusion_matrix(true_intents, pred_intents, labels=labels)
        return overall, per_intent, cm, labels, pred_intents

    # ── 2. Cross-validation ───────────────────────────────────────────────────

    def evaluate_cross_validation(self, questions_pre, true_intents):
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC

        clf = SkPipeline([
            ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 3),
                                      min_df=1, max_df=0.8, sublinear_tf=True)),
            ("svm",  LinearSVC(C=1.0, max_iter=10000, random_state=42)),
        ])
        label_counts = Counter(true_intents)
        min_count    = min(label_counts.values())
        n_splits     = min(self.k_folds, min_count)

        if n_splits < 2:
            return {"note": "Not enough samples per class (need >=2).",
                    "cv_scores": [], "mean_accuracy": None, "std_accuracy": None}

        skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, questions_pre, true_intents, cv=skf, scoring="accuracy")
        return {
            "n_folds":       n_splits,
            "cv_scores":     scores.tolist(),
            "mean_accuracy": float(scores.mean()),
            "std_accuracy":  float(scores.std()),
        }

    # ── 3. Semantic similarity ────────────────────────────────────────────────

    def evaluate_semantic_similarity(self, questions_raw):
        encoder          = self.model_data["sentence_encoder"]
        train_embeddings = self.model_data["training_embeddings"]
        query_embeddings = encoder.encode(questions_raw, show_progress_bar=False)
        sim_matrix       = cosine_similarity(query_embeddings, train_embeddings)

        max_sims = []
        for row in sim_matrix:
            sorted_sims = np.sort(row)[::-1]
            top_sim     = sorted_sims[1] if sorted_sims[0] > 0.999 else sorted_sims[0]
            max_sims.append(float(top_sim))

        return {
            "mean_similarity":     float(np.mean(max_sims)),
            "median_similarity":   float(np.median(max_sims)),
            "min_similarity":      float(np.min(max_sims)),
            "max_similarity":      float(np.max(max_sims)),
            "std_similarity":      float(np.std(max_sims)),
            "pct_above_threshold": float(np.mean(np.array(max_sims) >= 0.35) * 100),
            "_raw_sims":           max_sims,   # used for histogram
        }

    # ── 4. Threshold sensitivity ──────────────────────────────────────────────

    def evaluate_threshold_sensitivity(self, questions_pre, questions_raw, true_intents):
        encoder          = self.model_data["sentence_encoder"]
        train_embeddings = self.model_data["training_embeddings"]
        X_vec            = self._vectorize(questions_pre)
        confidences      = self._svm_confidence(X_vec)
        pred_intents     = self._svm_predict(X_vec)
        query_embeddings = encoder.encode(questions_raw, show_progress_bar=False)
        sim_matrix       = cosine_similarity(query_embeddings, train_embeddings)
        max_sims         = np.max(sim_matrix, axis=1)

        sim_thresholds  = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        conf_thresholds = [40, 45, 50, 55, 60, 65, 70]
        results = []
        total   = len(true_intents)

        for sim_t in sim_thresholds:
            for conf_t in conf_thresholds:
                mask       = (max_sims >= sim_t) & (confidences >= conf_t)
                n_accepted = int(np.sum(mask))
                acc        = accuracy_score(np.array(true_intents)[mask],
                                            np.array(pred_intents)[mask]) if n_accepted > 0 else 0.0
                results.append({
                    "sim_threshold":        sim_t,
                    "conf_threshold":       conf_t,
                    "accept_rate_pct":      round(n_accepted / total * 100, 1),
                    "accuracy_on_accepted": round(acc * 100, 2),
                    "n_accepted":           n_accepted,
                    "n_rejected":           total - n_accepted,
                })
        return results

    # ── 5. Intent coverage ────────────────────────────────────────────────────

    def evaluate_intent_coverage(self, true_intents):
        counts = Counter(true_intents)
        total  = len(true_intents)
        stats  = {}
        for intent, count in sorted(counts.items(), key=lambda x: -x[1]):
            stats[intent] = {
                "sample_count": count,
                "percentage":   round(count / total * 100, 2),
                "quality_flag": (
                    "EXCELLENT (20+ samples)" if count >= 20
                    else "GOOD (10+ samples)"  if count >= 10
                    else "FAIR (5+ samples)"   if count >= 5
                    else "POOR (under 5 samples)"
                ),
            }
        return stats

    # ── Master runner ─────────────────────────────────────────────────────────

    def run_full_evaluation(self):
        print("=" * 60)
        print("  CHATBOT EVALUATION SUITE")
        print("=" * 60)

        # Check if model has saved test set (proper data splitting)
        has_test_set = ('test_questions' in self.model_data and 
                        'test_intents' in self.model_data and
                        len(self.model_data['test_questions']) > 0)

        if has_test_set:
            print("\n[1/5] Loading held-out test set...")
            questions_pre = self.model_data['test_questions']
            true_intents  = self.model_data['test_intents']
            questions_raw = questions_pre  # test set only has preprocessed
            n = len(questions_pre)
            print(f"      {n} test samples (held-out from training)")
            print("      ✓ Using proper train/test split")
        else:
            print("\n[1/5] Loading data (WARNING: no train/test split)...")
            questions_pre, questions_raw, true_intents = self._load_all_data()
            n = len(questions_pre)
            print(f"      {n} samples, {len(set(true_intents))} intents")
            print("      ⚠ Evaluating on training data (inflated scores)")
            print("      → Retrain model with updated pipeline to fix this")

        print("[2/5] Classification metrics...")
        overall, per_intent, cm, labels, pred_intents = self.evaluate_classification(
            questions_pre, true_intents)

        print("[3/5] Cross-validation...")
        # CV always uses full dataset from DB (not test set)
        all_questions, _, all_intents = self._load_all_data()
        cv_results = self.evaluate_cross_validation(all_questions, all_intents)

        print("[4/5] Semantic similarity...")
        if has_test_set:
            print("      (skipped - test set has no raw questions)")
            sem_results = {
                "mean_similarity": 0.0, "median_similarity": 0.0,
                "min_similarity": 0.0, "max_similarity": 0.0,
                "std_similarity": 0.0, "pct_above_threshold": 0.0,
                "_raw_sims": [], "note": "Test set only - no raw questions available"
            }
        else:
            sem_results = self.evaluate_semantic_similarity(questions_raw)

        print("[5/5] Threshold sensitivity...")
        if has_test_set:
            print("      (skipped - test set has no raw questions)")
            threshold_results = []
        else:
            threshold_results = self.evaluate_threshold_sensitivity(
                questions_pre, questions_raw, true_intents)

        intent_coverage = self.evaluate_intent_coverage(true_intents)
        print("\n✓ Evaluation complete.\n")

        return {
            "timestamp":         self.timestamp,
            "n_samples":         n,
            "n_intents":         len(labels),
            "overall_metrics":   overall,
            "per_intent_metrics":per_intent,
            "confusion_matrix":  cm.tolist(),
            "intent_labels":     labels,
            "cross_validation":  cv_results,
            "semantic_similarity": sem_results,
            "threshold_sensitivity": threshold_results,
            "intent_coverage":   intent_coverage,
            "predicted_intents": pred_intents,
            "true_intents":      true_intents,
        }

    # ── Print report ──────────────────────────────────────────────────────────

    def print_thesis_report(self, results: dict):
        o   = results["overall_metrics"]
        cv  = results["cross_validation"]
        sem = results["semantic_similarity"]

        print("=" * 60)
        print("  THESIS EVALUATION REPORT")
        print(f"  Generated: {results['timestamp']}")
        print("=" * 60)
        print(f"\n  Dataset: {results['n_samples']} samples | {results['n_intents']} intents")

        print("\n-- OVERALL PERFORMANCE --------------------------------------")
        for label, key in [
            ("Accuracy",        "accuracy"),
            ("Macro Precision", "macro_precision"),
            ("Macro Recall",    "macro_recall"),
            ("Macro F1-Score",  "macro_f1"),
        ]:
            print(f"  {label:<22}: {o[key]*100:.2f}%")

        if cv.get("mean_accuracy") is not None:
            print(f"\n-- CROSS-VALIDATION ({cv['n_folds']}-fold) ----------------------------")
            for i, s in enumerate(cv["cv_scores"], 1):
                print(f"  Fold {i}: {s*100:.2f}%")
            print(f"  Mean   : {cv['mean_accuracy']*100:.2f}%")
            print(f"  Std Dev: +-{cv['std_accuracy']*100:.2f}%")
        else:
            print(f"\n-- CROSS-VALIDATION: {cv.get('note','Unavailable')}")

        print("\n-- SEMANTIC SIMILARITY --------------------------------------")
        print(f"  Mean     : {sem['mean_similarity']:.4f}")
        print(f"  Median   : {sem['median_similarity']:.4f}")
        print(f"  Std Dev  : {sem['std_similarity']:.4f}")
        print(f"  Min/Max  : {sem['min_similarity']:.4f} / {sem['max_similarity']:.4f}")
        print(f"  Above 0.35 threshold: {sem['pct_above_threshold']:.1f}%")

        print("\n-- PER-INTENT METRICS ---------------------------------------")
        print(f"  {'Intent':<30} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>5}")
        print("  " + "-" * 55)
        for intent, m in sorted(results["per_intent_metrics"].items()):
            print(f"  {intent:<30} {m['precision']*100:>5.1f}% "
                  f"{m['recall']*100:>5.1f}% {m['f1']*100:>5.1f}% {m['support']:>5}")
        print("\n" + "=" * 60)

    # ── Export CSV ────────────────────────────────────────────────────────────

    def export_csv(self, results: dict, out_dir: str = "evaluation_output"):
        # Create timestamped subfolder for this run
        ts = results["timestamp"]
        run_dir = os.path.join(out_dir, f"run_{ts}")
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, "overall_metrics.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["Metric", "Value"])
            for k, v in results["overall_metrics"].items():
                w.writerow([k, f"{v*100:.4f}%"])

        with open(os.path.join(run_dir, "per_intent_metrics.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["Intent", "Precision (%)", "Recall (%)", "F1 (%)", "Support"])
            for intent, m in sorted(results["per_intent_metrics"].items()):
                w.writerow([intent, f"{m['precision']*100:.2f}",
                            f"{m['recall']*100:.2f}", f"{m['f1']*100:.2f}", m["support"]])

        with open(os.path.join(run_dir, "cross_validation.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w  = csv.writer(f)
            cv = results["cross_validation"]
            if cv.get("mean_accuracy") is not None:
                w.writerow(["Fold", "Accuracy (%)"])
                for i, s in enumerate(cv["cv_scores"], 1):
                    w.writerow([i, f"{s*100:.4f}"])
                w.writerow(["Mean",    f"{cv['mean_accuracy']*100:.4f}"])
                w.writerow(["Std Dev", f"{cv['std_accuracy']*100:.4f}"])
            else:
                w.writerow(["Note", cv.get("note", "")])

        with open(os.path.join(run_dir, "threshold_sensitivity.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["Sim Threshold", "Conf Threshold",
                        "Accept Rate (%)", "Accuracy on Accepted (%)",
                        "N Accepted", "N Rejected"])
            for r in results["threshold_sensitivity"]:
                w.writerow([r["sim_threshold"], r["conf_threshold"],
                            r["accept_rate_pct"], r["accuracy_on_accepted"],
                            r["n_accepted"], r["n_rejected"]])

        with open(os.path.join(run_dir, "intent_coverage.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["Intent", "Sample Count", "Percentage (%)", "Quality Flag"])
            for intent, s in results["intent_coverage"].items():
                w.writerow([intent, s["sample_count"], s["percentage"], s["quality_flag"]])

        with open(os.path.join(run_dir, "semantic_similarity.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["Metric", "Value"])
            for k, v in results["semantic_similarity"].items():
                if k == "_raw_sims":
                    continue

                if isinstance(v, (int, float)):
                    w.writerow([k, f"{v:.6f}"])
                else:
                    w.writerow([k, str(v)])


        exportable = {k: v for k, v in results.items()
                      if k not in ("predicted_intents", "true_intents")}
        with open(os.path.join(run_dir, "full_results.json"), "w",
                  encoding="utf-8") as f:
            json.dump(exportable, f, indent=2, default=str)

        print(f"\n✓ CSVs saved to: {os.path.abspath(run_dir)}/")
        return run_dir

    # ── Export Graphs ─────────────────────────────────────────────────────────

    def export_graphs(self, results: dict, out_dir: str = "evaluation_output"):
        """Generate and save all 7 evaluation graphs as PNG files."""
        # Create timestamped subfolder for this run
        ts = results["timestamp"]
        run_dir = os.path.join(out_dir, f"run_{ts}")
        os.makedirs(run_dir, exist_ok=True)

        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor":   "white",
            "axes.edgecolor":   "#cccccc",
            "grid.color":       "#eeeeee",
            "font.family":      "DejaVu Sans",
            "axes.titlesize":   13,
            "axes.titleweight": "bold",
            "axes.labelsize":   11,
            "xtick.labelsize":  10,
            "ytick.labelsize":  10,
        })

        om  = results["overall_metrics"]
        cv  = results["cross_validation"]
        sem = results["semantic_similarity"]
        pim = results["per_intent_metrics"]
        paths = {}

        print("\n  Generating graphs...")

        # ── 1. Overall Metrics ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        metric_labels = ["Accuracy", "Macro\nPrecision", "Macro\nRecall", "Macro\nF1"]

        metric_keys   = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
        values = [om[k] * 100 for k in metric_keys]
        colors = ["#2196F3", "#42A5F5", "#64B5F6", "#1565C0", "#1976D2", "#1E88E5", "#0D47A1"]
        bars   = ax.bar(metric_labels, values, color=colors, edgecolor="white", zorder=3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Score (%)")
        ax.set_title("Overall Classification Performance Metrics", pad=15)
        ax.axhline(y=90, color="#FF5722", linestyle="--", linewidth=1,
                   alpha=0.6, label="90% reference line", zorder=2)
        ax.legend(fontsize=9)
        ax.grid(axis="y", zorder=0)
        plt.tight_layout()
        p = os.path.join(run_dir, "1_overall_metrics.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths["overall"] = p
        print(f"  [1/7] 1_overall_metrics.png")

        # ── 2. Cross-Validation ───────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        if cv.get("mean_accuracy") is not None:
            folds  = [f"Fold {i+1}" for i in range(len(cv["cv_scores"]))]
            scores = [s * 100 for s in cv["cv_scores"]]
            mean   = cv["mean_accuracy"] * 100
            std    = cv["std_accuracy"]  * 100
            fold_colors = ["#EF5350" if s < 75 else "#42A5F5" for s in scores]
            bars = ax.bar(folds, scores, color=fold_colors, edgecolor="white",
                          linewidth=0.8, zorder=3, width=0.5)
            for bar, val in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.axhline(y=mean, color="#1565C0", linestyle="-", linewidth=2.0, zorder=4)
            ax.axhspan(mean - std, mean + std, alpha=0.12, color="#1565C0")
            ax.legend(
                handles=[
                    mpatches.Patch(color="#42A5F5", label="Above 75% (acceptable)"),
                    mpatches.Patch(color="#EF5350", label="Below 75% (concern)"),
                    mpatches.Patch(color="#1565C0", alpha=0.5,
                                label=f"Mean +- Std ({mean:.2f}% +- {std:.2f}%)"),
                ],
                fontsize=9,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0
            )

            ax.set_ylim(0, 115)
            ax.set_title(f"{cv['n_folds']}-Fold Cross-Validation Accuracy\n"
                         f"Mean = {mean:.2f}%  Std = +-{std:.2f}%", pad=15)
        else:
            ax.text(0.5, 0.5, cv.get("note", "Not available"),
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title("Cross-Validation", pad=15)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", zorder=0)
        plt.tight_layout()
        p = os.path.join(run_dir, "2_cross_validation.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths["cv"] = p
        print(f"  [2/7] 2_cross_validation.png")

        # ── 3. Per-Intent Grouped Bar Chart ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 6))
        intents = sorted(pim.keys())
        prec = [pim[i]["precision"] * 100 for i in intents]
        rec  = [pim[i]["recall"]    * 100 for i in intents]
        f1   = [pim[i]["f1"]        * 100 for i in intents]
        x    = np.arange(len(intents))
        w    = 0.25
        b1 = ax.bar(x - w, prec, w, label="Precision", color="#1976D2", edgecolor="white", zorder=3)
        b2 = ax.bar(x,      rec,  w, label="Recall",    color="#26A69A", edgecolor="white", zorder=3)
        b3 = ax.bar(x + w,  f1,   w, label="F1-Score",  color="#7B1FA2", edgecolor="white", zorder=3)
        for bg in [b1, b2, b3]:
            for bar in bg:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([i.replace("_", "\n") for i in intents], fontsize=10)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Score (%)")
        ax.set_title("Per-Intent Classification Metrics (Precision / Recall / F1)", pad=15)
        ax.axhline(y=90, color="#FF5722", linestyle="--", linewidth=1, alpha=0.5, zorder=2)
        ax.legend(fontsize=10)
        ax.grid(axis="y", zorder=0)
        plt.tight_layout()
        p = os.path.join(run_dir, "3_per_intent_metrics.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths["per_intent"] = p
        print(f"  [3/7] 3_per_intent_metrics.png")

        # ── 4. Confusion Matrix Heatmap ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_data = np.array(results["confusion_matrix"])
        labels  = results["intent_labels"]
        short   = [l.replace("_", "\n") for l in labels]
        sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues",
                    xticklabels=short, yticklabels=short,
                    linewidths=0.5, linecolor="#dddddd",
                    cbar_kws={"label": "Number of Predictions"}, ax=ax)
        ax.set_xlabel("Predicted Intent", fontsize=11, labelpad=10)
        ax.set_ylabel("True Intent",      fontsize=11, labelpad=10)
        ax.set_title("Confusion Matrix — Intent Classification", pad=15)
        plt.xticks(rotation=30, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        p = os.path.join(run_dir, "4_confusion_matrix.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths["confusion"] = p
        print(f"  [4/7] 4_confusion_matrix.png")

        # ── 5. Semantic Similarity Histogram ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        raw_sims = sem.get("_raw_sims", [])
        if raw_sims:
            n_bins = min(20, len(set(round(s, 2) for s in raw_sims)))
            counts_h, bin_edges, patches = ax.hist(
                raw_sims, bins=n_bins, edgecolor="white", linewidth=0.8, zorder=3)
            for patch, left in zip(patches, bin_edges):
                norm_val = (left - min(raw_sims)) / max(max(raw_sims) - min(raw_sims), 0.001)
                patch.set_facecolor(plt.cm.YlGnBu(0.3 + 0.7 * norm_val))
            ax.set_xlabel("Cosine Similarity Score")
            ax.set_ylabel("Number of Queries")
        else:
            bar_labels = ["Min", "Mean", "Median", "Max"]
            bar_vals   = [sem["min_similarity"], sem["mean_similarity"],
                          sem["median_similarity"], sem["max_similarity"]]
            ax.bar(bar_labels, bar_vals, color=plt.cm.YlGnBu([0.3, 0.5, 0.6, 0.9]),
                   edgecolor="white", zorder=3)
            ax.set_ylabel("Cosine Similarity")
        ax.axvline(x=sem["mean_similarity"], color="#E53935", linestyle="--", linewidth=2,
                   label=f"Mean = {sem['mean_similarity']:.4f}", zorder=4)
        ax.axvline(x=0.35, color="#FF9800", linestyle=":", linewidth=1.5,
                   label="Acceptance threshold (0.35)", zorder=4)
        ax.set_title(
            f"Semantic Similarity Distribution (Sentence-BERT)\n"
            f"Mean={sem['mean_similarity']:.4f}  Median={sem['median_similarity']:.4f}  "
            f"{sem['pct_above_threshold']:.1f}% above 0.35 threshold", pad=15)
        ax.legend(fontsize=9)
        ax.grid(axis="y", zorder=0)
        plt.tight_layout()
        p = os.path.join(run_dir, "5_semantic_similarity.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths["semantic"] = p
        print(f"  [5/7] 5_semantic_similarity.png")

        # ── 6. Threshold Sensitivity ──────────────────────────────────────────
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        ts_data     = results["threshold_sensitivity"]
        conf_vals   = sorted(set(r["conf_threshold"] for r in ts_data))
        colors_line = plt.cm.Blues(np.linspace(0.4, 0.9, len(conf_vals)))
        for conf_t, col in zip(conf_vals, colors_line):
            subset = sorted([r for r in ts_data if r["conf_threshold"] == conf_t],
                            key=lambda r: r["sim_threshold"])
            ax1.plot([r["sim_threshold"] for r in subset],
                     [r["accept_rate_pct"] for r in subset],
                     "o-", color=col, linewidth=1.5, markersize=5, alpha=0.8,
                     label=f"conf={conf_t}%")
        default_sub = sorted([r for r in ts_data if r["conf_threshold"] == 55],
                              key=lambda r: r["sim_threshold"])
        ax2.plot([r["sim_threshold"] for r in default_sub],
                 [r["accuracy_on_accepted"] for r in default_sub],
                 "s--", color="#7B1FA2", linewidth=2.5, markersize=7,
                 label="Accuracy @ conf=55% (default)")
        current = next((r for r in ts_data
                        if r["sim_threshold"] == 0.35 and r["conf_threshold"] == 55), None)
        if current:
            ax1.scatter([0.35], [current["accept_rate_pct"]],
                        color="#FF5722", s=120, zorder=5, label="Current default")
            ax2.scatter([0.35], [current["accuracy_on_accepted"]],
                        color="#FF5722", s=120, zorder=5)
        ax1.set_xlabel("Semantic Similarity Threshold")
        ax1.set_ylabel("Accept Rate (%)",          color="#1565C0")
        ax2.set_ylabel("Accuracy on Accepted (%)", color="#7B1FA2")
        ax1.tick_params(axis="y", labelcolor="#1565C0")
        ax2.tick_params(axis="y", labelcolor="#7B1FA2")
        ax1.set_ylim(0, 115)
        ax2.set_ylim(85, 105)
        ax1.grid(axis="y", zorder=0)
        ax1.set_title("Threshold Sensitivity: Accept Rate vs Accuracy Trade-off", pad=15)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left", ncol=2)
        plt.tight_layout()
        p = os.path.join(run_dir, "6_threshold_sensitivity.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths["threshold"] = p
        print(f"  [6/7] 6_threshold_sensitivity.png")

        # ── 7. Combined Summary ───────────────────────────────────────────────
        fig = plt.figure(figsize=(18, 11))
        fig.suptitle("Chatbot Evaluation Summary — Lipa City Colleges Campus Services",
                     fontsize=15, fontweight="bold", y=1.01)
        subplot_order = [
            ("1_overall_metrics.png",       "1. Overall Metrics"),
            ("2_cross_validation.png",       "2. Cross-Validation"),
            ("3_per_intent_metrics.png",     "3. Per-Intent Metrics"),
            ("4_confusion_matrix.png",       "4. Confusion Matrix"),
            ("5_semantic_similarity.png",    "5. Semantic Similarity"),
            ("6_threshold_sensitivity.png",  "6. Threshold Sensitivity"),
        ]
        for idx, (fname, title) in enumerate(subplot_order):
            ax  = fig.add_subplot(2, 3, idx + 1)
            img_path = os.path.join(run_dir, fname)
            if os.path.exists(img_path):
                ax.imshow(mpimg.imread(img_path))
            ax.axis("off")
            ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        plt.tight_layout()
        p = os.path.join(run_dir, "0_summary_all_graphs.png")
        plt.savefig(p, dpi=130, bbox_inches="tight")
        plt.close()
        paths["summary"] = p
        print(f"  [7/7] 0_summary_all_graphs.png")

        print(f"\n✓ Graphs saved to: {os.path.abspath(run_dir)}/")
        return paths
        return paths


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluator = ChatbotEvaluator(k_folds=5)
    results   = evaluator.run_full_evaluation()
    evaluator.print_thesis_report(results)
    evaluator.export_csv(results)
    evaluator.export_graphs(results)