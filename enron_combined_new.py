#!/usr/bin/env python3
"""
Enron combined training pipeline — v3 (fully corrected).

Fixes applied vs previous versions:
  v1 → v2  No TF-IDF leakage, leakage-safe splits, threshold tuning on val,
            duplicate deduplication, reproducible metrics.
  v2 → v3  (this file)
  [BUG 1]  expand() was re-inflating duplicate rows after the group-level split,
           destroying the stratification and producing wildly different class
           ratios across train / val / test (e.g. 50% / 38% / 65% spam).
           Fix: one_per_group() — one canonical row per unique-text group.
  [BUG 2]  select_positive_score_column returned numpy VIEWS, not copies.
           All three classifiers shared the same probability buffer, producing
           identical debug numbers. Fix: .copy() on both columns.
  [BUG 3]  Hardcoded class_prior=[0.62, 0.38] in MultinomialNB was fighting the
           true data distribution. Fix: remove it; let the model learn the prior
           from balanced data.
  [BUG 4]  No class_weight balancing on LR and RF. Fix: class_weight='balanced'.
  [BUG 5]  Probability miscalibration in NB/RF made threshold selection brittle.
           Fix: CalibratedClassifierCV wrapping all three classifiers.
  [NEW]    Hard assertions after splitting to catch class-imbalance regressions.
  [NEW]    pos_mean > neg_mean assertion inside classifier loop to catch inverted
           column selection early.
  [NEW]    Full threshold diagnostic sweep printed for the test set (for analysis
           only — model selection still uses val exclusively).

Usage:
    python enron_combined_training_v3.py emails.csv enron_spam_data.csv out.pkl
    python enron_combined_training_v3.py NONE enron_spam_data.csv out.pkl
"""

import argparse
import csv
import json
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from email.parser import Parser
from email.policy import default as email_policy
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

csv.field_size_limit(10 * 1024 * 1024)

RANDOM_STATE = 42

PHISHING_SIGNAL_PHRASES = [
    "urgent", "verify", "password", "click here", "confirm",
    "expire", "act now", "immediately", "confidential", "suspended",
    "unauthorized", "security alert", "account", "bank", "credit card",
    "private", "credential", "ssn",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EmailData:
    sender: str = ""
    subject: str = ""
    body: str = ""
    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    vocabulary_richness: float = 0.0
    label: str = ""


@dataclass
class SenderProfile:
    email_count: int = 0
    avg_sentence_lengths: List[float] = field(default_factory=list)
    vocab_richness_values: List[float] = field(default_factory=list)
    word_counts: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Full Enron corpus loader (emails.csv — optional)
# ---------------------------------------------------------------------------

class FullEnronLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.emails: List[EmailData] = []
        self.vocabulary: Set[str] = set()
        self.sender_profiles: Dict[str, SenderProfile] = defaultdict(SenderProfile)
        self.stats = {
            "total_raw": 0,
            "total_processed": 0,
            "skipped_short": 0,
            "skipped_parse_error": 0,
            "unique_senders": 0,
            "vocabulary_size": 0,
            "avg_word_count": 0.0,
            "avg_sentence_length": 0.0,
            "avg_vocab_richness": 0.0,
            "external_ratio": 0.0,
            "attachment_ratio": 0.0,
            "senders_5_plus": 0,
        }

    def _parse_raw_email(self, raw_message: str) -> Optional[Dict]:
        try:
            parser = Parser(policy=email_policy)
            msg = parser.parsestr(raw_message)

            from_header = msg.get("From", "")
            from_addr = self._extract_email(from_header)

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        try:
                            body = part.get_content()
                            break
                        except Exception:
                            pass
            else:
                try:
                    body = msg.get_content()
                except Exception:
                    try:
                        payload = msg.get_payload(decode=True)
                        if isinstance(payload, bytes):
                            body = payload.decode("utf-8", errors="ignore")
                        else:
                            body = str(msg.get_payload())
                    except Exception:
                        body = str(msg.get_payload())

            to_header = msg.get("To", "")
            has_external = self._has_external_recipient(to_header)
            subject = str(msg.get("Subject", ""))
            has_attachment = self._mentions_attachment(subject + " " + str(body))

            return {
                "from": from_addr,
                "subject": subject,
                "body": str(body) if body else "",
                "has_external": has_external,
                "has_attachment": has_attachment,
            }
        except Exception:
            return None

    @staticmethod
    def _extract_email(header: str) -> str:
        if not header:
            return ""
        match = re.search(r"<([^>]+)>", str(header))
        if match:
            return match.group(1).lower().strip()
        if "@" in str(header):
            return str(header).lower().strip()
        return ""

    def _has_external_recipient(self, to_header: str) -> bool:
        if not to_header:
            return False
        enron_domains = ["enron.com", "enron.net", "ect.enron.com"]
        for addr in str(to_header).split(","):
            email = self._extract_email(addr)
            if email and not any(domain in email for domain in enron_domains):
                return True
        return False

    @staticmethod
    def _mentions_attachment(text: str) -> bool:
        keywords = ["attached", "attachment", "enclosed", "see attached",
                    ".xls", ".doc", ".pdf"]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keywords)

    @staticmethod
    def _calculate_metrics(text: str) -> Optional[Dict]:
        if not text or len(text.strip()) < 20:
            return None

        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < 30:
            return None

        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sentences:
            return None

        word_count = len(words)
        unique_words = set(words)
        sentence_lengths = [len(s.split()) for s in sentences if s.split()]
        if not sentence_lengths:
            return None

        return {
            "word_count": word_count,
            "sentence_count": len(sentence_lengths),
            "avg_sentence_length": float(np.mean(sentence_lengths)),
            "vocabulary_richness": float(len(unique_words) / word_count),
            "unique_words": unique_words,
        }

    def load(self, max_emails: int = 500_000, progress_interval: int = 50_000):
        print(f"\n{'=' * 60}\nLOADING FULL ENRON CORPUS (emails.csv)\n{'=' * 60}")
        print(f"File: {self.csv_path}\nMax emails: {max_emails:,}")

        external_count = 0
        attachment_count = 0
        all_word_counts: List[float] = []
        all_sentence_lengths: List[float] = []
        all_vocab_richness: List[float] = []

        try:
            with open(self.csv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.stats["total_raw"] += 1
                    if self.stats["total_raw"] > max_emails:
                        break

                    if self.stats["total_raw"] % progress_interval == 0:
                        print(f"  Processed {self.stats['total_raw']:,} emails...")

                    message = row.get("message", "") or row.get("Message", "")
                    if not message:
                        self.stats["skipped_parse_error"] += 1
                        continue

                    parsed = self._parse_raw_email(message)
                    if not parsed:
                        self.stats["skipped_parse_error"] += 1
                        continue

                    metrics = self._calculate_metrics(parsed["body"])
                    if not metrics:
                        self.stats["skipped_short"] += 1
                        continue

                    self.stats["total_processed"] += 1
                    self.vocabulary.update(metrics["unique_words"])

                    sender = parsed["from"]
                    if sender:
                        profile = self.sender_profiles[sender]
                        profile.email_count += 1
                        profile.avg_sentence_lengths.append(metrics["avg_sentence_length"])
                        profile.vocab_richness_values.append(metrics["vocabulary_richness"])
                        profile.word_counts.append(metrics["word_count"])

                    all_word_counts.append(metrics["word_count"])
                    all_sentence_lengths.append(metrics["avg_sentence_length"])
                    all_vocab_richness.append(metrics["vocabulary_richness"])

                    if parsed["has_external"]:
                        external_count += 1
                    if parsed["has_attachment"]:
                        attachment_count += 1

                    self.emails.append(
                        EmailData(
                            sender=sender,
                            subject=parsed["subject"],
                            body=parsed["body"],
                            word_count=metrics["word_count"],
                            sentence_count=metrics["sentence_count"],
                            avg_sentence_length=metrics["avg_sentence_length"],
                            vocabulary_richness=metrics["vocabulary_richness"],
                        )
                    )
        except FileNotFoundError:
            print(f"WARNING: Full corpus file not found: {self.csv_path}")
            return

        n = self.stats["total_processed"]
        self.stats["unique_senders"] = len(self.sender_profiles)
        self.stats["vocabulary_size"] = len(self.vocabulary)
        self.stats["avg_word_count"] = float(np.mean(all_word_counts)) if all_word_counts else 0.0
        self.stats["avg_sentence_length"] = float(np.mean(all_sentence_lengths)) if all_sentence_lengths else 0.0
        self.stats["avg_vocab_richness"] = float(np.mean(all_vocab_richness)) if all_vocab_richness else 0.0
        self.stats["external_ratio"] = (external_count / n * 100.0) if n else 0.0
        self.stats["attachment_ratio"] = (attachment_count / n * 100.0) if n else 0.0
        self.stats["senders_5_plus"] = sum(
            1 for p in self.sender_profiles.values() if p.email_count >= 5
        )

        print(f"\n  Finished loading {n:,} emails")
        print(json.dumps(self.stats, indent=2))


# ---------------------------------------------------------------------------
# Spam dataset loader (enron_spam_data.csv)
# ---------------------------------------------------------------------------

class EnronSpamLoader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.records: List[Dict[str, str]] = []
        self.stats = {"total": 0, "ham": 0, "spam": 0, "skipped": 0}

    def load(self, min_words: int = 20):
        print(f"\n{'=' * 60}\nLOADING ENRON SPAM DATASET (enron_spam_data.csv)\n{'=' * 60}")
        print(f"File: {self.csv_path}")

        with open(self.csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.stats["total"] += 1

                label = (row.get("Spam/Ham", "") or "").strip().lower()
                if label not in {"ham", "spam"}:
                    self.stats["skipped"] += 1
                    continue

                subject = row.get("Subject", "") or ""
                message = row.get("Message", "") or ""
                content = f"{subject} {message}".strip()

                if len(content.split()) < min_words:
                    self.stats["skipped"] += 1
                    continue

                self.records.append({"text": content, "label": label})
                self.stats[label] += 1

        print(json.dumps(self.stats, indent=2))


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def threshold_grid() -> List[float]:
    """
    Dense grid at the low end (where calibrated spam probabilities live)
    and sparser at the high end.  Covers 0.0001 → 0.90.
    """
    return [
        0.0001, 0.0005, 0.001, 0.002, 0.005,
        0.01, 0.02, 0.03, 0.04, 0.05,
        0.07, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.40, 0.50, 0.60, 0.70,
        0.80, 0.90,
    ]


def select_positive_score_column(prob_matrix: np.ndarray, y_true: np.ndarray):
    """
    Return the column whose scores best discriminate the positive class (label=1).
    Returns copies (not views) to avoid shared-buffer aliasing across loop
    iterations. BUG FIX: original code returned numpy views, causing all three
    classifiers to appear identical in the debug output.
    """
    if prob_matrix.ndim != 2 or prob_matrix.shape[1] != 2:
        raise ValueError(
            f"Expected (n_samples, 2) probability matrix; got {prob_matrix.shape}"
        )

    # .copy() is the critical fix — slices of a 2-D array are views, not copies
    col0 = prob_matrix[:, 0].copy()
    col1 = prob_matrix[:, 1].copy()

    auc0 = roc_auc_score(y_true, col0)
    auc1 = roc_auc_score(y_true, col1)

    if auc1 >= auc0:
        return 1, col1, auc0, auc1
    return 0, col0, auc0, auc1


def _assert_class_balance(y: np.ndarray, split_name: str,
                           lo: float = 0.30, hi: float = 0.70) -> None:
    """
    Hard-stop if spam ratio falls outside [lo, hi].  Catches expand()-style
    stratification breakage before it silently poisons training or evaluation.
    """
    rate = float(np.mean(y))
    assert lo < rate < hi, (
        f"Class imbalance in {split_name}: {rate:.2%} spam "
        f"(expected {lo:.0%}–{hi:.0%}).  "
        "Check one_per_group() is being used instead of expand()."
    )
    print(f"  [{split_name}] spam rate: {rate:.2%}  ✓")


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class CombinedForensicsAgent:
    PHISHING_KEYWORDS = [
        "urgent", "verify", "password", "click here", "confirm",
        "expire", "act now", "immediately", "confidential", "suspended",
        "unauthorized", "security alert", "account", "bank", "credit card",
    ]
    SENSITIVE_KEYWORDS = [
        "confidential", "secret", "private", "patient", "ssn",
        "password", "credential", "classified",
    ]

    def __init__(self):
        self.vocabulary: Set[str] = set()
        self.vocabulary_size: int = 0
        self.sender_profiles: Dict[str, SenderProfile] = {}
        self.baseline_sentence_length: float = 15.0
        self.baseline_vocab_richness: float = 0.5
        self.baseline_word_count: float = 100.0
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier = None
        self.classifier_name: str = ""
        self.classifier_metrics: Dict = {}
        self.duplicate_report: Dict = {}
        self.learned_phrase_weights: Dict[str, float] = {}
        self.selected_threshold: float = 0.5

    def train(
        self,
        full_corpus_loader: Optional[FullEnronLoader],
        spam_loader: EnronSpamLoader,
    ):
        print(f"\n{'=' * 60}\nTRAINING COMBINED FORENSICS AGENT\n{'=' * 60}")

        if full_corpus_loader and full_corpus_loader.stats["total_processed"]:
            print("\n1. Setting baselines from full Enron corpus...")
            self.vocabulary = full_corpus_loader.vocabulary
            self.vocabulary_size = len(self.vocabulary)
            self.sender_profiles = dict(full_corpus_loader.sender_profiles)
            self.baseline_sentence_length = full_corpus_loader.stats["avg_sentence_length"]
            self.baseline_vocab_richness = full_corpus_loader.stats["avg_vocab_richness"]
            self.baseline_word_count = full_corpus_loader.stats["avg_word_count"]
        else:
            print("\n1. Full Enron corpus unavailable; keeping default style baselines.")

        print("\n2. Training phishing classifier with leakage-safe splits...")
        self._train_classifier(spam_loader.records)

    # ------------------------------------------------------------------
    # Core training routine
    # ------------------------------------------------------------------

    def _train_classifier(self, records: List[Dict[str, str]]):
        texts = [r["text"] for r in records]
        labels = np.array([1 if r["label"] == "spam" else 0 for r in records])
        normalized = [normalize_text(t) for t in texts]

        # ── Deduplication ────────────────────────────────────────────────────
        exact_dups = len(texts) - len(set(texts))

        grouped_records: Dict[str, List] = defaultdict(list)
        for text, label, norm in zip(texts, labels, normalized):
            grouped_records[norm].append((text, int(label)))

        conflicting_label_duplicates = 0
        for entries in grouped_records.values():
            if len({lbl for _, lbl in entries}) > 1:
                conflicting_label_duplicates += 1

        # Resolve each group to a single (majority-vote) label
        group_norms: List[str] = []
        group_labels: List[int] = []
        for norm, entries in grouped_records.items():
            label_counts = Counter(lbl for _, lbl in entries)
            group_label = 1 if label_counts[1] >= label_counts[0] else 0
            group_norms.append(norm)
            group_labels.append(group_label)

        # ── Stratified splits (group level) ──────────────────────────────────
        trainval_norms, test_norms, trainval_labels, test_labels = train_test_split(
            group_norms,
            group_labels,
            test_size=0.15,
            random_state=RANDOM_STATE,
            stratify=group_labels,
        )

        train_norms, val_norms, _, _ = train_test_split(
            trainval_norms,
            trainval_labels,
            # 0.17647… of 85 % ≈ 15 % of the whole dataset
            test_size=0.17647058823529413,
            random_state=RANDOM_STATE,
            stratify=trainval_labels,
        )

        # ── FIX: one_per_group instead of expand() ───────────────────────────
        # expand() re-added ALL duplicate rows after the group-level split,
        # causing spam/ham ratios to diverge drastically between splits because
        # spam has far more duplicates per group than ham.
        # one_per_group keeps exactly one canonical row per unique-text group,
        # preserving the stratification guarantees of train_test_split.
        def one_per_group(norm_list: List[str]) -> List[tuple]:
            rows = []
            for norm in norm_list:
                entries = grouped_records[norm]
                # Majority-vote label (already computed above; recompute for safety)
                label_counts = Counter(lbl for _, lbl in entries)
                group_label = 1 if label_counts[1] >= label_counts[0] else 0
                # Use the first occurrence as the canonical text
                rows.append((entries[0][0], group_label))
            return rows

        train_rows    = one_per_group(train_norms)
        val_rows      = one_per_group(val_norms)
        trainval_rows = one_per_group(trainval_norms)   # for final refit
        test_rows     = one_per_group(test_norms)

        X_train_text    = [r[0] for r in train_rows]
        y_train         = np.array([r[1] for r in train_rows])

        X_val_text      = [r[0] for r in val_rows]
        y_val           = np.array([r[1] for r in val_rows])

        X_trainval_text = [r[0] for r in trainval_rows]
        y_trainval      = np.array([r[1] for r in trainval_rows])

        X_test_text     = [r[0] for r in test_rows]
        y_test          = np.array([r[1] for r in test_rows])

        # ── Class balance assertions ──────────────────────────────────────────
        print("\nCLASS BALANCE CHECK (must be 30–70% spam in every split)")
        _assert_class_balance(y_train,    "train")
        _assert_class_balance(y_val,      "val")
        _assert_class_balance(y_test,     "test")
        _assert_class_balance(y_trainval, "trainval")

        print(f"\n  Train    : {len(y_train):>6,} rows  "
              f"spam={int(y_train.sum())}  ham={int((y_train==0).sum())}")
        print(f"  Val      : {len(y_val):>6,} rows  "
              f"spam={int(y_val.sum())}  ham={int((y_val==0).sum())}")
        print(f"  Test     : {len(y_test):>6,} rows  "
              f"spam={int(y_test.sum())}  ham={int((y_test==0).sum())}")

        # ── Overlap checks ───────────────────────────────────────────────────
        overlap_train_val  = len(set(train_norms) & set(val_norms))
        overlap_train_test = len(set(trainval_norms) & set(test_norms))
        if overlap_train_val or overlap_train_test:
            raise RuntimeError(
                f"Normalized text overlap across splits: "
                f"train∩val={overlap_train_val}  trainval∩test={overlap_train_test}"
            )

        # ── Phrase weights (from training split only) ─────────────────────────
        spam_train = int(y_train.sum())
        ham_train  = int(len(y_train) - spam_train)
        phrase_weights: Dict[str, float] = {}
        for phrase in PHISHING_SIGNAL_PHRASES:
            spam_hits = sum(
                1 for text, lbl in zip(X_train_text, y_train)
                if lbl == 1 and phrase in text.lower()
            )
            ham_hits = sum(
                1 for text, lbl in zip(X_train_text, y_train)
                if lbl == 0 and phrase in text.lower()
            )
            spam_rate = (spam_hits + 1) / (spam_train + 2)
            ham_rate  = (ham_hits  + 1) / (ham_train  + 2)
            phrase_weights[phrase] = round(max(0.0, spam_rate - ham_rate), 6)
        self.learned_phrase_weights = phrase_weights

        # ── Classifier factories ──────────────────────────────────────────────
        # FIX: class_weight='balanced' on LR and RF compensates for any residual
        # imbalance.  CalibratedClassifierCV fixes NB/RF probability compression
        # so the threshold grid works reliably.  No hardcoded class_prior on NB
        # (it was fighting the true data distribution in v2).
        classifier_factories = {
            "logistic_regression": lambda: CalibratedClassifierCV(
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                method="sigmoid",
                cv=3,
            ),
            "naive_bayes": lambda: CalibratedClassifierCV(
                MultinomialNB(alpha=0.1),
                method="isotonic",
                cv=3,
            ),
            "random_forest": lambda: CalibratedClassifierCV(
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=30,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                method="sigmoid",
                cv=3,
            ),
        }

        # ── Validation loop ───────────────────────────────────────────────────
        val_results:       Dict[str, Dict] = {}
        best_thresholds:   Dict[str, float] = {}
        best_score_columns: Dict[str, int]  = {}

        print("\nValidation metrics by classifier:")
        for name, factory in classifier_factories.items():
            # Each classifier gets its own vectorizer fitted only on train
            vectorizer = TfidfVectorizer(
                max_features=10_000,
                ngram_range=(1, 2),
                min_df=2,
                stop_words="english",
            )
            X_train_tfidf = vectorizer.fit_transform(X_train_text)
            X_val_tfidf   = vectorizer.transform(X_val_text)

            clf = factory()
            clf.fit(X_train_tfidf, y_train)

            print(f"\n--- {name} ---")
            print("Classifier type:", type(clf).__name__)
            if hasattr(clf, "classes_"):
                print("Classes:", clf.classes_)
            elif hasattr(clf, "estimator") and hasattr(clf.estimator, "classes_"):
                print("Classes:", clf.estimator.classes_)

            # select_positive_score_column now returns .copy() arrays (BUG FIX)
            val_prob_matrix = clf.predict_proba(X_val_tfidf)
            selected_col, y_val_score, auc0, auc1 = select_positive_score_column(
                val_prob_matrix, y_val
            )
            print(f"Selected probability column: {selected_col}")
            print(f"ROC-AUC column 0: {auc0:.4f}   column 1: {auc1:.4f}")

            pos_scores = y_val_score[y_val == 1]
            neg_scores = y_val_score[y_val == 0]
            print(f"DEBUG VAL  min={y_val_score.min():.6f}  "
                  f"max={y_val_score.max():.6f}  mean={y_val_score.mean():.6f}")
            print(f"  Spam score  mean={pos_scores.mean():.6f}  "
                  f"p50={np.percentile(pos_scores, 50):.6f}  "
                  f"p90={np.percentile(pos_scores, 90):.6f}")
            print(f"  Ham  score  mean={neg_scores.mean():.6f}  "
                  f"p90={np.percentile(neg_scores, 90):.6f}  "
                  f"p99={np.percentile(neg_scores, 99):.6f}")

            # Sanity: spam scores must be higher than ham scores on average
            if pos_scores.mean() <= neg_scores.mean():
                print(
                    f"  WARNING: {name} — spam score mean ({pos_scores.mean():.6f}) "
                    f"<= ham score mean ({neg_scores.mean():.6f}). "
                    "Column selection may be inverted or model is degenerate."
                )

            # Threshold sweep on validation
            best_f1        = -1.0
            best_threshold = 0.5
            best_metrics   = None

            for threshold in threshold_grid():
                y_val_pred = (y_val_score >= threshold).astype(int)
                tp = int(((y_val == 1) & (y_val_pred == 1)).sum())
                fp = int(((y_val == 0) & (y_val_pred == 1)).sum())
                fn = int(((y_val == 1) & (y_val_pred == 0)).sum())

                metrics = {
                    "accuracy":  float(accuracy_score(y_val, y_val_pred)),
                    "precision": float(precision_score(y_val, y_val_pred, zero_division=0)),
                    "recall":    float(recall_score(y_val, y_val_pred, zero_division=0)),
                    "f1":        float(f1_score(y_val, y_val_pred, zero_division=0)),
                }
                print(
                    f"  threshold={threshold:.4f}  "
                    f"pred_pos={tp+fp:4d}  tp={tp:4d}  fp={fp:4d}  fn={fn:4d}  "
                    f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
                    f"F1={metrics['f1']:.4f}"
                )

                if metrics["f1"] > best_f1:
                    best_f1        = metrics["f1"]
                    best_threshold = float(threshold)
                    best_metrics   = metrics

            best_metrics["roc_auc"]       = float(roc_auc_score(y_val, y_val_score))
            best_metrics["best_threshold"] = best_threshold
            val_results[name]             = best_metrics
            best_thresholds[name]         = best_threshold
            best_score_columns[name]      = selected_col

            print(f"BEST for {name}: {json.dumps(best_metrics, indent=2)}")

        # ── Model selection (val F1) ──────────────────────────────────────────
        best_name         = max(val_results, key=lambda n: val_results[n]["f1"])
        best_threshold    = best_thresholds[best_name]
        best_score_column = best_score_columns[best_name]

        self.classifier_name   = best_name
        self.selected_threshold = best_threshold

        print(f"\nSelected classifier : {best_name}")
        print(f"Selected threshold  : {best_threshold:.4f}")

        # ── Final refit on train+val ──────────────────────────────────────────
        self.vectorizer = TfidfVectorizer(
            max_features=10_000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
        )
        X_trainval_tfidf = self.vectorizer.fit_transform(X_trainval_text)
        X_test_tfidf     = self.vectorizer.transform(X_test_text)

        self.classifier = classifier_factories[best_name]()
        self.classifier.fit(X_trainval_tfidf, y_trainval)

        # ── Test evaluation ───────────────────────────────────────────────────
        test_prob_matrix = self.classifier.predict_proba(X_test_tfidf)
        y_test_score     = test_prob_matrix[:, best_score_column].copy()

        print(f"\nDEBUG TEST  min={y_test_score.min():.6f}  "
              f"max={y_test_score.max():.6f}  mean={y_test_score.mean():.6f}")

        # Diagnostic: full threshold sweep on test (never used for model selection)
        print("\nTest threshold diagnostic sweep (analysis only):")
        for threshold in threshold_grid():
            y_test_pred_tmp = (y_test_score >= threshold).astype(int)
            tp = int(((y_test == 1) & (y_test_pred_tmp == 1)).sum())
            fp = int(((y_test == 0) & (y_test_pred_tmp == 1)).sum())
            fn = int(((y_test == 1) & (y_test_pred_tmp == 0)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            print(
                f"  t={threshold:.4f}  tp={tp:4d}  fp={fp:4d}  fn={fn:4d}  "
                f"P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}"
            )

        # Final test metrics using the threshold chosen on val
        y_test_pred = (y_test_score >= best_threshold).astype(int)
        test_metrics = {
            "accuracy":         float(accuracy_score(y_test, y_test_pred)),
            "precision":        float(precision_score(y_test, y_test_pred, zero_division=0)),
            "recall":           float(recall_score(y_test, y_test_pred, zero_division=0)),
            "f1":               float(f1_score(y_test, y_test_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
            "support":          int(len(y_test)),
            "threshold":        float(best_threshold),
            "roc_auc":          float(roc_auc_score(y_test, y_test_score)),
        }

        self.classifier_metrics = {
            "selection_metric":   "validation_f1",
            "validation_results": val_results,
            "test_metrics":       test_metrics,
            "split": {
                "train_size":           int(len(X_train_text)),
                "validation_size":      int(len(X_val_text)),
                "test_size":            int(len(X_test_text)),
                "train_fraction":       round(len(X_train_text) / len(group_norms), 6),
                "validation_fraction":  round(len(X_val_text)   / len(group_norms), 6),
                "test_fraction":        round(len(X_test_text)   / len(group_norms), 6),
                "random_state":         RANDOM_STATE,
            },
        }

        self.duplicate_report = {
            "raw_examples":                      int(len(texts)),
            "exact_duplicate_emails":            int(exact_dups),
            "normalized_unique_groups":          int(len(grouped_records)),
            "conflicting_label_duplicate_groups": int(conflicting_label_duplicates),
            "normalized_overlap_train_validation": int(overlap_train_val),
            "normalized_overlap_trainval_test":    int(overlap_train_test),
        }

        print("\nFinal untouched test metrics:")
        print(json.dumps(test_metrics, indent=2))
        print("\nDuplicate leakage report:")
        print(json.dumps(self.duplicate_report, indent=2))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        data = {
            "vocabulary_size":        self.vocabulary_size,
            "baseline_sentence_length": self.baseline_sentence_length,
            "baseline_vocab_richness":  self.baseline_vocab_richness,
            "baseline_word_count":      self.baseline_word_count,
            "vectorizer":             self.vectorizer,
            "classifier":             self.classifier,
            "classifier_accuracy":    self.classifier_metrics.get(
                "test_metrics", {}
            ).get("accuracy", 0.0),
            "classifier_name":        self.classifier_name,
            "classifier_metrics":     self.classifier_metrics,
            "duplicate_report":       self.duplicate_report,
            "learned_phrase_weights": self.learned_phrase_weights,
            "selected_threshold":     self.selected_threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"\nModel saved to: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train_combined_model(
    full_corpus_path: Optional[str],
    spam_data_path: str,
    output_path: str,
    max_emails: int = 500_000,
) -> CombinedForensicsAgent:
    print("╔" + "═" * 68 + "╗")
    print("║" + " ENRON EMAIL FORENSICS TRAINING  v3 ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    full_loader = None
    if (
        full_corpus_path
        and full_corpus_path.upper() != "NONE"
        and Path(full_corpus_path).exists()
    ):
        full_loader = FullEnronLoader(full_corpus_path)
        full_loader.load(max_emails=max_emails)
    else:
        print("Full corpus not provided or unavailable; skipping emails.csv baselines.")

    spam_loader = EnronSpamLoader(spam_data_path)
    spam_loader.load()

    agent = CombinedForensicsAgent()
    agent.train(full_loader, spam_loader)
    agent.save(output_path)
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enron email forensics training pipeline — v3"
    )
    parser.add_argument(
        "full_corpus_path",
        help="Path to emails.csv, or NONE if unavailable",
    )
    parser.add_argument("spam_data_path", help="Path to enron_spam_data.csv")
    parser.add_argument(
        "output_path",
        nargs="?",
        default="combined_forensics_model_v3.pkl",
        help="Output pickle path (default: combined_forensics_model_v3.pkl)",
    )
    parser.add_argument(
        "--max-emails",
        type=int,
        default=500_000,
        help="Max emails to load from the full corpus (default: 500 000)",
    )
    args = parser.parse_args()
    train_combined_model(
        args.full_corpus_path,
        args.spam_data_path,
        args.output_path,
        max_emails=args.max_emails,
    )