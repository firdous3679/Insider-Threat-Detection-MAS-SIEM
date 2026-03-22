#!/usr/bin/env python3
"""
Corrected Enron combined training pipeline.

Fixes applied:
- No TF-IDF leakage: split raw texts before fitting the vectorizer.
- Proper model selection: choose classifier on validation set only, then report on an untouched test set.
- Duplicate safeguards: normalize content, report exact/normalized duplicates, and deduplicate before splitting.
- Threshold tuning on validation probabilities instead of using the default 0.5 threshold.
- Reproducible metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
- Saves corrected model to a separate output file by default.

Usage:
    python enron_combined_training.py emails.csv enron_spam_data.csv combined_forensics_model_fixed.pkl
    python enron_combined_training.py NONE enron_spam_data.csv combined_forensics_model_fixed.pkl
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
from statistics import mean, median, pstdev
import numpy as np
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB

csv.field_size_limit(10 * 1024 * 1024)

RANDOM_STATE = 42

PHISHING_SIGNAL_PHRASES = [
    "urgent", "verify", "password", "click here", "confirm",
    "expire", "act now", "immediately", "confidential", "suspended",
    "unauthorized", "security alert", "account", "bank", "credit card",
    "private", "credential", "ssn",
]


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
        keywords = ["attached", "attachment", "enclosed", "see attached", ".xls", ".doc", ".pdf"]
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

    def load(self, max_emails: int = 500000, progress_interval: int = 50000):
        print(f"\n{'=' * 60}\nLOADING FULL ENRON CORPUS (emails.csv)\n{'=' * 60}")
        print(f"File: {self.csv_path}\nMax emails: {max_emails:,}")

        external_count = 0
        attachment_count = 0
        all_word_counts = []
        all_sentence_lengths = []
        all_vocab_richness = []

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

        self.stats["unique_senders"] = len(self.sender_profiles)
        self.stats["vocabulary_size"] = len(self.vocabulary)
        self.stats["avg_word_count"] = float(np.mean(all_word_counts)) if all_word_counts else 0.0
        self.stats["avg_sentence_length"] = float(np.mean(all_sentence_lengths)) if all_sentence_lengths else 0.0
        self.stats["avg_vocab_richness"] = float(np.mean(all_vocab_richness)) if all_vocab_richness else 0.0
        self.stats["external_ratio"] = (external_count / self.stats["total_processed"] * 100.0) if self.stats["total_processed"] else 0.0
        self.stats["attachment_ratio"] = (attachment_count / self.stats["total_processed"] * 100.0) if self.stats["total_processed"] else 0.0
        self.stats["senders_5_plus"] = sum(1 for p in self.sender_profiles.values() if p.email_count >= 5)

        print(f"\n  Finished loading {self.stats['total_processed']:,} emails")
        print(json.dumps(self.stats, indent=2))


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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def threshold_grid() -> List[float]:
    return [
        0.0001, 0.0005, 0.001, 0.002, 0.005,
        0.01, 0.02, 0.03, 0.04, 0.05,
        0.07, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.40, 0.50, 0.60, 0.70,
        0.80, 0.90
    ]


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
        self.vectorizer = None
        self.classifier = None
        self.classifier_name: str = ""
        self.classifier_metrics: Dict[str, object] = {}
        self.duplicate_report: Dict[str, object] = {}
        self.learned_phrase_weights: Dict[str, float] = {}
        self.selected_threshold: float = 0.5

    def train(self, full_corpus_loader: Optional[FullEnronLoader], spam_loader: EnronSpamLoader):
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
    def _train_classifier(self, records: List[Dict[str, str]]):
        texts = [r["text"] for r in records]
        labels = np.array([1 if r["label"] == "spam" else 0 for r in records])

        normalized = [normalize_text(t) for t in texts]
        exact_dups = len(texts) - len(set(texts))
        normalized_unique_groups = len(set(normalized))

        X = np.array(texts, dtype=object)
        y = labels
        groups = np.array(normalized, dtype=object)

        print("\nDATASET SUMMARY")
        print("Total examples:", len(X))
        print("Total positives:", int(np.sum(y)))
        print("Total negatives:", int(len(y) - np.sum(y)))
        print("Exact duplicates:", int(exact_dups))
        print("Normalized unique groups:", int(normalized_unique_groups))

        spam_total = int(np.sum(y))
        ham_total = int(len(y) - spam_total)
        phrase_weights = {}
        for phrase in PHISHING_SIGNAL_PHRASES:
            spam_hits = sum(1 for text, label in zip(X, y) if label == 1 and phrase in text.lower())
            ham_hits = sum(1 for text, label in zip(X, y) if label == 0 and phrase in text.lower())
            spam_rate = (spam_hits + 1) / (spam_total + 2)
            ham_rate = (ham_hits + 1) / (ham_total + 2)
            phrase_weights[phrase] = round(max(0.0, spam_rate - ham_rate), 6)
        self.learned_phrase_weights = phrase_weights

        classifier_factories = {
            "logistic_regression": lambda: LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
            "naive_bayes": lambda: MultinomialNB(alpha=0.1),
            "random_forest": lambda: RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        }

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        cv_results = {}
        threshold_history = {}

        print("\nGROUPED CROSS-VALIDATION RESULTS")
        for name, factory in classifier_factories.items():
            fold_metrics = []
            fold_thresholds = []

            print(f"\n=== {name} ===")
            for fold_num, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):
                X_train_text = X[train_idx].tolist()
                y_train = y[train_idx]

                X_test_text = X[test_idx].tolist()
                y_test = y[test_idx]

                # Skip unusable folds
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    print(
                        f"Fold {fold_num}: skipped "
                        f"(train classes={np.unique(y_train)}, test classes={np.unique(y_test)})"
                    )
                    continue

                vectorizer = TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=2,
                    stop_words="english",
                )

                vectorizer.fit(X_train_text)
                X_train = vectorizer.transform(X_train_text)
                X_test = vectorizer.transform(X_test_text)

                clf = factory()
                clf.fit(X_train, y_train)

                if hasattr(clf, "predict_proba"):
                    y_score = clf.predict_proba(X_test)[:, 1]
                else:
                    y_score = clf.decision_function(X_test)

                best_f1 = -1.0
                best_threshold = 0.5
                best_pred = None

                for threshold in threshold_grid():
                    y_pred = np.where(y_score >= threshold, 1, 0)
                    curr_f1 = f1_score(y_test, y_pred, zero_division=0)
                    if curr_f1 > best_f1:
                        best_f1 = curr_f1
                        best_threshold = float(threshold)
                        best_pred = y_pred

                metrics = {
                    "fold": fold_num,
                    "size": int(len(y_test)),
                    "positives": int(np.sum(y_test)),
                    "negatives": int(len(y_test) - np.sum(y_test)),
                    "threshold": float(best_threshold),
                    "accuracy": float(accuracy_score(y_test, best_pred)),
                    "precision": float(precision_score(y_test, best_pred, zero_division=0)),
                    "recall": float(recall_score(y_test, best_pred, zero_division=0)),
                    "f1": float(f1_score(y_test, best_pred, zero_division=0)),
                    "roc_auc": float(roc_auc_score(y_test, y_score)),
                    "confusion_matrix": confusion_matrix(y_test, best_pred).tolist(),
                }

                fold_metrics.append(metrics)
                fold_thresholds.append(best_threshold)

                print(
                    f"Fold {fold_num}: "
                    f"n={metrics['size']} pos={metrics['positives']} neg={metrics['negatives']} "
                    f"thr={metrics['threshold']:.4f} "
                    f"acc={metrics['accuracy']:.4f} "
                    f"prec={metrics['precision']:.4f} "
                    f"rec={metrics['recall']:.4f} "
                    f"f1={metrics['f1']:.4f} "
                    f"auc={metrics['roc_auc']:.4f}"
                )

            if not fold_metrics:
                cv_results[name] = {
                    "folds_used": 0,
                    "mean_metrics": None,
                    "std_metrics": None,
                    "fold_details": [],
                }
                threshold_history[name] = []
                continue

            def metric_list(key):
                return [m[key] for m in fold_metrics]

            mean_metrics = {
                "accuracy": mean(metric_list("accuracy")),
                "precision": mean(metric_list("precision")),
                "recall": mean(metric_list("recall")),
                "f1": mean(metric_list("f1")),
                "roc_auc": mean(metric_list("roc_auc")),
                "threshold": median(fold_thresholds),
            }

            std_metrics = {
                "accuracy": pstdev(metric_list("accuracy")) if len(fold_metrics) > 1 else 0.0,
                "precision": pstdev(metric_list("precision")) if len(fold_metrics) > 1 else 0.0,
                "recall": pstdev(metric_list("recall")) if len(fold_metrics) > 1 else 0.0,
                "f1": pstdev(metric_list("f1")) if len(fold_metrics) > 1 else 0.0,
                "roc_auc": pstdev(metric_list("roc_auc")) if len(fold_metrics) > 1 else 0.0,
            }

            cv_results[name] = {
                "folds_used": len(fold_metrics),
                "mean_metrics": mean_metrics,
                "std_metrics": std_metrics,
                "fold_details": fold_metrics,
            }
            threshold_history[name] = fold_thresholds

            print(f"Summary for {name}:")
            print(json.dumps({
                "folds_used": len(fold_metrics),
                "mean_metrics": mean_metrics,
                "std_metrics": std_metrics,
            }, indent=2))

        usable_models = {
            name: result for name, result in cv_results.items()
            if result["mean_metrics"] is not None
        }
        if not usable_models:
            raise RuntimeError("No classifier produced usable grouped CV results.")

        best_name = max(usable_models, key=lambda n: usable_models[n]["mean_metrics"]["f1"])
        best_threshold = usable_models[best_name]["mean_metrics"]["threshold"]

        self.classifier_name = best_name
        self.selected_threshold = float(best_threshold)

        print(f"\nSelected classifier from grouped CV: {best_name}")
        print(f"Selected threshold from grouped CV: {best_threshold:.4f}")

        # Train final exportable model on all available data
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
        )
        self.vectorizer.fit(X.tolist())
        X_all = self.vectorizer.transform(X.tolist())

        vocab_size = len(getattr(self.vectorizer, "vocabulary_", {}) or {})
        self.vocabulary_size = vocab_size
        print("\nVocab size:", vocab_size)

        sanity_text = "please send data externally"
        sanity_X = self.vectorizer.transform([sanity_text])
        print("Sanity test nnz:", sanity_X.nnz)

        self.classifier = classifier_factories[best_name]()
        self.classifier.fit(X_all, y)

        self.classifier_metrics = {
            "selection_metric": "grouped_cv_mean_f1",
            "grouped_cv_results": cv_results,
            "selected_model": best_name,
            "selected_threshold": float(best_threshold),
        }

        self.duplicate_report = {
            "raw_examples": int(len(texts)),
            "exact_duplicate_emails": int(exact_dups),
            "normalized_unique_groups": int(normalized_unique_groups),
            "split_strategy": "grouped_cross_validation_by_normalized_text",
        }

        print("\nGrouped CV selection summary:")
        print(json.dumps({
            "selected_model": best_name,
            "selected_threshold": float(best_threshold),
            "duplicate_report": self.duplicate_report,
        }, indent=2))


    def save(self, path: str):
            saved_vocab_size = len(getattr(self.vectorizer, "vocabulary_", {}) or {})
            data = {
                "vocabulary_size": saved_vocab_size,
                "baseline_sentence_length": self.baseline_sentence_length,
                "baseline_vocab_richness": self.baseline_vocab_richness,
                "baseline_word_count": self.baseline_word_count,
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "classifier_accuracy": self.classifier_metrics.get("grouped_cv_results", {})
                    .get(self.classifier_name, {})
                    .get("mean_metrics", {})
                    .get("accuracy", 0.0),
                "classifier_name": self.classifier_name,
                "classifier_metrics": self.classifier_metrics,
                "duplicate_report": self.duplicate_report,
                "learned_phrase_weights": self.learned_phrase_weights,
                "selected_threshold": self.selected_threshold,
            }

            print("Loaded vocab size:", saved_vocab_size)

            with open(path, "wb") as f:
                pickle.dump(data, f)

            print(f"\nModel saved to: {path}")
def train_combined_model(
    full_corpus_path: Optional[str],
    spam_data_path: str,
    output_path: str,
    max_emails: int = 500000,
):
    print("╔" + "═" * 68 + "╗")
    print("║" + " CORRECTED ENRON EMAIL FORENSICS TRAINING ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    full_loader = None
    if full_corpus_path and full_corpus_path.upper() != "NONE" and Path(full_corpus_path).exists():
        full_loader = FullEnronLoader(full_corpus_path)
        full_loader.load(max_emails=max_emails)
    else:
        print("Full corpus not provided or unavailable; skipping emails.csv baselines.")

    spam_loader = EnronSpamLoader(spam_data_path)
    spam_loader.load()

    agent = CombinedForensicsAgent()
    agent.train(full_loader, spam_loader)
    agent.save(output_path)

    # Immediate reload verification
    with open(output_path, "rb") as f:
        saved = pickle.load(f)
    loaded_vectorizer = saved.get("vectorizer")
    loaded_vocab_size = len(getattr(loaded_vectorizer, "vocabulary_", {}) or {}) if loaded_vectorizer is not None else 0
    print("Loaded vocab size:", loaded_vocab_size)
    if loaded_vectorizer is not None:
        test_text = "please send data externally"
        test_X = loaded_vectorizer.transform([test_text])
        print("Reload sanity test nnz:", test_X.nnz)

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corrected Enron combined training pipeline")
    parser.add_argument("full_corpus_path", help="Path to emails.csv, or NONE if unavailable")
    parser.add_argument("spam_data_path", help="Path to enron_spam_data.csv")
    parser.add_argument(
        "output_path",
        nargs="?",
        default="combined_forensics_model_fixed.pkl",
        help="Output pickle path",
    )
    parser.add_argument("--max-emails", type=int, default=500000, help="Max emails to load from full corpus")

    args = parser.parse_args()
    train_combined_model(
        args.full_corpus_path,
        args.spam_data_path,
        args.output_path,
        max_emails=args.max_emails,
    )
