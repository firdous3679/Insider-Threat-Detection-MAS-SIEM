#!/usr/bin/env python3
"""
Corrected Enron combined training pipeline.

Fixes applied:
- No TF-IDF leakage: split raw texts before fitting the vectorizer.
- Proper model selection: choose classifier on validation folds only, then report on an untouched test set.
- Duplicate safeguards: normalize content, report exact/normalized duplicates, and deduplicate before splitting.
- Reproducible metrics: accuracy, precision, recall, F1, ROC-AUC (when available), confusion matrix.
- Saves corrected model to a separate output file by default.

Usage:
    python enron_combined_training.py emails.csv enron_spam_data.csv combined_forensics_model_fixed.pkl
    python enron_combined_training.py NONE enron_spam_data.csv combined_forensics_model_fixed.pkl
"""

import argparse
import csv
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from email.parser import Parser
from email.policy import default as email_policy
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

csv.field_size_limit(10 * 1024 * 1024)

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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB

RANDOM_STATE = 42

PHISHING_SIGNAL_PHRASES = [
    "urgent", "verify", "password", "click here", "confirm",
    "expire", "act now", "immediately", "confidential", "suspended",
    "unauthorized", "security alert", "account", "bank", "credit card",
    "private", "credential", "ssn"
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
            "avg_word_count": 0,
            "avg_sentence_length": 0,
            "avg_vocab_richness": 0,
            "external_ratio": 0,
            "attachment_ratio": 0,
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
                        body = payload.decode("utf-8", errors="ignore") if isinstance(payload, bytes) else str(msg.get_payload())
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
            if email and not any(d in email for d in enron_domains):
                return True
        return False

    @staticmethod
    def _mentions_attachment(text: str) -> bool:
        keywords = ["attached", "attachment", "enclosed", "see attached", ".xls", ".doc", ".pdf"]
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)

    @staticmethod
    def _calculate_metrics(text: str) -> Optional[Dict]:
        if not text or len(text.strip()) < 20:
            return None
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < 30:
            return None
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip() and len(s.split()) > 0]
        if not sentences:
            return None
        word_count = len(words)
        unique_words = set(words)
        sentence_lengths = [len(s.split()) for s in sentences]
        return {
            "word_count": word_count,
            "sentence_count": len(sentences),
            "avg_sentence_length": float(np.mean(sentence_lengths)),
            "vocabulary_richness": float(len(unique_words) / word_count),
            "unique_words": unique_words,
        }

    def load(self, max_emails: int = 500000, progress_interval: int = 50000):
        print(f"\n{'=' * 60}\nLOADING FULL ENRON CORPUS (emails.csv)\n{'=' * 60}")
        print(f"File: {self.csv_path}\nMax emails: {max_emails:,}")
        external_count = 0
        attachment_count = 0
        all_word_counts, all_sentence_lengths, all_vocab_richness = [], [], []
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
        self.stats["external_ratio"] = (external_count / self.stats["total_processed"] * 100) if self.stats["total_processed"] else 0.0
        self.stats["attachment_ratio"] = (attachment_count / self.stats["total_processed"] * 100) if self.stats["total_processed"] else 0.0
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
        norm_counter = Counter(normalized)
        duplicated_norm_groups = sum(1 for c in norm_counter.values() if c > 1)

        grouped_records = defaultdict(list)
        conflicting_label_duplicates = 0
        for text, label, norm in zip(texts, labels, normalized):
            grouped_records[norm].append((text, int(label), norm))
        for entries in grouped_records.values():
            if len({label for _, label, _ in entries}) > 1:
                conflicting_label_duplicates += 1

        groups = []
        for norm, entries in grouped_records.items():
            label_counts = Counter(label for _, label, _ in entries)
            group_label = 1 if label_counts[1] >= label_counts[0] else 0
            groups.append((norm, group_label, entries))

        group_norms = [g[0] for g in groups]
        group_labels = [g[1] for g in groups]
        trainval_norms, test_norms, _, _ = train_test_split(
            group_norms,
            group_labels,
            test_size=0.15,
            random_state=RANDOM_STATE,
            stratify=group_labels,
        )
        train_norms, val_norms, _, _ = train_test_split(
            trainval_norms,
            [1 if sum(label for _, label, _ in grouped_records[n]) >= (len(grouped_records[n]) / 2) else 0 for n in trainval_norms],
            test_size=0.17647058823529413,
            random_state=RANDOM_STATE,
            stratify=[1 if sum(label for _, label, _ in grouped_records[n]) >= (len(grouped_records[n]) / 2) else 0 for n in trainval_norms],
        )

        def expand(norm_list):
            rows = []
            for norm in norm_list:
                rows.extend(grouped_records[norm])
            return rows

        train_rows = expand(train_norms)
        val_rows = expand(val_norms)
        test_rows = expand(test_norms)

        X_train_text = [r[0] for r in train_rows]
        y_train = np.array([r[1] for r in train_rows])
        X_val_text = [r[0] for r in val_rows]
        y_val = np.array([r[1] for r in val_rows])
        X_trainval_text = [r[0] for r in train_rows + val_rows]
        y_trainval = np.array([r[1] for r in train_rows + val_rows])
        X_test_text = [r[0] for r in test_rows]
        y_test = np.array([r[1] for r in test_rows])

        overlap_train_val = len(set(train_norms) & set(val_norms))
        overlap_train_test = len(set(trainval_norms) & set(test_norms))
        if overlap_train_val or overlap_train_test:
            raise RuntimeError('Normalized text overlap remained across splits')

        spam_train = sum(y_train)
        ham_train = len(y_train) - spam_train
        phrase_weights = {}
        for phrase in PHISHING_SIGNAL_PHRASES:
            spam_hits = sum(1 for text, label in zip(X_train_text, y_train) if label == 1 and phrase in text.lower())
            ham_hits = sum(1 for text, label in zip(X_train_text, y_train) if label == 0 and phrase in text.lower())
            spam_rate = (spam_hits + 1) / (spam_train + 2)
            ham_rate = (ham_hits + 1) / (ham_train + 2)
            phrase_weights[phrase] = round(max(0.0, spam_rate - ham_rate), 6)
        self.learned_phrase_weights = phrase_weights

        classifier_factories = {
            "logistic_regression": lambda: LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
            "naive_bayes": lambda: MultinomialNB(alpha=0.1),
            "random_forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=30, random_state=RANDOM_STATE, n_jobs=-1),
        }

        val_results = {}
        print("\nValidation metrics by classifier:")
        for name, factory in classifier_factories.items():
            vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), min_df=2, stop_words="english", extra_phrases=PHISHING_SIGNAL_PHRASES)
            X_train = vectorizer.fit_transform(X_train_text)
            X_val = vectorizer.transform(X_val_text)
            clf = factory()
            clf.fit(X_train, y_train)
            y_val_pred = clf.predict(X_val)
            metrics = {
                "accuracy": accuracy_score(y_val, y_val_pred),
                "precision": precision_score(y_val, y_val_pred, zero_division=0),
                "recall": recall_score(y_val, y_val_pred, zero_division=0),
                "f1": f1_score(y_val, y_val_pred, zero_division=0),
            }
            if hasattr(clf, "predict_proba"):
                y_val_prob = clf.predict_proba(X_val)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_val, y_val_prob)
            val_results[name] = metrics
            print(f"  {name}: {metrics}")

        best_name = max(val_results, key=lambda n: val_results[n]["f1"])
        self.classifier_name = best_name
        print(f"\nSelected classifier from validation set: {best_name}")

        self.vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), min_df=2, stop_words="english", extra_phrases=PHISHING_SIGNAL_PHRASES)
        X_trainval = self.vectorizer.fit_transform(X_trainval_text)
        X_test = self.vectorizer.transform(X_test_text)
        self.classifier = classifier_factories[best_name]()
        self.classifier.fit(X_trainval, y_trainval)

        y_test_pred = self.classifier.predict(X_test)
        y_test_prob = self.classifier.predict_proba(X_test)[:, 1] if hasattr(self.classifier, "predict_proba") else None
        test_metrics = {
            "accuracy": float(accuracy_score(y_test, y_test_pred)),
            "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
            "support": int(len(y_test)),
        }
        if y_test_prob is not None:
            test_metrics["roc_auc"] = float(roc_auc_score(y_test, y_test_prob))

        self.classifier_metrics = {
            "selection_metric": "validation_f1",
            "validation_results": val_results,
            "test_metrics": test_metrics,
            "split": {
                "train_size": int(len(X_train_text)),
                "validation_size": int(len(X_val_text)),
                "test_size": int(len(X_test_text)),
                "train_fraction": round(len(X_train_text) / len(texts), 6),
                "validation_fraction": round(len(X_val_text) / len(texts), 6),
                "test_fraction": round(len(X_test_text) / len(texts), 6),
                "random_state": RANDOM_STATE,
            },
        }
        self.duplicate_report = {
            "raw_examples": int(len(texts)),
            "exact_duplicate_emails": int(exact_dups),
            "normalized_unique_groups": int(len(grouped_records)),
            "conflicting_label_duplicate_groups": int(conflicting_label_duplicates),
            "normalized_overlap_train_validation": int(overlap_train_val),
            "normalized_overlap_trainval_test": int(overlap_train_test),
        }
        print("\nFinal untouched test metrics:")
        print(json.dumps(test_metrics, indent=2))
        print("\nDuplicate leakage report:")
        print(json.dumps(self.duplicate_report, indent=2))

    def save(self, path: str):
        data = {
            "vocabulary_size": self.vocabulary_size,
            "baseline_sentence_length": self.baseline_sentence_length,
            "baseline_vocab_richness": self.baseline_vocab_richness,
            "baseline_word_count": self.baseline_word_count,
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "classifier_accuracy": self.classifier_metrics.get("test_metrics", {}).get("accuracy", 0.0),
            "classifier_name": self.classifier_name,
            "classifier_metrics": self.classifier_metrics,
            "duplicate_report": self.duplicate_report,
            "learned_phrase_weights": self.learned_phrase_weights,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"\nModel saved to: {path}")


def train_combined_model(full_corpus_path: Optional[str], spam_data_path: str, output_path: str, max_emails: int = 500000):
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
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corrected Enron combined training pipeline")
    parser.add_argument("full_corpus_path", help="Path to emails.csv, or NONE if unavailable")
    parser.add_argument("spam_data_path", help="Path to enron_spam_data.csv")
    parser.add_argument("output_path", nargs="?", default="combined_forensics_model_fixed.pkl", help="Output pickle path")
    parser.add_argument("--max-emails", type=int, default=500000, help="Max emails to load from full corpus")
    args = parser.parse_args()
    train_combined_model(args.full_corpus_path, args.spam_data_path, args.output_path, max_emails=args.max_emails)
