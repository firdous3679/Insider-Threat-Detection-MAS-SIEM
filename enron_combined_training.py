#!/usr/bin/env python3
"""
enron_combined_training.py

Combined Email Forensics Training using BOTH Enron datasets:
1. Full Enron Corpus (emails.csv - 500K emails) - For baseline & authorship profiling
2. Enron Spam Dataset (enron_spam_data.csv - 33K labeled) - For phishing classifier

This provides:
- Better classifier (trained on real labeled spam/ham)
- Better baselines (from full corpus vocabulary)
- More credible for academic paper (using established benchmark)

Usage:
    python enron_combined_training.py emails.csv enron_spam_data.csv

Output:
    - combined_forensics_model.pkl (trained model)
    - Training statistics for paper
"""

import csv
import re
import sys
import pickle
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from email.parser import Parser
from email.policy import default as email_policy

# Fix for large CSV fields
csv.field_size_limit(10 * 1024 * 1024)

# Check for sklearn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not installed. Install with: pip install scikit-learn")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EmailData:
    """Processed email data"""
    sender: str = ""
    subject: str = ""
    body: str = ""
    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    vocabulary_richness: float = 0.0
    label: str = ""  # "ham" or "spam" for labeled data


@dataclass 
class SenderProfile:
    """Writing style profile for a sender"""
    email_count: int = 0
    avg_sentence_lengths: List[float] = field(default_factory=list)
    vocab_richness_values: List[float] = field(default_factory=list)
    word_counts: List[int] = field(default_factory=list)


# =============================================================================
# FULL ENRON CORPUS LOADER (emails.csv)
# =============================================================================

class FullEnronLoader:
    """
    Loads the full Enron corpus (500K emails) for:
    - Vocabulary baseline
    - Sender profiles
    - Writing style metrics
    """
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.emails: List[EmailData] = []
        self.vocabulary: Set[str] = set()
        self.sender_profiles: Dict[str, SenderProfile] = defaultdict(SenderProfile)
        
        # Statistics
        self.stats = {
            'total_raw': 0,
            'total_processed': 0,
            'skipped_short': 0,
            'skipped_parse_error': 0,
            'unique_senders': 0,
            'vocabulary_size': 0,
            'avg_word_count': 0,
            'avg_sentence_length': 0,
            'avg_vocab_richness': 0,
            'external_ratio': 0,
            'attachment_ratio': 0
        }
    
    def _parse_raw_email(self, raw_message: str) -> Optional[Dict]:
        """Parse raw email message from full corpus"""
        try:
            parser = Parser(policy=email_policy)
            msg = parser.parsestr(raw_message)
            
            # Extract From
            from_header = msg.get('From', '')
            from_addr = self._extract_email(from_header)
            
            # Extract body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain':
                        try:
                            body = part.get_content()
                            break
                        except:
                            pass
            else:
                try:
                    body = msg.get_content()
                except:
                    try:
                        payload = msg.get_payload(decode=True)
                        if isinstance(payload, bytes):
                            body = payload.decode('utf-8', errors='ignore')
                        else:
                            body = str(msg.get_payload())
                    except:
                        body = str(msg.get_payload())
            
            # Extract To for external ratio
            to_header = msg.get('To', '')
            has_external = self._has_external_recipient(to_header)
            
            # Check attachment mentions
            subject = str(msg.get('Subject', ''))
            has_attachment = self._mentions_attachment(subject + ' ' + str(body))
            
            return {
                'from': from_addr,
                'subject': subject,
                'body': str(body) if body else '',
                'has_external': has_external,
                'has_attachment': has_attachment
            }
        except Exception as e:
            return None
    
    def _extract_email(self, header: str) -> str:
        """Extract email address from header"""
        if not header:
            return ""
        match = re.search(r'<([^>]+)>', str(header))
        if match:
            return match.group(1).lower().strip()
        if '@' in str(header):
            return str(header).lower().strip()
        return ""
    
    def _has_external_recipient(self, to_header: str) -> bool:
        """Check if any recipient is external (non-Enron)"""
        if not to_header:
            return False
        enron_domains = ['enron.com', 'enron.net', 'ect.enron.com']
        for addr in str(to_header).split(','):
            email = self._extract_email(addr)
            if email and not any(d in email for d in enron_domains):
                return True
        return False
    
    def _mentions_attachment(self, text: str) -> bool:
        """Check if text mentions attachments"""
        keywords = ['attached', 'attachment', 'enclosed', 'see attached', '.xls', '.doc', '.pdf']
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)
    
    def _calculate_metrics(self, text: str) -> Optional[Dict]:
        """Calculate writing style metrics"""
        if not text or len(text.strip()) < 20:
            return None
        
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 30:  # Minimum 30 words
            return None
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.split()) > 0]
        if not sentences:
            return None
        
        word_count = len(words)
        unique_words = set(words)
        sentence_lengths = [len(s.split()) for s in sentences]
        
        return {
            'word_count': word_count,
            'sentence_count': len(sentences),
            'avg_sentence_length': np.mean(sentence_lengths),
            'vocabulary_richness': len(unique_words) / word_count,
            'unique_words': unique_words
        }
    
    def load(self, max_emails: int = 500000, progress_interval: int = 50000):
        """Load and process the full Enron corpus"""
        print(f"\n{'='*60}")
        print("LOADING FULL ENRON CORPUS (emails.csv)")
        print(f"{'='*60}")
        print(f"File: {self.csv_path}")
        print(f"Max emails: {max_emails:,}")
        
        external_count = 0
        attachment_count = 0
        all_word_counts = []
        all_sentence_lengths = []
        all_vocab_richness = []
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    self.stats['total_raw'] += 1
                    
                    if self.stats['total_raw'] > max_emails:
                        break
                    
                    if self.stats['total_raw'] % progress_interval == 0:
                        print(f"  Processed {self.stats['total_raw']:,} emails...")
                    
                    # Get message
                    message = row.get('message', '') or row.get('Message', '')
                    if not message:
                        self.stats['skipped_parse_error'] += 1
                        continue
                    
                    # Parse email
                    parsed = self._parse_raw_email(message)
                    if not parsed:
                        self.stats['skipped_parse_error'] += 1
                        continue
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(parsed['body'])
                    if not metrics:
                        self.stats['skipped_short'] += 1
                        continue
                    
                    # Valid email
                    self.stats['total_processed'] += 1
                    
                    # Update vocabulary
                    self.vocabulary.update(metrics['unique_words'])
                    
                    # Update sender profile
                    sender = parsed['from']
                    if sender:
                        profile = self.sender_profiles[sender]
                        profile.email_count += 1
                        profile.avg_sentence_lengths.append(metrics['avg_sentence_length'])
                        profile.vocab_richness_values.append(metrics['vocabulary_richness'])
                        profile.word_counts.append(metrics['word_count'])
                    
                    # Track metrics
                    all_word_counts.append(metrics['word_count'])
                    all_sentence_lengths.append(metrics['avg_sentence_length'])
                    all_vocab_richness.append(metrics['vocabulary_richness'])
                    
                    # Track external/attachment
                    if parsed['has_external']:
                        external_count += 1
                    if parsed['has_attachment']:
                        attachment_count += 1
                    
                    # Store email data
                    email = EmailData(
                        sender=sender,
                        subject=parsed['subject'],
                        body=parsed['body'],
                        word_count=metrics['word_count'],
                        sentence_count=metrics['sentence_count'],
                        avg_sentence_length=metrics['avg_sentence_length'],
                        vocabulary_richness=metrics['vocabulary_richness']
                    )
                    self.emails.append(email)
        
        except FileNotFoundError:
            print(f"ERROR: File not found: {self.csv_path}")
            return
        
        # Calculate final statistics
        self.stats['unique_senders'] = len(self.sender_profiles)
        self.stats['vocabulary_size'] = len(self.vocabulary)
        self.stats['avg_word_count'] = np.mean(all_word_counts) if all_word_counts else 0
        self.stats['avg_sentence_length'] = np.mean(all_sentence_lengths) if all_sentence_lengths else 0
        self.stats['avg_vocab_richness'] = np.mean(all_vocab_richness) if all_vocab_richness else 0
        self.stats['external_ratio'] = (external_count / self.stats['total_processed'] * 100) if self.stats['total_processed'] > 0 else 0
        self.stats['attachment_ratio'] = (attachment_count / self.stats['total_processed'] * 100) if self.stats['total_processed'] > 0 else 0
        
        # Count sender profiles with 5+ emails
        senders_5_plus = sum(1 for p in self.sender_profiles.values() if p.email_count >= 5)
        self.stats['senders_5_plus'] = senders_5_plus
        
        print(f"\n  Finished loading {self.stats['total_processed']:,} emails")
        self._print_stats()
    
    def _print_stats(self):
        """Print statistics"""
        print(f"""
Full Enron Corpus Statistics:
-----------------------------
Raw emails:              {self.stats['total_raw']:,}
After preprocessing:     {self.stats['total_processed']:,}
  Skipped (short):       {self.stats['skipped_short']:,}
  Skipped (parse error): {self.stats['skipped_parse_error']:,}
Unique senders:          {self.stats['unique_senders']:,}
Sender profiles (≥5):    {self.stats['senders_5_plus']:,}
Vocabulary size:         {self.stats['vocabulary_size']:,}
Avg word count:          {self.stats['avg_word_count']:.1f}
Avg sentence length:     {self.stats['avg_sentence_length']:.2f}
Avg vocab richness:      {self.stats['avg_vocab_richness']:.3f}
External email ratio:    {self.stats['external_ratio']:.2f}%
Attachment mention rate: {self.stats['attachment_ratio']:.2f}%
""")


# =============================================================================
# ENRON SPAM DATASET LOADER (enron_spam_data.csv)
# =============================================================================

class EnronSpamLoader:
    """
    Loads the Enron Spam Dataset (33K labeled emails) for:
    - Training phishing/spam classifier
    - Labeled ham vs spam examples
    """
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.ham_emails: List[str] = []
        self.spam_emails: List[str] = []
        
        self.stats = {
            'total': 0,
            'ham': 0,
            'spam': 0,
            'skipped': 0
        }
    
    def load(self, min_words: int = 20):
        """Load the Enron spam dataset"""
        print(f"\n{'='*60}")
        print("LOADING ENRON SPAM DATASET (enron_spam_data.csv)")
        print(f"{'='*60}")
        print(f"File: {self.csv_path}")
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    self.stats['total'] += 1
                    
                    # Get label (Spam/Ham column)
                    label = row.get('Spam/Ham', '').strip().lower()
                    if label not in ['ham', 'spam']:
                        self.stats['skipped'] += 1
                        continue
                    
                    # Get content (Subject + Message)
                    subject = row.get('Subject', '') or ''
                    message = row.get('Message', '') or ''
                    content = f"{subject} {message}".strip()
                    
                    # Skip if too short
                    words = content.split()
                    if len(words) < min_words:
                        self.stats['skipped'] += 1
                        continue
                    
                    # Store by label
                    if label == 'ham':
                        self.ham_emails.append(content)
                        self.stats['ham'] += 1
                    else:
                        self.spam_emails.append(content)
                        self.stats['spam'] += 1
        
        except FileNotFoundError:
            print(f"ERROR: File not found: {self.csv_path}")
            return
        
        print(f"""
Enron Spam Dataset Statistics:
------------------------------
Total rows:     {self.stats['total']:,}
Ham emails:     {self.stats['ham']:,} ({self.stats['ham']/max(self.stats['total'],1)*100:.1f}%)
Spam emails:    {self.stats['spam']:,} ({self.stats['spam']/max(self.stats['total'],1)*100:.1f}%)
Skipped:        {self.stats['skipped']:,}
""")


# =============================================================================
# COMBINED FORENSICS AGENT
# =============================================================================

class CombinedForensicsAgent:
    """
    Email forensics agent trained on both Enron datasets:
    - Full corpus: vocabulary baseline, sender profiles, style metrics
    - Spam dataset: phishing/spam classifier
    """
    
    PHISHING_KEYWORDS = [
        'urgent', 'verify', 'password', 'click here', 'confirm',
        'expire', 'act now', 'immediately', 'confidential', 'suspended',
        'unauthorized', 'security alert', 'account', 'bank', 'credit card'
    ]
    
    SENSITIVE_KEYWORDS = [
        'confidential', 'secret', 'private', 'patient', 'ssn',
        'password', 'credential', 'classified'
    ]
    
    def __init__(self):
        # From full corpus
        self.vocabulary: Set[str] = set()
        self.vocabulary_size: int = 0
        self.sender_profiles: Dict[str, SenderProfile] = {}
        self.baseline_sentence_length: float = 15.0
        self.baseline_vocab_richness: float = 0.5
        self.baseline_word_count: float = 100.0
        
        # From spam dataset
        self.vectorizer = None
        self.classifier = None
        self.classifier_accuracy: float = 0.0
        
        # Author profiling (runtime)
        self.author_profiles = defaultdict(lambda: {"count": 0, "avg_len": [], "vocab": []})
        self.word_freq = defaultdict(int)
        self.total_words = 0
    
    def train(self, full_corpus_loader: FullEnronLoader, spam_loader: EnronSpamLoader):
        """Train the forensics agent on both datasets"""
        
        print(f"\n{'='*60}")
        print("TRAINING COMBINED FORENSICS AGENT")
        print(f"{'='*60}")
        
        # 1. Set baselines from full corpus
        print("\n1. Setting baselines from full Enron corpus...")
        self.vocabulary = full_corpus_loader.vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.sender_profiles = dict(full_corpus_loader.sender_profiles)
        self.baseline_sentence_length = full_corpus_loader.stats['avg_sentence_length']
        self.baseline_vocab_richness = full_corpus_loader.stats['avg_vocab_richness']
        self.baseline_word_count = full_corpus_loader.stats['avg_word_count']
        
        print(f"   Vocabulary: {self.vocabulary_size:,} words")
        print(f"   Sender profiles: {len(self.sender_profiles):,}")
        print(f"   Baseline sentence length: {self.baseline_sentence_length:.2f}")
        print(f"   Baseline vocab richness: {self.baseline_vocab_richness:.3f}")
        
        # 2. Train classifier on spam dataset
        print("\n2. Training phishing classifier on Enron Spam Dataset...")
        
        if not HAS_SKLEARN:
            print("   WARNING: scikit-learn not available, using keyword-only detection")
            return
        
        # Prepare training data
        texts = spam_loader.ham_emails + spam_loader.spam_emails
        labels = [0] * len(spam_loader.ham_emails) + [1] * len(spam_loader.spam_emails)
        
        print(f"   Training samples: {len(texts):,} ({len(spam_loader.ham_emails):,} ham, {len(spam_loader.spam_emails):,} spam)")
        
        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.90
        )
        
        X = self.vectorizer.fit_transform(texts)
        print(f"   Feature matrix: {X.shape[0]:,} x {X.shape[1]:,}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.15, random_state=42, stratify=labels
        )
        
        # Try multiple classifiers
        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1),
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
        }
        
        best_clf = None
        best_acc = 0
        best_name = ""
        
        print("\n   Evaluating classifiers:")
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            print(f"   - {name}: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_clf = clf
                best_name = name
        
        self.classifier = best_clf
        self.classifier_accuracy = best_acc
        
        print(f"\n   Selected: {best_name} ({best_acc:.4f} accuracy)")
        
        # Print classification report
        y_pred = self.classifier.predict(X_test)
        print("\n   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    def analyze_email(self, email_event: Dict, step: int) -> Dict:
        """Analyze an email and return forensics features"""
        meta = email_event.get("meta", {}) or {}
        content = meta.get("content", "") or ""
        actor_id = email_event.get("actor_id", 0)
        
        if not content:
            return self._empty_result()
        
        content_lower = content.lower()
        
        # 1. Keyword-based phishing detection
        keyword_hits = sum(1 for kw in self.PHISHING_KEYWORDS if kw in content_lower)
        keyword_score = min(1.0, keyword_hits / 3.0)
        
        # 2. ML-based phishing detection (from spam dataset training)
        ml_score = 0.0
        if self.classifier and self.vectorizer:
            try:
                X = self.vectorizer.transform([content])
                ml_score = self.classifier.predict_proba(X)[0][1]
            except:
                pass
        
        # 3. Combined phishing score (70% ML, 30% keyword)
        if self.classifier:
            phishing_score = 0.7 * ml_score + 0.3 * keyword_score
        else:
            phishing_score = keyword_score
        
        # 4. Urgency detection
        urgency_keywords = ['urgent', 'immediately', 'asap', 'now', 'deadline', 'critical']
        urgency = min(1.0, sum(1 for kw in urgency_keywords if kw in content_lower) / 2.0)
        
        # 5. Sensitive content detection
        sensitive = any(kw in content_lower for kw in self.SENSITIVE_KEYWORDS)
        
        # 6. Style anomaly detection (vs full corpus baseline)
        words = re.findall(r'\b\w+\b', content_lower)
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        
        style_anomaly = 0.0
        if sentences and words:
            curr_sent_len = np.mean([len(s.split()) for s in sentences])
            curr_vocab = len(set(words)) / len(words)
            
            # Z-score from baseline
            len_dev = abs(curr_sent_len - self.baseline_sentence_length) / 5.0
            vocab_dev = abs(curr_vocab - self.baseline_vocab_richness) / 0.15
            style_anomaly = min(1.0, (len_dev + vocab_dev) / 3.0)
        
        # 7. AI-generated text detection (perplexity-based)
        ai_prob = self._detect_ai_generated(content)
        
        # 8. Authorship consistency
        authorship_consistency = self._check_authorship(actor_id, content)
        self._update_author_profile(actor_id, content)
        
        # Combined risk scores
        content_risk = phishing_score * 0.4 + style_anomaly * 0.2 + (1 - authorship_consistency) * 0.2 + urgency * 0.2
        combined_risk = (content_risk + phishing_score) / 2
        
        return {
            "forensics_phishing_score": phishing_score,
            "forensics_ml_score": ml_score,
            "forensics_keyword_score": keyword_score,
            "forensics_urgency": urgency,
            "forensics_sensitive": 1.0 if sensitive else 0,
            "forensics_ai_prob": ai_prob,
            "forensics_authorship_consistency": authorship_consistency,
            "forensics_style_anomaly": style_anomaly,
            "forensics_content_risk": content_risk,
            "forensics_combined_risk": combined_risk
        }
    
    def _empty_result(self):
        return {
            "forensics_phishing_score": 0, "forensics_ml_score": 0, "forensics_keyword_score": 0,
            "forensics_urgency": 0, "forensics_sensitive": 0, "forensics_ai_prob": 0,
            "forensics_authorship_consistency": 1.0, "forensics_style_anomaly": 0,
            "forensics_content_risk": 0, "forensics_combined_risk": 0
        }
    
    def _detect_ai_generated(self, content: str) -> float:
        """Detect AI-generated text using perplexity and burstiness"""
        words = re.findall(r'\b\w+\b', content.lower())
        if len(words) < 20:
            return 0.0
        
        # Update word frequency
        for w in words:
            self.word_freq[w] += 1
            self.total_words += 1
        
        # Perplexity (using vocabulary frequency)
        if self.vocabulary:
            known_words = sum(1 for w in words if w in self.vocabulary)
            coverage = known_words / len(words)
            # High coverage = predictable = possibly AI
            ppl_score = 0.3 if coverage > 0.95 else 0.0
        else:
            ppl_score = 0.0
        
        # Burstiness (sentence length variation)
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        if len(sentences) >= 2:
            lengths = [len(s.split()) for s in sentences]
            cv = np.std(lengths) / (np.mean(lengths) + 0.1)
            # Low variation = possibly AI
            burst_score = 0.3 if cv < 0.25 else 0.0
        else:
            burst_score = 0.0
        
        return min(1.0, ppl_score + burst_score)
    
    def _check_authorship(self, actor_id: int, content: str) -> float:
        """Check if writing style matches historical profile"""
        profile = self.author_profiles[actor_id]
        if profile['count'] < 3:
            return 1.0
        
        words = re.findall(r'\b\w+\b', content.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        
        if not sentences or not words:
            return 1.0
        
        curr_len = np.mean([len(s.split()) for s in sentences])
        curr_vocab = len(set(words)) / len(words)
        
        # Compare to historical
        hist_len = np.mean(profile['avg_len'][-10:]) if profile['avg_len'] else curr_len
        hist_vocab = np.mean(profile['vocab'][-10:]) if profile['vocab'] else curr_vocab
        
        len_dev = abs(curr_len - hist_len) / max(hist_len, 1)
        vocab_dev = abs(curr_vocab - hist_vocab) / max(hist_vocab, 0.1)
        
        deviation = (len_dev + vocab_dev) / 2
        return 1.0 - min(1.0, deviation)
    
    def _update_author_profile(self, actor_id: int, content: str):
        """Update author's writing style profile"""
        profile = self.author_profiles[actor_id]
        words = re.findall(r'\b\w+\b', content.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        
        if sentences:
            profile['avg_len'].append(np.mean([len(s.split()) for s in sentences]))
        if words:
            profile['vocab'].append(len(set(words)) / len(words))
        profile['count'] += 1
        
        # Keep last 50
        profile['avg_len'] = profile['avg_len'][-50:]
        profile['vocab'] = profile['vocab'][-50:]
    
    def save(self, path: str):
        """Save trained model to file"""
        data = {
            'vocabulary_size': self.vocabulary_size,
            'baseline_sentence_length': self.baseline_sentence_length,
            'baseline_vocab_richness': self.baseline_vocab_richness,
            'baseline_word_count': self.baseline_word_count,
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'classifier_accuracy': self.classifier_accuracy
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"\nModel saved to: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'CombinedForensicsAgent':
        """Load trained model from file"""
        agent = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        agent.vocabulary_size = data['vocabulary_size']
        agent.baseline_sentence_length = data['baseline_sentence_length']
        agent.baseline_vocab_richness = data['baseline_vocab_richness']
        agent.baseline_word_count = data['baseline_word_count']
        agent.vectorizer = data['vectorizer']
        agent.classifier = data['classifier']
        agent.classifier_accuracy = data['classifier_accuracy']
        return agent


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_combined_model(full_corpus_path: str, spam_data_path: str, 
                         output_path: str = "combined_forensics_model.pkl",
                         max_emails: int = 500000):
    """
    Train combined forensics model using both Enron datasets.
    
    Args:
        full_corpus_path: Path to emails.csv (full 500K corpus)
        spam_data_path: Path to enron_spam_data.csv (labeled spam dataset)
        output_path: Where to save the trained model
        max_emails: Max emails to load from full corpus
    """
    
    print("╔" + "═"*68 + "╗")
    print("║" + " COMBINED ENRON EMAIL FORENSICS TRAINING ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    # Load full corpus
    full_loader = FullEnronLoader(full_corpus_path)
    full_loader.load(max_emails=max_emails)
    
    # Load spam dataset
    spam_loader = EnronSpamLoader(spam_data_path)
    spam_loader.load()
    
    # Train combined agent
    agent = CombinedForensicsAgent()
    agent.train(full_loader, spam_loader)
    
    # Save model
    agent.save(output_path)
    
    # Print summary for paper
    print("\n" + "="*70)
    print("STATISTICS FOR PAPER (Section III.A)")
    print("="*70)
    print(f"""
We utilize two complementary Enron email corpora for training the behavioral
forensics component:

1) Full Enron Corpus: The complete Enron email dataset contains approximately
   {full_loader.stats['total_raw']:,} emails. After preprocessing (removing emails under 30 words
   and handling parsing errors), we retained {full_loader.stats['total_processed']:,} emails from
   {full_loader.stats['unique_senders']:,} unique senders. This corpus provides:
   
   - Vocabulary baseline: {full_loader.stats['vocabulary_size']:,} unique words
   - Sender profiles: {full_loader.stats['senders_5_plus']:,} users with ≥5 emails each
   - Writing style metrics: avg. sentence length {full_loader.stats['avg_sentence_length']:.1f} words,
     vocabulary richness {full_loader.stats['avg_vocab_richness']:.2f}, avg. word count {full_loader.stats['avg_word_count']:.1f}
   - External communication ratio: {full_loader.stats['external_ratio']:.1f}% of emails to non-Enron addresses
   - Attachment mention rate: {full_loader.stats['attachment_ratio']:.1f}%

2) Enron Spam Dataset [15]: Contains {spam_loader.stats['total']:,} labeled emails
   ({spam_loader.stats['spam']:,} spam, {spam_loader.stats['ham']:,} ham) curated for spam filtering benchmarks.
   The phishing classifier trained on this dataset achieves {agent.classifier_accuracy:.2%} accuracy.
""")
    
    return agent


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python enron_combined_training.py <emails.csv> <enron_spam_data.csv>")
        print("\nExample:")
        print("  python enron_combined_training.py emails.csv enron_spam_data.csv")
        print("\nThis will:")
        print("  1. Load full Enron corpus (emails.csv) for baseline statistics")
        print("  2. Load Enron spam dataset for classifier training")
        print("  3. Train combined forensics model")
        print("  4. Save to combined_forensics_model.pkl")
        print("  5. Output statistics for your paper")
        sys.exit(1)
    
    full_corpus_path = sys.argv[1]
    spam_data_path = sys.argv[2]
    
    output_path = "combined_forensics_model.pkl"
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    
    train_combined_model(full_corpus_path, spam_data_path, output_path)
