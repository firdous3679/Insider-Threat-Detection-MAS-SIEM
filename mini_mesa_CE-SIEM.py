# advanced_insider_threat_v4.py
# Enhanced with:
# - TomAbd (Theory of Mind + Abductive Reasoning) Framework
# - Behavioral Forensics Pipeline (NLP + AI-Text Detection)
# - All features from V3

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Tuple, Set
from collections import Counter, defaultdict
from enum import Enum
import json
import math
import random
import re
import numpy as np
import mesa
from mesa.datacollection import DataCollector

# Optional ML
try:
    from sklearn.ensemble import IsolationForest
except Exception:
    IsolationForest = None

# Optional NLP (for behavioral forensics)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    HAS_NLP = True
except Exception:
    HAS_NLP = False


# =========================
# TomAbd Framework
# =========================

class GoalType(Enum):
    """Possible insider goals that can be inferred"""
    NONE = "none"
    DATA_THEFT = "data_theft"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SABOTAGE = "sabotage"
    CREDENTIAL_HARVESTING = "credential_harvesting"
    RECONNAISSANCE = "reconnaissance"
    STAGING = "staging"
    EXFILTRATION = "exfiltration"


@dataclass
class Belief:
    """Represents a belief about the world or another agent"""
    subject: str
    predicate: str
    value: Any
    confidence: float = 1.0
    timestamp: int = 0
    
    def decay(self, current_step: int, decay_rate: float = 0.01) -> float:
        age = current_step - self.timestamp
        self.confidence = self.confidence * (1.0 - decay_rate) ** age
        return self.confidence


@dataclass
class InferredGoal:
    """An abductively inferred goal for an agent"""
    goal_type: GoalType
    target: Optional[str] = None
    probability: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    first_observed: int = 0
    last_updated: int = 0


@dataclass
class Plan:
    """A hypothesized plan an agent might be following"""
    plan_id: str
    goal: GoalType
    expected_actions: List[str]
    observed_actions: List[str] = field(default_factory=list)
    completion_ratio: float = 0.0
    confidence: float = 0.0


class TomAbdAgent:
    """
    Theory of Mind + Abductive Reasoning module.
    Based on the TomAbd framework by Montes et al. (2023)
    """
    
    PLAN_TEMPLATES = {
        "exfiltration_direct": Plan(
            plan_id="exfil_direct",
            goal=GoalType.EXFILTRATION,
            expected_actions=["db_query:sensitive", "data_export", "email_send:external"],
        ),
        "exfiltration_staged": Plan(
            plan_id="exfil_staged",
            goal=GoalType.STAGING,
            expected_actions=["db_query:sensitive", "db_query:sensitive", "data_export:internal", "email_send:external"],
        ),
        "reconnaissance": Plan(
            plan_id="recon",
            goal=GoalType.RECONNAISSANCE,
            expected_actions=["login", "db_query", "db_query", "db_query"],
        ),
        "credential_theft": Plan(
            plan_id="cred_theft",
            goal=GoalType.CREDENTIAL_HARVESTING,
            expected_actions=["login:after_hours", "login:new_device", "db_query:sensitive"],
        ),
    }
    
    ABDUCTIVE_RULES = [
        (["db_query:patients_table", "data_export"], GoalType.DATA_THEFT, 0.4),
        (["data_export", "email_send:external"], GoalType.EXFILTRATION, 0.5),
        (["db_query:sensitive", "db_query:sensitive", "db_query:sensitive"], GoalType.RECONNAISSANCE, 0.3),
        (["login:after_hours", "db_query:sensitive"], GoalType.DATA_THEFT, 0.35),
        (["login:new_device", "login:after_hours"], GoalType.CREDENTIAL_HARVESTING, 0.4),
        (["data_export:internal", "data_export:internal"], GoalType.STAGING, 0.45),
    ]
    
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.self_beliefs: Dict[str, Belief] = {}
        self.other_beliefs: Dict[int, Dict[str, Belief]] = defaultdict(dict)
        self.inferred_goals: Dict[int, Dict[GoalType, InferredGoal]] = defaultdict(dict)
        self.plan_hypotheses: Dict[int, List[Plan]] = defaultdict(list)
        self.observed_actions: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        
    def observe_action(self, step: int, actor_id: int, action: Dict[str, Any]):
        action_code = self._encode_action(action)
        self.observed_actions[actor_id].append((step, action_code))
        self.observed_actions[actor_id] = self.observed_actions[actor_id][-100:]
        self._update_beliefs(step, actor_id, action)
        self._abductive_inference(step, actor_id)
        self._update_plan_hypotheses(step, actor_id, action_code)
        
    def _encode_action(self, action: Dict[str, Any]) -> str:
        event_type = action.get("event_type", "unknown")
        resource = action.get("resource", "")
        meta = action.get("meta", {}) or {}
        dst = action.get("dst", "")
        
        code = event_type
        if resource:
            code += f":{resource}"
        if event_type == "email_send":
            if dst.endswith("outside") or dst.endswith("@outside"):
                code += ":external"
            else:
                code += ":internal"
        if event_type == "login":
            if meta.get("after_hours"):
                code += ":after_hours"
            if meta.get("device_change"):
                code += ":new_device"
        if event_type == "data_export":
            if meta.get("destination") == "internal_staging" or "INTERNAL" in str(action.get("action", "")):
                code += ":internal"
        return code
    
    def _update_beliefs(self, step: int, actor_id: int, action: Dict[str, Any]):
        meta = action.get("meta", {}) or {}
        if "role" in meta:
            self.other_beliefs[actor_id]["role"] = Belief(
                subject=f"agent_{actor_id}", predicate="has_role",
                value=meta["role"], confidence=1.0, timestamp=step
            )
        if meta.get("after_hours"):
            existing = self.other_beliefs[actor_id].get("works_after_hours")
            if existing:
                existing.confidence = min(1.0, existing.confidence + 0.1)
                existing.timestamp = step
            else:
                self.other_beliefs[actor_id]["works_after_hours"] = Belief(
                    subject=f"agent_{actor_id}", predicate="works_after_hours",
                    value=True, confidence=0.5, timestamp=step
                )
        if action.get("event_type") == "db_query" and action.get("resource") == "patients_table":
            key = "accesses_sensitive_data"
            existing = self.other_beliefs[actor_id].get(key)
            if existing:
                existing.confidence = min(1.0, existing.confidence + 0.05)
                existing.timestamp = step
            else:
                self.other_beliefs[actor_id][key] = Belief(
                    subject=f"agent_{actor_id}", predicate=key,
                    value=True, confidence=0.3, timestamp=step
                )

    def _abductive_inference(self, step: int, actor_id: int):
        recent = [code for _, code in self.observed_actions[actor_id][-20:]]
        for pattern, goal_type, confidence_boost in self.ABDUCTIVE_RULES:
            if self._pattern_matches(recent, pattern):
                if goal_type not in self.inferred_goals[actor_id]:
                    self.inferred_goals[actor_id][goal_type] = InferredGoal(
                        goal_type=goal_type, probability=confidence_boost,
                        supporting_evidence=[f"pattern:{','.join(pattern)}"],
                        first_observed=step, last_updated=step
                    )
                else:
                    goal = self.inferred_goals[actor_id][goal_type]
                    goal.probability = min(0.95, goal.probability + confidence_boost * (1 - goal.probability))
                    goal.last_updated = step
        for goal in self.inferred_goals[actor_id].values():
            if step - goal.last_updated > 20:
                goal.probability *= 0.95
                
    def _pattern_matches(self, actions: List[str], pattern: List[str]) -> bool:
        if len(pattern) > len(actions):
            return False
        pattern_idx = 0
        for action in actions:
            if pattern_idx >= len(pattern):
                return True
            if action.startswith(pattern[pattern_idx].split(":")[0]):
                if ":" in pattern[pattern_idx]:
                    if pattern[pattern_idx].split(":")[1] in action:
                        pattern_idx += 1
                else:
                    pattern_idx += 1
        return pattern_idx >= len(pattern)
    
    def _update_plan_hypotheses(self, step: int, actor_id: int, action_code: str):
        if not self.plan_hypotheses[actor_id]:
            for name, template in self.PLAN_TEMPLATES.items():
                self.plan_hypotheses[actor_id].append(Plan(
                    plan_id=template.plan_id, goal=template.goal,
                    expected_actions=template.expected_actions.copy(),
                    observed_actions=[], completion_ratio=0.0, confidence=0.1
                ))
        for plan in self.plan_hypotheses[actor_id]:
            next_idx = len(plan.observed_actions)
            if next_idx < len(plan.expected_actions):
                expected = plan.expected_actions[next_idx]
                if action_code.startswith(expected.split(":")[0]):
                    qualifier_match = True
                    if ":" in expected:
                        qualifier_match = expected.split(":")[1] in action_code
                    if qualifier_match:
                        plan.observed_actions.append(action_code)
                        plan.completion_ratio = len(plan.observed_actions) / len(plan.expected_actions)
                        plan.confidence = min(0.9, plan.confidence + 0.15)
                    else:
                        plan.confidence = max(0.05, plan.confidence - 0.02)
                else:
                    plan.confidence = max(0.05, plan.confidence - 0.05)
                    
    def get_threat_assessment(self, actor_id: int) -> Dict[str, Any]:
        assessment = {
            "has_malicious_intent": False, "intent_confidence": 0.0,
            "primary_goal": GoalType.NONE.value, "goal_probability": 0.0,
            "active_plan": None, "plan_completion": 0.0,
            "risk_factors": [], "beliefs_summary": {},
        }
        goals = self.inferred_goals.get(actor_id, {})
        if goals:
            top_goal = max(goals.values(), key=lambda g: g.probability)
            if top_goal.probability > 0.3:
                assessment["has_malicious_intent"] = True
                assessment["intent_confidence"] = top_goal.probability
                assessment["primary_goal"] = top_goal.goal_type.value
                assessment["goal_probability"] = top_goal.probability
                assessment["risk_factors"].extend(top_goal.supporting_evidence[:3])
        plans = self.plan_hypotheses.get(actor_id, [])
        if plans:
            top_plan = max(plans, key=lambda p: p.confidence * p.completion_ratio)
            if top_plan.completion_ratio > 0.3:
                assessment["active_plan"] = top_plan.plan_id
                assessment["plan_completion"] = top_plan.completion_ratio
        return assessment
    
    def get_cognitive_features(self, actor_id: int) -> Dict[str, float]:
        assessment = self.get_threat_assessment(actor_id)
        features = {
            "tom_intent_score": assessment["intent_confidence"],
            "tom_plan_completion": assessment["plan_completion"],
            "tom_data_theft_prob": 0.0, "tom_exfil_prob": 0.0,
            "tom_staging_prob": 0.0, "tom_recon_prob": 0.0,
        }
        goals = self.inferred_goals.get(actor_id, {})
        if GoalType.DATA_THEFT in goals:
            features["tom_data_theft_prob"] = goals[GoalType.DATA_THEFT].probability
        if GoalType.EXFILTRATION in goals:
            features["tom_exfil_prob"] = goals[GoalType.EXFILTRATION].probability
        if GoalType.STAGING in goals:
            features["tom_staging_prob"] = goals[GoalType.STAGING].probability
        if GoalType.RECONNAISSANCE in goals:
            features["tom_recon_prob"] = goals[GoalType.RECONNAISSANCE].probability
        return features


# =========================
# Behavioral Forensics Pipeline
# =========================

class EmailForensicsAgent:
    """
    Behavioral Forensics for Email Analysis.
    Two parallel analysis streams:
    1. Content-based NLP: tokenization, entity extraction, phishing detection
    2. AI-text forensics: synthetic text detection, authorship consistency
    """
    
    PHISHING_KEYWORDS = [
        "urgent", "immediately", "verify", "suspend", "account",
        "password", "click here", "confirm", "expire", "limited time",
        "act now", "winner", "congratulations", "free", "prize"
    ]
    
    SENSITIVE_KEYWORDS = [
        "confidential", "secret", "private", "ssn", "social security",
        "credit card", "password", "patient", "medical", "financial",
        "proprietary", "trade secret", "classified"
    ]
    
    POLICY_VIOLATION_PATTERNS = [
        r"do not share", r"internal only", r"not for distribution", r"delete after reading"
    ]
    
    def __init__(self):
        self.author_profiles: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
            "avg_sentence_length": [], "vocabulary_richness": [],
            "punctuation_patterns": Counter(), "common_phrases": Counter(),
            "email_count": 0, "style_vectors": []
        })
        self.word_frequencies: Counter = Counter()
        self.total_words: int = 0
        self.phishing_classifier = None
        self.vectorizer = None
        if HAS_NLP:
            self._init_classifier()
            
    def _init_classifier(self):
        phishing_samples = [
            "Urgent: Your account has been suspended. Click here to verify.",
            "Congratulations! You've won a free prize. Claim now!",
            "Your password will expire. Update immediately.",
            "Act now to avoid account suspension.",
        ]
        legitimate_samples = [
            "Please find attached the quarterly report.",
            "Meeting scheduled for tomorrow at 3pm.",
            "Here are the documents you requested.",
            "Thank you for your email. I'll review and respond.",
        ]
        texts = phishing_samples + legitimate_samples
        labels = [1] * len(phishing_samples) + [0] * len(legitimate_samples)
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        X = self.vectorizer.fit_transform(texts)
        self.phishing_classifier = MultinomialNB()
        self.phishing_classifier.fit(X, labels)
        
    def analyze_email(self, email_event: Dict[str, Any], step: int) -> Dict[str, Any]:
        meta = email_event.get("meta", {}) or {}
        content = meta.get("content", meta.get("subject", ""))
        actor_id = email_event.get("actor_id", 0)
        nlp_analysis = self._content_nlp_analysis(content)
        ai_forensics = self._ai_text_forensics(content, actor_id)
        integrated = self._integrate_features(nlp_analysis, ai_forensics, email_event)
        self._update_author_profile(actor_id, content)
        return integrated
    
    def _content_nlp_analysis(self, content: str) -> Dict[str, Any]:
        if not content:
            return {"phishing_score": 0.0, "spam_indicators": 0, "sensitive_content": False,
                    "entity_count": 0, "urgency_level": 0.0, "policy_violations": []}
        content_lower = content.lower()
        words = re.findall(r'\b\w+\b', content_lower)
        phishing_count = sum(1 for kw in self.PHISHING_KEYWORDS if kw in content_lower)
        phishing_score = min(1.0, phishing_count / 5.0)
        urgency_words = ["urgent", "immediately", "asap", "now", "quickly", "hurry"]
        urgency_count = sum(1 for w in urgency_words if w in content_lower)
        urgency_level = min(1.0, urgency_count / 3.0)
        sensitive_count = sum(1 for kw in self.SENSITIVE_KEYWORDS if kw in content_lower)
        sensitive_content = sensitive_count > 0
        spam_indicators = content.count("!") + sum(1 for w in content.split() if w.isupper() and len(w) > 2)
        policy_violations = [p for p in self.POLICY_VIOLATION_PATTERNS if re.search(p, content_lower)]
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        url_pattern = r'https?://\S+'
        emails_found = len(re.findall(email_pattern, content))
        urls_found = len(re.findall(url_pattern, content))
        ml_phishing_prob = 0.0
        if self.phishing_classifier and self.vectorizer and content.strip():
            try:
                X = self.vectorizer.transform([content])
                ml_phishing_prob = self.phishing_classifier.predict_proba(X)[0][1]
            except:
                pass
        return {
            "phishing_score": max(phishing_score, ml_phishing_prob),
            "spam_indicators": spam_indicators, "sensitive_content": sensitive_content,
            "sensitive_keyword_count": sensitive_count, "entity_count": emails_found + urls_found,
            "urls_found": urls_found, "urgency_level": urgency_level,
            "policy_violations": policy_violations, "word_count": len(words)
        }
    
    def _ai_text_forensics(self, content: str, actor_id: int) -> Dict[str, Any]:
        if not content or len(content) < 20:
            return {"ai_generated_prob": 0.0, "perplexity_score": 50.0,
                    "burstiness_score": 0.5, "authorship_consistency": 1.0, "style_deviation": 0.0}
        words = re.findall(r'\b\w+\b', content.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        perplexity = self._estimate_perplexity(words)
        burstiness = self._calculate_burstiness(sentences)
        consistency, style_deviation = self._check_authorship_consistency(actor_id, content)
        ai_prob = 0.0
        if perplexity < 30:
            ai_prob += 0.3
        if burstiness < 0.3:
            ai_prob += 0.3
        if style_deviation > 0.5:
            ai_prob += 0.2
        return {"ai_generated_prob": min(1.0, ai_prob), "perplexity_score": perplexity,
                "burstiness_score": burstiness, "authorship_consistency": consistency,
                "style_deviation": style_deviation}
    
    def _estimate_perplexity(self, words: List[str]) -> float:
        if not words:
            return 50.0
        for word in words:
            self.word_frequencies[word] += 1
            self.total_words += 1
        total_prob = sum(self.word_frequencies[w] / max(1, self.total_words) for w in words)
        avg_prob = total_prob / len(words)
        perplexity = 1.0 / max(avg_prob, 0.001)
        return min(100.0, perplexity)
    
    def _calculate_burstiness(self, sentences: List[str]) -> float:
        if len(sentences) < 2:
            return 0.5
        lengths = [len(s.split()) for s in sentences]
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        cv = std_len / max(mean_len, 1)
        return min(1.0, cv / 1.5)
    
    def _check_authorship_consistency(self, actor_id: int, content: str) -> Tuple[float, float]:
        profile = self.author_profiles[actor_id]
        if profile["email_count"] < 3:
            return 1.0, 0.0
        words = re.findall(r'\b\w+\b', content.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        if not sentences:
            return 1.0, 0.0
        current_avg_len = np.mean([len(s.split()) for s in sentences])
        current_vocab_richness = len(set(words)) / max(len(words), 1)
        len_deviation = 0.0
        if profile["avg_sentence_length"]:
            hist_avg_len = np.mean(profile["avg_sentence_length"][-10:])
            len_deviation = abs(current_avg_len - hist_avg_len) / max(hist_avg_len, 1)
        vocab_deviation = 0.0
        if profile["vocabulary_richness"]:
            hist_vocab = np.mean(profile["vocabulary_richness"][-10:])
            vocab_deviation = abs(current_vocab_richness - hist_vocab) / max(hist_vocab, 0.1)
        total_deviation = (len_deviation + vocab_deviation) / 2
        return 1.0 - min(1.0, total_deviation), total_deviation
    
    def _update_author_profile(self, actor_id: int, content: str):
        profile = self.author_profiles[actor_id]
        words = re.findall(r'\b\w+\b', content.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        if sentences:
            profile["avg_sentence_length"].append(np.mean([len(s.split()) for s in sentences]))
        if words:
            profile["vocabulary_richness"].append(len(set(words)) / len(words))
        profile["email_count"] += 1
        profile["avg_sentence_length"] = profile["avg_sentence_length"][-50:]
        profile["vocabulary_richness"] = profile["vocabulary_richness"][-50:]
    
    def _integrate_features(self, nlp: Dict, ai_forensics: Dict, event: Dict) -> Dict[str, Any]:
        meta = event.get("meta", {}) or {}
        content_risk = nlp["phishing_score"] * 0.4 + nlp["urgency_level"] * 0.2
        if nlp["sensitive_content"]:
            content_risk += 0.2
        if nlp["policy_violations"]:
            content_risk += 0.2 * len(nlp["policy_violations"])
        authorship_risk = ai_forensics["ai_generated_prob"] * 0.5 + ai_forensics["style_deviation"] * 0.3
        attachment_kb = meta.get("attachment_kb", 0)
        attachment_risk = min(1.0, attachment_kb / 5000) if attachment_kb > 500 else 0.0
        return {
            "forensics_phishing_score": nlp["phishing_score"],
            "forensics_spam_indicators": nlp["spam_indicators"],
            "forensics_sensitive_content": 1.0 if nlp["sensitive_content"] else 0.0,
            "forensics_urgency": nlp["urgency_level"],
            "forensics_policy_violations": len(nlp.get("policy_violations", [])),
            "forensics_ai_prob": ai_forensics["ai_generated_prob"],
            "forensics_perplexity": ai_forensics["perplexity_score"],
            "forensics_burstiness": ai_forensics["burstiness_score"],
            "forensics_authorship_consistency": ai_forensics["authorship_consistency"],
            "forensics_content_risk": content_risk,
            "forensics_authorship_risk": authorship_risk,
            "forensics_attachment_risk": attachment_risk,
            "forensics_combined_risk": (content_risk + authorship_risk + attachment_risk) / 3,
            "forensics_analyzed": True
        }


# =========================
# Event Schema & SIEM Config
# =========================
@dataclass
class Event:
    step: int
    event_type: str
    actor_id: int
    resource: Optional[str] = None
    action: Optional[str] = None
    dst: Optional[str] = None
    bytes: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None
    label: Optional[str] = None
    scenario: Optional[str] = None
    phase: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SIEMConfig:
    use_policy: bool = True
    use_baseline: bool = True
    use_trust: bool = True
    use_online_learning: bool = True
    use_ml: bool = True
    use_tom: bool = True
    use_forensics: bool = True

    early_threshold: float = 2.8
    base_confirmed_threshold: float = 4.2
    window: int = 48
    cooldown: int = 6

    ewma_alpha: float = 0.08
    ewma_var_alpha: float = 0.08
    z_clip: float = 6.0
    baseline_weight: float = 0.9

    trust_init: float = 0.70
    trust_min: float = 0.10
    trust_max: float = 0.95
    trust_slope: float = 1.2
    trust_decay: float = 0.00
    trust_tp_delta: float = -0.15
    trust_fp_delta: float = +0.04

    lr: float = 0.04
    l2: float = 1e-4
    w_clip: float = 6.0

    ml_weight: float = 2.0
    ml_score_threshold: float = 0.0
    ml_margin: float = 2.0
    ml_weight_confirmed_scale: float = 0.8
    ml_train_sample_every: int = 1
    ml_min_samples: int = 25
    ml_refit_every: int = 15
    ml_freeze_after_warmup: bool = False

    tom_weight: float = 1.5
    tom_intent_threshold: float = 0.4
    forensics_weight: float = 1.2
    forensics_risk_threshold: float = 0.5

    w: Dict[str, float] = field(default_factory=lambda: {
        "anchor_email": 2.0, "anchor_login": 2.0, "export_large": 2.0,
        "export_small": 1.0, "staging_export": 1.5, "after_hours": 1.0,
        "sens_read_burst": 1.0, "login_burst": 1.0, "unapproved_partner": 2.5,
        "email_burst": 1.0, "exfil_chain_tight": 3.5, "exfil_chain_speed": 2.0,
        "staging_pattern": 3.0, "burst_after_quiet": 2.5, "large_export_no_ticket": 2.5,
        "large_attachment_unapproved": 2.5, "email_rate_spike": 2.0,
        "low_ticket_compliance": 2.0, "low_approval_ratio": 2.0,
        "tom_malicious_intent": 2.5, "tom_active_plan": 2.0,
        "forensics_phishing": 2.0, "forensics_ai_generated": 1.5,
        "forensics_policy_violation": 2.5,
    })


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =========================
# Enhanced ML Model
# =========================
class EnhancedRoleAnomalyModel:
    def __init__(self, cfg: SIEMConfig):
        self.cfg = cfg
        self.clean_buffers: Dict[str, List[List[float]]] = defaultdict(list)
        self.all_buffers: Dict[str, List[List[float]]] = defaultdict(list)
        self.models: Dict[str, Any] = {}
        self.last_fit_step: Dict[str, int] = defaultdict(lambda: -10**9)

    def extract_rich_features(self, recent: List[Dict[str, Any]], window_size: int) -> List[float]:
        if not recent:
            return [0.0] * 20
        login_count = sum(1 for r in recent if r.get("event_type") == "login")
        db_count = sum(1 for r in recent if r.get("event_type") == "db_query")
        sens_reads = sum(1 for r in recent if r.get("event_type") == "db_query" and r.get("resource") == "patients_table")
        exports = sum(1 for r in recent if r.get("event_type") == "data_export")
        ext_emails = sum(1 for r in recent if r.get("event_type") == "email_send" and (r.get("dst") or "").endswith("outside"))
        after_hours = sum(1 for r in recent if (r.get("meta") or {}).get("after_hours"))
        sens_ratio = sens_reads / max(1, db_count)
        export_ratio = exports / max(1, db_count)
        external_ratio = ext_emails / max(1, sum(1 for r in recent if r.get("event_type") == "email_send"))
        after_hours_ratio = after_hours / max(1, len(recent))
        total_bytes = sum(r.get("bytes", 0) for r in recent if r.get("event_type") in ["db_query", "data_export"])
        max_bytes = max((r.get("bytes", 0) for r in recent if r.get("event_type") in ["db_query", "data_export"]), default=0)
        large_exports = sum(1 for r in recent if r.get("event_type") == "data_export" and r.get("bytes", 0) > 100000)
        return [
            float(login_count), float(db_count), float(sens_reads), float(exports),
            float(ext_emails), float(after_hours), float(sens_ratio), float(export_ratio),
            float(external_ratio), float(after_hours_ratio), float(total_bytes) / 10000.0,
            float(max_bytes) / 10000.0, float(large_exports), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]

    def add(self, role: str, vec: List[float], actor_id: int, recent: List[Dict[str, Any]]):
        self.all_buffers[role].append(vec)
        self.clean_buffers[role].append(vec)

    def maybe_fit(self, role: str, step: int):
        if IsolationForest is None:
            return
        buffer = self.clean_buffers[role] if len(self.clean_buffers[role]) >= self.cfg.ml_min_samples else self.all_buffers[role]
        if len(buffer) < self.cfg.ml_min_samples:
            return
        if (step - self.last_fit_step[role]) < self.cfg.ml_refit_every:
            return
        X = buffer[-min(500, len(buffer)):]
        clf = IsolationForest(n_estimators=250, max_samples='auto', contamination=0.01, random_state=42, bootstrap=True)
        clf.fit(X)
        self.models[role] = clf
        self.last_fit_step[role] = step

    def score(self, role: str, vec: List[float]) -> float:
        if IsolationForest is None:
            return 0.0
        clf = self.models.get(role)
        if clf is None:
            return 0.0
        raw_score = -float(clf.decision_function([vec])[0])
        raw_score = max(0.0, raw_score)
        if raw_score < self.cfg.ml_score_threshold:
            return 0.0
        excess = raw_score - self.cfg.ml_score_threshold
        return min(excess * (1.0 + excess * 0.5), 5.0)


# =========================
# Agents
# =========================
class BenignEmployee(mesa.Agent):
    def __init__(self, model: mesa.Model, role: str = "staff"):
        super().__init__(model)
        self.role = role
        self.on_call = (self.role == "staff" and self.random.random() < 0.12)
        self.device_id = f"corp-laptop-{self.unique_id}"
        self.model.register_user(self.unique_id, role=self.role,
            allowed_after_hours=(self.role in {"analyst", "admin"}) or self.on_call,
            approved_partners=set() if self.role == "staff" else {"partner@outside"})

    def act(self):
        s = self.model.steps
        hour = s % 24
        at_work = 8 <= hour <= 17
        after_hours = not at_work
        if self.random.random() < (0.08 if at_work else 0.01):
            self.model.emit_action(Event(step=s, event_type="login", actor_id=self.unique_id,
                meta={"role": self.role, "after_hours": after_hours}, label="benign"))
        if at_work and self.random.random() < 0.28:
            table = self.random.choice(["public_table", "project_table", "patients_table"])
            self.model.emit_action(Event(step=s, event_type="db_query", actor_id=self.unique_id,
                resource=table, action="SELECT", bytes=self.random.randint(200, 6000),
                meta={"role": self.role, "after_hours": after_hours}, label="benign"))
        if at_work and self.random.random() < 0.10:
            self.model.emit_action(Event(step=s, event_type="email_send", actor_id=self.unique_id,
                dst="someone@org", meta={"attachment_kb": 0, "subject": "question",
                    "content": "Hi, just following up. Thanks", "role": self.role}, label="benign"))


class BenignPowerUser(mesa.Agent):
    def __init__(self, model: mesa.Model, role: str = "analyst", report_every: int = 36):
        super().__init__(model)
        self.role = role
        self.report_every = report_every
        self.in_cycle = False
        self.phase = 0
        self.cycle_ticket_id = None
        self.model.register_user(self.unique_id, role=self.role, allowed_after_hours=True,
            approved_partners={"partner@outside"})

    def act(self):
        s = self.model.steps
        hour = s % 24
        at_work = 8 <= hour <= 17
        after_hours = not at_work
        if at_work and self.random.random() < 0.22:
            self.model.emit_action(Event(step=s, event_type="db_query", actor_id=self.unique_id,
                resource="patients_table", action="SELECT", bytes=self.random.randint(2000, 16000),
                meta={"role": self.role, "after_hours": after_hours}, label="benign"))
        if (s % self.report_every == 0) and not self.in_cycle:
            self.in_cycle = True
            self.phase = 1
            self.cycle_ticket_id = f"TCKT-{(self.unique_id*1000 + s) % 99999:05d}"
        if self.in_cycle:
            if self.phase == 1:
                self.model.emit_action(Event(step=s, event_type="db_query", actor_id=self.unique_id,
                    resource="patients_table", action="SELECT", bytes=self.random.randint(3000, 25000),
                    meta={"role": self.role, "cycle": "partner_reporting", "ticket_id": self.cycle_ticket_id},
                    label="benign"))
                self.phase = 2
            elif self.phase == 2:
                self.model.emit_action(Event(step=s, event_type="data_export", actor_id=self.unique_id,
                    resource="patients_table", action="EXPORT", bytes=self.random.randint(25000, 180000),
                    meta={"role": self.role, "cycle": "partner_reporting", "ticket_id": self.cycle_ticket_id},
                    label="benign"))
                self.phase = 3
            elif self.phase == 3:
                self.model.emit_action(Event(step=s, event_type="email_send", actor_id=self.unique_id,
                    dst="partner@outside", meta={"role": self.role, "cycle": "partner_reporting",
                        "ticket_id": self.cycle_ticket_id, "approved_partner": True,
                        "attachment_kb": self.random.randint(220, 2500), "subject": "partner report",
                        "content": "Please find attached the scheduled partner report."}, label="benign"))
                self.in_cycle = False
                self.phase = 0


class MaliciousInsider(mesa.Agent):
    def __init__(self, model: mesa.Model, scenario: str, start_step: int, repeat_every: Optional[int] = 48):
        super().__init__(model)
        self.scenario = scenario
        self.start_step = start_step
        self.repeat_every = repeat_every
        self.phase = 0
        self.model.register_user(self.unique_id, role="staff", allowed_after_hours=False, approved_partners=set())

    def _after_hours(self, step: int) -> bool:
        hour = step % 24
        return (hour < 7) or (hour > 19)

    def act(self):
        if not getattr(self.model, "attack_enabled", True):
            return
        s = self.model.steps
        after_hours = self._after_hours(s)
        if self.phase == 0:
            if s >= self.start_step:
                self.phase = 1
            else:
                return

        if self.scenario == "exfil":
            if self.phase == 1:
                self.model.emit_action(Event(step=s, event_type="db_query", actor_id=self.unique_id,
                    resource="patients_table", action="SELECT", bytes=self.random.randint(8000, 70000),
                    meta={"after_hours": after_hours}, label="malicious", scenario=self.scenario))
                if self.random.random() < 0.60:
                    self.phase = 2
            elif self.phase == 2:
                self.model.emit_action(Event(step=s, event_type="data_export", actor_id=self.unique_id,
                    resource="patients_table", action="EXPORT", bytes=self.random.randint(80000, 260000),
                    meta={"after_hours": after_hours}, label="malicious", scenario=self.scenario))
                self.phase = 3
            elif self.phase == 3:
                self.model.emit_action(Event(step=s, event_type="email_send", actor_id=self.unique_id,
                    dst="external@outside", meta={"attachment_kb": self.random.randint(600, 4000),
                        "subject": "Urgent: Patient records",
                        "content": "Here is the confidential patient data. Confirm receipt immediately.",
                        "after_hours": after_hours}, label="malicious", scenario=self.scenario))
                self.phase = 4

        elif self.scenario == "stealth":
            if self.phase == 1:
                if self.random.random() < 0.35:
                    self.model.emit_action(Event(step=s, event_type="db_query", actor_id=self.unique_id,
                        resource="patients_table", action="SELECT", bytes=self.random.randint(800, 5000),
                        meta={"after_hours": after_hours}, label="malicious", scenario=self.scenario))
                if self.random.random() < 0.08:
                    self.phase = 3
            elif self.phase == 3:
                self.model.emit_action(Event(step=s, event_type="email_send", actor_id=self.unique_id,
                    dst="external@outside", meta={"attachment_kb": self.random.randint(300, 2500),
                        "subject": "summary", "content": "Attached summary as discussed.",
                        "after_hours": after_hours}, label="malicious", scenario=self.scenario))
                self.phase = 4

        elif self.scenario == "acct_takeover":
            if after_hours and self.random.random() < 0.70:
                self.model.emit_action(Event(step=s, event_type="login", actor_id=self.unique_id,
                    meta={"role": "admin", "after_hours": True, "takeover": True, "device_change": True},
                    label="malicious", scenario=self.scenario))
                self.model.emit_action(Event(step=s, event_type="db_query", actor_id=self.unique_id,
                    resource="patients_table", action="SELECT", bytes=self.random.randint(1500, 9000),
                    meta={"after_hours": True}, label="malicious", scenario=self.scenario))

        elif self.scenario == "staging_exfil":
            if self.phase == 1:
                for _ in range(3):
                    self.model.emit_action(Event(step=s, event_type="db_query", actor_id=self.unique_id,
                        resource="patients_table", action="SELECT", bytes=self.random.randint(5000, 45000),
                        meta={"after_hours": after_hours}, label="malicious", scenario=self.scenario))
                if self.random.random() < 0.55:
                    self.phase = 2
            elif self.phase == 2:
                self.model.emit_action(Event(step=s, event_type="data_export", actor_id=self.unique_id,
                    resource="patients_table", action="EXPORT_INTERNAL", bytes=self.random.randint(60000, 220000),
                    meta={"after_hours": after_hours, "destination": "internal_staging"},
                    label="malicious", scenario=self.scenario))
                self.phase = 3
            elif self.phase == 3:
                self.model.emit_action(Event(step=s, event_type="email_send", actor_id=self.unique_id,
                    dst="external@outside", meta={"attachment_kb": self.random.randint(600, 4500),
                        "subject": "report", "content": "Private data attached. Do not share.",
                        "after_hours": after_hours}, label="malicious", scenario=self.scenario))
                self.phase = 4

        elif self.scenario == "email_only":
            if self.random.random() < 0.20:
                self.model.emit_action(Event(step=s, event_type="email_send", actor_id=self.unique_id,
                    dst="external@outside", meta={"attachment_kb": self.random.randint(250, 2500),
                        "subject": "doc", "content": "Confidential documents. Act now!",
                        "after_hours": after_hours}, label="malicious", scenario=self.scenario))

        if self.phase == 4 and self.repeat_every is not None:
            if (s - self.start_step) % self.repeat_every == 0 and s > self.start_step:
                self.phase = 1


# =========================
# Monitoring Agents
# =========================
class DBMonitor(mesa.Agent):
    def __init__(self, model: mesa.Model):
        super().__init__(model)
    
    def monitor(self):
        s = self.model.steps
        for e in self.model.action_events:
            if e.event_type in {"db_query", "data_export"}:
                self.model.emit_ingest(Event(step=s, event_type="siem_ingest", actor_id=e.actor_id,
                    meta={"source": "db", "record": e.to_dict()}, label=e.label, scenario=e.scenario))


class EmailMonitor(mesa.Agent):
    def __init__(self, model: mesa.Model):
        super().__init__(model)
        self.forensics = EmailForensicsAgent()
    
    def monitor(self):
        s = self.model.steps
        for e in self.model.action_events:
            if e.event_type == "email_send":
                forensics_results = {}
                if getattr(self.model, 'siem_cfg', None) and self.model.siem_cfg.use_forensics:
                    forensics_results = self.forensics.analyze_email(e.to_dict(), s)
                record = e.to_dict()
                record["forensics"] = forensics_results
                self.model.emit_ingest(Event(step=s, event_type="siem_ingest", actor_id=e.actor_id,
                    meta={"source": "email", "record": record, "forensics": forensics_results},
                    label=e.label, scenario=e.scenario))


class AuthMonitor(mesa.Agent):
    def __init__(self, model: mesa.Model):
        super().__init__(model)
    
    def monitor(self):
        s = self.model.steps
        for e in self.model.action_events:
            if e.event_type == "login":
                self.model.emit_ingest(Event(step=s, event_type="siem_ingest", actor_id=e.actor_id,
                    meta={"source": "auth", "record": e.to_dict()}, label=e.label, scenario=e.scenario))


# =========================
# SIEM Agent (Enhanced with ToM + Forensics)
# =========================
class SIEMAgent(mesa.Agent):
    def __init__(self, model: mesa.Model, cfg: SIEMConfig):
        super().__init__(model)
        self.cfg = cfg
        self.last_alert_step: Dict[Tuple[int, str], int] = {}
        self.recent_by_user: Dict[int, List[Dict[str, Any]]] = {}
        self.mu: Dict[int, Dict[str, float]] = {}
        self.var: Dict[int, Dict[str, float]] = {}
        self.trust: Dict[int, float] = {}
        self.ml = EnhancedRoleAnomalyModel(cfg)
        self.tom_agents: Dict[int, TomAbdAgent] = {}
        self.forensics_cache: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    def _get_tom_agent(self, actor_id: int) -> TomAbdAgent:
        if actor_id not in self.tom_agents:
            self.tom_agents[actor_id] = TomAbdAgent(actor_id)
        return self.tom_agents[actor_id]

    def _is_external(self, dst: str) -> bool:
        d = (dst or "").lower()
        return d.endswith("outside") or d.endswith("@outside")

    def _is_email_anchor(self, rec: Dict[str, Any]) -> bool:
        if rec.get("event_type") != "email_send":
            return False
        meta = rec.get("meta") or {}
        return (rec.get("dst") or "").endswith("outside") and meta.get("attachment_kb", 0) > 200

    def _is_login_anchor(self, rec: Dict[str, Any]) -> bool:
        if rec.get("event_type") != "login":
            return False
        meta = rec.get("meta") or {}
        return (meta.get("after_hours") and meta.get("role") == "admin") or meta.get("takeover")

    def extract_features(self, recent: List[Dict[str, Any]], actor_id: int) -> Dict[str, float]:
        feats = {k: 0.0 for k in self.cfg.w.keys()}
        login_count = sum(1 for r in recent if r.get("event_type") == "login")
        sens_reads = sum(1 for r in recent if r.get("event_type") == "db_query" and r.get("resource") == "patients_table")
        external_emails = sum(1 for r in recent if r.get("event_type") == "email_send"
            and (r.get("dst") or "").endswith("outside") and ((r.get("meta") or {}).get("attachment_kb", 0) > 200))
        feats["login_burst"] = 1.0 if login_count >= 3 else 0.0
        feats["sens_read_burst"] = 1.0 if sens_reads >= 6 else 0.0
        feats["email_burst"] = 1.0 if external_emails >= 3 else 0.0
        feats["after_hours"] = 1.0 if any((r.get("meta") or {}).get("after_hours") for r in recent) else 0.0
        export_bytes = [r.get("bytes", 0) for r in recent if r.get("event_type") == "data_export"
            and r.get("resource") == "patients_table" and r.get("action") == "EXPORT"]
        if export_bytes:
            mx = max(export_bytes)
            feats["export_large"] = 1.0 if mx >= 120000 else 0.0
            feats["export_small"] = 1.0 if mx < 120000 else 0.0
        feats["staging_export"] = 1.0 if any(r.get("event_type") == "data_export"
            and r.get("action") in {"EXPORT_INTERNAL", "EXPORT_STAGING"} for r in recent) else 0.0
        for i in range(len(recent) - 2):
            if (recent[i].get("event_type") == "db_query" and recent[i].get("resource") == "patients_table"
                and recent[i+1].get("event_type") == "data_export"
                and recent[i+2].get("event_type") == "email_send" and self._is_external(recent[i+2].get("dst", ""))):
                step_gap = recent[i+2].get("step", 0) - recent[i].get("step", 0)
                if step_gap <= 5:
                    feats["exfil_chain_tight"] = 1.0
                    break
        if self.cfg.use_tom:
            tom_agent = self._get_tom_agent(actor_id)
            tom_features = tom_agent.get_cognitive_features(actor_id)
            if tom_features["tom_intent_score"] > self.cfg.tom_intent_threshold:
                feats["tom_malicious_intent"] = tom_features["tom_intent_score"]
            if tom_features["tom_plan_completion"] > 0.5:
                feats["tom_active_plan"] = tom_features["tom_plan_completion"]
        if self.cfg.use_forensics:
            recent_forensics = self.forensics_cache.get(actor_id, [])[-10:]
            if recent_forensics:
                avg_phishing = np.mean([f.get("forensics_phishing_score", 0) for f in recent_forensics])
                avg_ai_prob = np.mean([f.get("forensics_ai_prob", 0) for f in recent_forensics])
                total_violations = sum(f.get("forensics_policy_violations", 0) for f in recent_forensics)
                if avg_phishing > 0.5:
                    feats["forensics_phishing"] = avg_phishing
                if avg_ai_prob > 0.5:
                    feats["forensics_ai_generated"] = avg_ai_prob
                if total_violations > 0:
                    feats["forensics_policy_violation"] = min(1.0, total_violations / 3)
        return feats

    def _score(self, feats: Dict[str, float]) -> float:
        return sum(self.cfg.w.get(k, 0.0) * float(feats.get(k, 0.0)) for k in feats)

    def apply_policy(self, actor_id: int, recent: List[Dict[str, Any]], feats: Dict[str, float], reasons: List[str]) -> Tuple[float, float]:
        mult, offset = 1.0, 0.0
        in_cycle = any(((r.get("meta") or {}).get("cycle") == "partner_reporting") for r in recent)
        if in_cycle and any(bool((r.get("meta") or {}).get("ticket_id")) for r in recent):
            mult *= 0.50
            offset += -1.2
            reasons.append("policy:partner_reporting_ticket")
        approved = self.model.user_profile.get(actor_id, {}).get("approved_partners", set()) or set()
        for r in recent:
            if r.get("event_type") == "email_send" and self._is_external(r.get("dst", "")):
                dst = r.get("dst", "")
                if dst not in approved and not (r.get("meta") or {}).get("approved_partner", False):
                    feats["unapproved_partner"] = 1.0
                    reasons.append("policy:unapproved_partner")
                    break
        return mult, offset

    def baseline_anomaly(self, actor_id: int, feats: Dict[str, float], reasons: List[str]) -> float:
        baseline_keys = ["export_large", "export_small", "staging_export", "after_hours", "sens_read_burst", "login_burst"]
        self.mu.setdefault(actor_id, {k: 0.0 for k in baseline_keys})
        self.var.setdefault(actor_id, {k: 1.0 for k in baseline_keys})
        zsum = 0.0
        for k in baseline_keys:
            x = float(feats.get(k, 0.0))
            m, v = self.mu[actor_id][k], self.var[actor_id][k]
            z = max(0.0, min(self.cfg.z_clip, (x - m) / (math.sqrt(v) + 1e-6)))
            zsum += z
            new_m = (1 - self.cfg.ewma_alpha) * m + self.cfg.ewma_alpha * x
            self.mu[actor_id][k] = new_m
            self.var[actor_id][k] = (1 - self.cfg.ewma_var_alpha) * v + self.cfg.ewma_var_alpha * (x - new_m) ** 2
        if zsum > 0.0:
            reasons.append(f"baseline:zsum={zsum:.2f}")
        return zsum * self.cfg.baseline_weight

    def confirmed_threshold(self, actor_id: int) -> float:
        base = self.cfg.base_confirmed_threshold
        if not self.cfg.use_trust:
            return base
        t = self.trust.get(actor_id, self.cfg.trust_init)
        return base + self.cfg.trust_slope * (t - 0.5)

    def update_trust(self, actor_id: int, is_malicious: bool):
        if not self.cfg.use_trust:
            return
        t = self.trust.get(actor_id, self.cfg.trust_init)
        t += (self.cfg.trust_tp_delta if is_malicious else self.cfg.trust_fp_delta)
        self.trust[actor_id] = _clip(t, self.cfg.trust_min, self.cfg.trust_max)

    def _can_fire(self, actor_id: int, tier: str, step: int) -> bool:
        return (step - self.last_alert_step.get((actor_id, tier), -10**9)) >= self.cfg.cooldown

    def ml_vector(self, recent: List[Dict[str, Any]]) -> List[float]:
        return self.ml.extract_rich_features(recent, self.cfg.window)

    def correlate(self):
        s = self.model.steps
        for ing in self.model.ingest_events:
            user = ing.actor_id
            rec = (ing.meta or {}).get("record", {})
            if not rec:
                continue
            self.recent_by_user.setdefault(user, []).append(rec)
            self.recent_by_user[user] = self.recent_by_user[user][-2000:]
            if self.cfg.use_tom:
                self._get_tom_agent(user).observe_action(s, user, rec)
            if self.cfg.use_forensics:
                forensics = (ing.meta or {}).get("forensics", {})
                if forensics:
                    self.forensics_cache[user].append(forensics)
                    self.forensics_cache[user] = self.forensics_cache[user][-50:]

        phase = getattr(self.model, "phase", "test")
        if self.cfg.use_ml and (phase == "train" or not self.cfg.ml_freeze_after_warmup) and (s % self.cfg.ml_train_sample_every == 0):
            for user, hist in self.recent_by_user.items():
                recent = hist[-self.cfg.window:]
                if recent:
                    role = self.model.user_profile.get(user, {}).get("role", "staff")
                    self.ml.add(role, self.ml_vector(recent), user, recent)
                    self.ml.maybe_fit(role, s)

        for ing in self.model.ingest_events:
            user = ing.actor_id
            rec = (ing.meta or {}).get("record", {})
            if not rec:
                continue
            if not (self._is_email_anchor(rec) or self._is_login_anchor(rec)):
                continue
            recent = self.recent_by_user.get(user, [])[-self.cfg.window:]
            reasons = []
            feats = self.extract_features(recent, user)
            feats["anchor_email"] = 1.0 if self._is_email_anchor(rec) else 0.0
            feats["anchor_login"] = 1.0 if self._is_login_anchor(rec) else 0.0
            score = self._score(feats)
            mult, offset = self.apply_policy(user, recent, feats, reasons) if self.cfg.use_policy else (1.0, 0.0)
            score = (score + offset) * mult
            if self.cfg.use_baseline:
                score += self.baseline_anomaly(user, feats, reasons)
            if self.cfg.use_tom and feats.get("tom_malicious_intent", 0) > 0:
                score += self.cfg.tom_weight * feats["tom_malicious_intent"]
                reasons.append(f"tom:intent={feats['tom_malicious_intent']:.2f}")
            if self.cfg.use_forensics:
                forensics_contrib = self.cfg.forensics_weight * (feats.get("forensics_phishing", 0) + feats.get("forensics_policy_violation", 0))
                if forensics_contrib > 0:
                    score += forensics_contrib
                    reasons.append(f"forensics:risk={forensics_contrib:.2f}")

            early_thr, conf_thr = self.cfg.early_threshold, self.confirmed_threshold(user)
            tom_assessment = self._get_tom_agent(user).get_threat_assessment(user) if self.cfg.use_tom else {}

            if score >= early_thr and self._can_fire(user, "early", s):
                self.model.emit_alert(Event(step=s, event_type="alert_early", actor_id=user,
                    meta={"score": score, "threshold": early_thr, "reasons": reasons, "tom_assessment": tom_assessment},
                    label=rec.get("label"), scenario=rec.get("scenario")))
                self.last_alert_step[(user, "early")] = s

            if score >= conf_thr and self._can_fire(user, "confirmed", s):
                self.model.emit_alert(Event(step=s, event_type="alert_confirmed", actor_id=user,
                    meta={"score": score, "threshold": conf_thr, "reasons": reasons, "tom_assessment": tom_assessment},
                    label=rec.get("label"), scenario=rec.get("scenario")))
                self.last_alert_step[(user, "confirmed")] = s
                self.update_trust(user, rec.get("label") == "malicious")
                self.model.event_log.append(Event(step=s, event_type="analyst_verdict", actor_id=user,
                    meta={"verdict": "malicious" if rec.get("label") == "malicious" else "benign"},
                    label=rec.get("label"), scenario=rec.get("scenario")))


# =========================
# Model
# =========================
class InsiderModel(mesa.Model):
    def __init__(self, n_employees: int = 30, n_power_users: int = 4, n_malicious_exfil: int = 3,
                 n_malicious_stealth: int = 2, n_malicious_acct_takeover: int = 1,
                 n_malicious_staging_exfil: int = 1, n_malicious_email_only: int = 1,
                 seed: int = 42, siem_cfg: Optional[SIEMConfig] = None, warmup_steps: int = 60):
        super().__init__(seed=seed)
        self.steps = 0
        self.action_events: List[Event] = []
        self.ingest_events: List[Event] = []
        self.alert_events: List[Event] = []
        self.event_log: List[Event] = []
        self.user_profile: Dict[int, Dict[str, Any]] = {}
        self.siem_cfg = siem_cfg or SIEMConfig()
        self.warmup_steps = warmup_steps
        self.phase = "train"
        self.attack_enabled = False

        for _ in range(n_employees):
            role = self.random.choices(["staff", "analyst", "admin"], weights=[0.78, 0.18, 0.04], k=1)[0]
            BenignEmployee(self, role=role)
        for _ in range(n_power_users):
            BenignPowerUser(self, role="analyst", report_every=self.random.choice([24, 36, 48]))

        base_start = 16 + warmup_steps
        for i in range(n_malicious_exfil):
            MaliciousInsider(self, scenario="exfil", start_step=base_start + i * 12, repeat_every=48)
        for j in range(n_malicious_stealth):
            MaliciousInsider(self, scenario="stealth", start_step=28 + warmup_steps + j * 18, repeat_every=48)
        for k in range(n_malicious_acct_takeover):
            MaliciousInsider(self, scenario="acct_takeover", start_step=22 + warmup_steps + k * 20, repeat_every=None)
        for k in range(n_malicious_staging_exfil):
            MaliciousInsider(self, scenario="staging_exfil", start_step=34 + warmup_steps + k * 24, repeat_every=72)
        for k in range(n_malicious_email_only):
            MaliciousInsider(self, scenario="email_only", start_step=18 + warmup_steps + k * 26, repeat_every=None)

        DBMonitor(self)
        EmailMonitor(self)
        AuthMonitor(self)
        SIEMAgent(self, cfg=self.siem_cfg)
        self.datacollector = DataCollector(model_reporters={
            "alerts_this_step": lambda m: len(m.alert_events),
            "malicious_actions_this_step": lambda m: sum(1 for e in m.action_events if e.label == "malicious"),
            "total_events_logged": lambda m: len(m.event_log), "phase": lambda m: m.phase,
        })

    def register_user(self, actor_id: int, role: str, allowed_after_hours: bool, approved_partners: Set[str]):
        self.user_profile.setdefault(actor_id, {})
        self.user_profile[actor_id].update({"role": role, "allowed_after_hours": allowed_after_hours,
                                            "approved_partners": set(approved_partners)})

    def _tag_phase(self, e: Event):
        e.phase = self.phase
        if e.meta is None:
            e.meta = {}
        e.meta["phase"] = self.phase

    def emit_action(self, e: Event):
        self._tag_phase(e)
        self.action_events.append(e)
        self.event_log.append(e)

    def emit_ingest(self, e: Event):
        self._tag_phase(e)
        self.ingest_events.append(e)
        self.event_log.append(e)

    def emit_alert(self, e: Event):
        self._tag_phase(e)
        self.alert_events.append(e)
        self.event_log.append(e)

    def step(self):
        self.steps += 1
        if self.steps >= self.warmup_steps:
            self.phase = "test"
            self.attack_enabled = True
        self.action_events, self.ingest_events, self.alert_events = [], [], []
        for agent in self.agents:
            if isinstance(agent, (BenignEmployee, BenignPowerUser, MaliciousInsider)):
                agent.act()
        for agent in self.agents:
            if isinstance(agent, (DBMonitor, EmailMonitor, AuthMonitor)):
                agent.monitor()
        for agent in self.agents:
            if isinstance(agent, SIEMAgent):
                agent.correlate()
        self.datacollector.collect(self)


# =========================
# Evaluation
# =========================
def evaluate_run(event_log: List[Event], warmup_steps: int) -> Dict[str, Any]:
    test_events = [e for e in event_log if e.phase == "test" or (e.meta or {}).get("phase") == "test"]
    confirmed_alerts = [e for e in test_events if e.event_type == "alert_confirmed"]
    early_alerts = [e for e in test_events if e.event_type == "alert_early"]
    malicious_actions = [e for e in test_events if e.label == "malicious" and e.event_type not in {"siem_ingest", "analyst_verdict"}]
    malicious_actors = set(e.actor_id for e in malicious_actions)
    confirmed_alerted_actors = set(e.actor_id for e in confirmed_alerts)
    
    tp_confirmed = sum(1 for e in confirmed_alerts if e.label == "malicious")
    fp_confirmed = sum(1 for e in confirmed_alerts if e.label != "malicious")
    tp_early = sum(1 for e in early_alerts if e.label == "malicious")
    fp_early = sum(1 for e in early_alerts if e.label != "malicious")
    
    detected_actors = confirmed_alerted_actors & malicious_actors
    actor_precision = len(detected_actors) / len(confirmed_alerted_actors) if confirmed_alerted_actors else 0.0
    actor_recall = len(detected_actors) / len(malicious_actors) if malicious_actors else 0.0
    actor_f1 = 2 * actor_precision * actor_recall / (actor_precision + actor_recall) if (actor_precision + actor_recall) > 0 else 0.0
    
    ttd_list = []
    for actor in detected_actors:
        first_mal = min((e.step for e in malicious_actions if e.actor_id == actor), default=None)
        first_alert = min((e.step for e in confirmed_alerts if e.actor_id == actor), default=None)
        if first_mal is not None and first_alert is not None:
            ttd_list.append(first_alert - first_mal)
    
    tom_detected = sum(1 for e in confirmed_alerts if (e.meta or {}).get("tom_assessment", {}).get("has_malicious_intent"))
    
    return {
        "actor_precision": actor_precision, "actor_recall": actor_recall, "actor_f1": actor_f1,
        "ttd_avg": np.mean(ttd_list) if ttd_list else 0.0, "ttd_max": max(ttd_list) if ttd_list else 0.0,
        "early_alert_precision": tp_early / len(early_alerts) if early_alerts else 0.0,
        "early_alerts_total": len(early_alerts), "early_alerts_fp": fp_early,
        "conf_alert_precision": tp_confirmed / len(confirmed_alerts) if confirmed_alerts else 0.0,
        "conf_alerts_total": len(confirmed_alerts), "conf_alerts_fp": fp_confirmed,
        "actors_total": len(malicious_actors), "actors_detected": len(detected_actors),
        "tom_detected_count": tom_detected,
    }


def run(T: int = 240, out_jsonl: str = "events_v4.jsonl", siem_cfg: Optional[SIEMConfig] = None,
        warmup_steps: int = 60, runs: int = 10):
    all_results = []
    for run_idx in range(runs):
        print(f"\n=== Run {run_idx + 1}/{runs} ===")
        model = InsiderModel(n_employees=30, n_power_users=4, n_malicious_exfil=3, n_malicious_stealth=2,
            n_malicious_acct_takeover=1, n_malicious_staging_exfil=1, n_malicious_email_only=1,
            seed=42 + run_idx, siem_cfg=siem_cfg, warmup_steps=warmup_steps)
        for _ in range(T):
            model.step()
        metrics = evaluate_run(model.event_log, warmup_steps)
        all_results.append(metrics)
        print(f"Actor F1: {metrics['actor_f1']:.3f} (P={metrics['actor_precision']:.3f}, R={metrics['actor_recall']:.3f})")
        print(f"Confirmed alerts: {metrics['conf_alerts_total']} (precision={metrics['conf_alert_precision']:.3f})")
        print(f"ToM-assisted detections: {metrics['tom_detected_count']}")
       
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for e in model.event_log:
            f.write(json.dumps(e.to_dict()) + "\n")
    
    avg_results = {key: np.mean([r[key] for r in all_results]) for key in all_results[0].keys()}
    print("\n" + "="*60)
    print(f"AVERAGED RESULTS ACROSS {runs} RUNS")
    print("="*60)
    print(f"Actor F1: {avg_results['actor_f1']:.4f} (P={avg_results['actor_precision']:.4f}, R={avg_results['actor_recall']:.4f})")
    print(f"TTD: avg={avg_results['ttd_avg']:.2f}, max={avg_results['ttd_max']:.2f}")
    print(f"Confirmed: {avg_results['conf_alerts_total']:.1f} alerts, precision={avg_results['conf_alert_precision']:.4f}")
    print(f"ToM detections: {avg_results['tom_detected_count']:.1f}")
    print(f"Actors:      {avg_results['actors_detected']:.1f}/{avg_results['actors_total']:.1f} detected")
    return avg_results


if __name__ == "__main__":
    cfg = SIEMConfig(use_policy=True, use_baseline=True, use_trust=True, use_online_learning=True,
        use_ml=True, use_tom=True, use_forensics=True, early_threshold=2.5, base_confirmed_threshold=3.5)
    
    print("="*70)
    print("ADVANCED INSIDER THREAT DETECTION SYSTEM v4")
    print("With TomAbd (Theory of Mind) and Behavioral Forensics")
    print("="*70)
    print("\nNew Features:")
    print("  TomAbd: Belief modeling, Goal inference, Plan tracking")
    print("  Forensics: NLP phishing detection, AI-text forensics, Authorship analysis")
    print("="*70 + "\n")
    
    run(T=240, out_jsonl="events_v4.jsonl", siem_cfg=cfg, warmup_steps=60, runs=10)