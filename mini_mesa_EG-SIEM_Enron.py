#!/usr/bin/env python3
"""
mini_mesa_EG-SIEM_combined.py

Evidence-Gated SIEM with Combined Enron-Trained Forensics

This version integrates the combined forensics model trained on:
- Full Enron Corpus (500K emails) - for baselines
- Enron Spam Dataset (33K labeled) - for classifier (99.98% accuracy)

Usage:
    python mini_mesa_EG-SIEM_combined.py
    
    # Or with custom model path:
    python mini_mesa_EG-SIEM_combined.py --model combined_forensics_model.pkl
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Tuple, Set
from collections import Counter, defaultdict
from enum import Enum
import json, math, re, os, sys
import numpy as np
import mesa
import pickle

try:
    from sklearn.ensemble import IsolationForest
except:
    IsolationForest = None

# ============================================================================
# LOAD COMBINED FORENSICS MODEL
# ============================================================================

class CombinedForensicsAgent:
    """
    Email forensics agent using the combined Enron-trained model.
    Loads from pickle file trained by enron_combined_training.py
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
    
    def __init__(self, model_path: str = None):
        # Baselines (will be loaded from model or use defaults)
        self.vocabulary_size = 575895
        self.baseline_sentence_length = 14.60
        self.baseline_vocab_richness = 0.633
        self.baseline_word_count = 341.0
        
        # Classifier (will be loaded from model)
        self.vectorizer = None
        self.classifier = None
        self.classifier_accuracy = 0.0
        
        # Runtime profiling
        self.author_profiles = defaultdict(lambda: {"count": 0, "avg_len": [], "vocab": []})
        self.word_freq = defaultdict(int)
        self.total_words = 0
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print(f"Note: No trained model loaded. Using keyword-only detection.")
    
    def _load_model(self, path: str):
        """Load trained model from pickle file"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.vocabulary_size = data.get('vocabulary_size', self.vocabulary_size)
            self.baseline_sentence_length = data.get('baseline_sentence_length', self.baseline_sentence_length)
            self.baseline_vocab_richness = data.get('baseline_vocab_richness', self.baseline_vocab_richness)
            self.baseline_word_count = data.get('baseline_word_count', self.baseline_word_count)
            self.vectorizer = data.get('vectorizer')
            self.classifier = data.get('classifier')
            self.classifier_accuracy = data.get('classifier_accuracy', 0.0)
            
            print(f"Loaded combined forensics model from: {path}")
            print(f"  Classifier accuracy: {self.classifier_accuracy:.2%}")
            print(f"  Vocabulary size: {self.vocabulary_size:,}")
            print(f"  Baseline sentence length: {self.baseline_sentence_length:.2f}")
        except Exception as e:
            print(f"Warning: Could not load model from {path}: {e}")
    
    def analyze_email(self, email_event: Dict, step: int) -> Dict:
        """Analyze email and return forensics features"""
        meta = email_event.get("meta", {}) or {}
        content = meta.get("content", "") or ""
        actor_id = email_event.get("actor_id", 0)
        
        if not content:
            return self._empty_result()
        
        content_lower = content.lower()
        
        # 1. Keyword-based detection
        keyword_hits = sum(1 for kw in self.PHISHING_KEYWORDS if kw in content_lower)
        keyword_score = min(1.0, keyword_hits / 3.0)
        
        # 2. ML-based detection (from trained classifier)
        ml_score = 0.0
        if self.classifier and self.vectorizer:
            try:
                X = self.vectorizer.transform([content])
                ml_score = self.classifier.predict_proba(X)[0][1]
            except:
                pass
        
        # 3. Combined phishing score
        if self.classifier:
            phishing_score = 0.7 * ml_score + 0.3 * keyword_score
        else:
            phishing_score = keyword_score
        
        # 4. Urgency detection
        urgency_keywords = ['urgent', 'immediately', 'asap', 'now', 'deadline']
        urgency = min(1.0, sum(1 for kw in urgency_keywords if kw in content_lower) / 2.0)
        
        # 5. Sensitive content
        sensitive = any(kw in content_lower for kw in self.SENSITIVE_KEYWORDS)
        
        # 6. Style anomaly (vs Enron baseline)
        style_anomaly = self._calculate_style_anomaly(content)
        
        # 7. AI detection
        ai_prob = self._detect_ai_generated(content)
        
        # 8. Authorship consistency
        authorship = self._check_authorship(actor_id, content)
        self._update_author_profile(actor_id, content)
        
        # Risk scores
        content_risk = phishing_score * 0.4 + style_anomaly * 0.2 + (1 - authorship) * 0.2 + urgency * 0.2
        combined_risk = (content_risk + phishing_score) / 2
        
        return {
            "forensics_phishing_score": phishing_score,
            "forensics_ml_score": ml_score,
            "forensics_keyword_score": keyword_score,
            "forensics_urgency": urgency,
            "forensics_sensitive": 1.0 if sensitive else 0,
            "forensics_ai_prob": ai_prob,
            "forensics_authorship_consistency": authorship,
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
    
    def _calculate_style_anomaly(self, content: str) -> float:
        """Calculate style anomaly vs Enron baseline"""
        words = re.findall(r'\b\w+\b', content.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        
        if not sentences or not words:
            return 0.0
        
        curr_sent_len = np.mean([len(s.split()) for s in sentences])
        curr_vocab = len(set(words)) / len(words)
        
        len_dev = abs(curr_sent_len - self.baseline_sentence_length) / 5.0
        vocab_dev = abs(curr_vocab - self.baseline_vocab_richness) / 0.15
        
        return min(1.0, (len_dev + vocab_dev) / 3.0)
    
    def _detect_ai_generated(self, content: str) -> float:
        """Detect AI-generated text"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        if len(sentences) < 2:
            return 0.0
        
        lengths = [len(s.split()) for s in sentences]
        cv = np.std(lengths) / (np.mean(lengths) + 0.1)
        return 0.3 if cv < 0.25 else 0.0
    
    def _check_authorship(self, actor_id: int, content: str) -> float:
        """Check authorship consistency"""
        profile = self.author_profiles[actor_id]
        if profile['count'] < 3:
            return 1.0
        
        words = re.findall(r'\b\w+\b', content.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        
        if not sentences or not words:
            return 1.0
        
        curr_len = np.mean([len(s.split()) for s in sentences])
        curr_vocab = len(set(words)) / len(words)
        
        hist_len = np.mean(profile['avg_len'][-10:]) if profile['avg_len'] else curr_len
        hist_vocab = np.mean(profile['vocab'][-10:]) if profile['vocab'] else curr_vocab
        
        deviation = (abs(curr_len - hist_len) / max(hist_len, 1) + 
                    abs(curr_vocab - hist_vocab) / max(hist_vocab, 0.1)) / 2
        return 1.0 - min(1.0, deviation)
    
    def _update_author_profile(self, actor_id: int, content: str):
        """Update author profile"""
        profile = self.author_profiles[actor_id]
        words = re.findall(r'\b\w+\b', content.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        
        if sentences:
            profile['avg_len'].append(np.mean([len(s.split()) for s in sentences]))
        if words:
            profile['vocab'].append(len(set(words)) / len(words))
        profile['count'] += 1
        profile['avg_len'] = profile['avg_len'][-50:]
        profile['vocab'] = profile['vocab'][-50:]


# ============================================================================
# TomAbd Framework
# ============================================================================

class GoalType(Enum):
    NONE = "none"
    DATA_THEFT = "data_theft"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    CREDENTIAL_HARVESTING = "credential_harvesting"
    RECONNAISSANCE = "reconnaissance"
    STAGING = "staging"
    EXFILTRATION = "exfiltration"

@dataclass
class InferredGoal:
    goal_type: GoalType
    probability: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    last_updated: int = 0

@dataclass
class Plan:
    plan_id: str
    goal: GoalType
    expected_actions: List[str]
    observed_actions: List[str] = field(default_factory=list)
    completion_ratio: float = 0.0
    confidence: float = 0.0

class TomAbdAgent:
    PLAN_TEMPLATES = {
        "exfil_direct": ("exfil_direct", GoalType.EXFILTRATION, ["db_query:sensitive", "data_export", "email_send:external"]),
        "exfil_staged": ("exfil_staged", GoalType.STAGING, ["db_query:sensitive", "db_query:sensitive", "data_export:internal", "email_send:external"]),
        "recon": ("recon", GoalType.RECONNAISSANCE, ["login", "db_query", "db_query", "db_query"]),
    }
    ABDUCTIVE_RULES = [
        (["db_query:patients_table", "data_export"], GoalType.DATA_THEFT, 0.4),
        (["data_export", "email_send:external"], GoalType.EXFILTRATION, 0.5),
        (["db_query:sensitive", "db_query:sensitive", "db_query:sensitive"], GoalType.RECONNAISSANCE, 0.3),
        (["login:after_hours", "db_query:sensitive"], GoalType.DATA_THEFT, 0.35),
    ]

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.inferred_goals: Dict[int, Dict[GoalType, InferredGoal]] = defaultdict(dict)
        self.plan_hypotheses: Dict[int, List[Plan]] = defaultdict(list)
        self.observed_actions: Dict[int, List[Tuple[int, str]]] = defaultdict(list)

    def observe_action(self, step: int, actor_id: int, action: Dict[str, Any]):
        code = self._encode_action(action)
        self.observed_actions[actor_id].append((step, code))
        self.observed_actions[actor_id] = self.observed_actions[actor_id][-100:]
        contras = self._check_contradictions(action)
        if contras:
            for g in self.inferred_goals[actor_id].values():
                g.probability *= 0.85 ** len(contras)
                g.contradicting_evidence.extend(contras)
        self._abductive_inference(step, actor_id)
        self._update_plans(step, actor_id, code)

    def _encode_action(self, action: Dict) -> str:
        et = action.get("event_type", "unknown")
        res = action.get("resource", "")
        meta = action.get("meta", {}) or {}
        dst = action.get("dst", "")
        code = et
        if res: code += f":{res}"
        if et == "email_send": code += ":external" if dst.endswith("outside") else ":internal"
        if et == "login":
            if meta.get("after_hours"): code += ":after_hours"
            if meta.get("device_change"): code += ":new_device"
        if et == "data_export" and (meta.get("destination") == "internal_staging" or "INTERNAL" in str(action.get("action", ""))):
            code += ":internal"
        return code

    def _check_contradictions(self, action: Dict) -> List[str]:
        contras = []
        et = action.get("event_type")
        meta = action.get("meta", {}) or {}
        if et == "login" and meta.get("compliance_training"):
            contras.append("compliance_training")
        if et == "data_export" and meta.get("approved"):
            contras.append("approved_export")
        return contras

    def _abductive_inference(self, step: int, actor_id: int):
        recent = [c for _, c in self.observed_actions[actor_id][-20:]]
        for pattern, goal, prob in self.ABDUCTIVE_RULES:
            if self._match_pattern(recent, pattern):
                if goal not in self.inferred_goals[actor_id]:
                    self.inferred_goals[actor_id][goal] = InferredGoal(goal)
                g = self.inferred_goals[actor_id][goal]
                g.probability = min(1.0, g.probability + prob * 0.5)
                g.supporting_evidence.append(f"pattern:{pattern}")
                g.last_updated = step

    def _match_pattern(self, recent: List[str], pattern: List[str]) -> bool:
        pi = 0
        for code in recent:
            if pi < len(pattern) and pattern[pi] in code:
                pi += 1
        return pi == len(pattern)

    def _update_plans(self, step: int, actor_id: int, code: str):
        if not self.plan_hypotheses[actor_id]:
            for name, (pid, goal, expected) in self.PLAN_TEMPLATES.items():
                self.plan_hypotheses[actor_id].append(Plan(pid, goal, expected.copy()))
        for plan in self.plan_hypotheses[actor_id]:
            if plan.expected_actions and any(e in code for e in plan.expected_actions[:2]):
                plan.observed_actions.append(code)
                plan.completion_ratio = len(plan.observed_actions) / len(plan.expected_actions)
                plan.confidence = min(1.0, plan.completion_ratio * 0.8)

    def get_threat_assessment(self, actor_id: int) -> Dict[str, Any]:
        goals = self.inferred_goals.get(actor_id, {})
        plans = self.plan_hypotheses.get(actor_id, [])
        max_goal_prob = max((g.probability for g in goals.values()), default=0)
        max_plan_conf = max((p.confidence for p in plans), default=0)
        return {"tom_intent_score": max_goal_prob, "tom_plan_score": max_plan_conf, 
                "tom_combined": max(max_goal_prob, max_plan_conf)}


# ============================================================================
# Event & Config
# ============================================================================

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
    def to_dict(self): return asdict(self)

@dataclass
class SIEMConfig:
    use_policy: bool = True
    use_baseline: bool = True
    use_trust: bool = True
    use_ml: bool = True
    use_tom: bool = True
    use_forensics: bool = True
    use_evidence_gate: bool = True
    use_peer_norm: bool = True
    use_regularity: bool = True
    early_threshold: float = 2.0
    base_confirmed_threshold: float = 4.0
    window: int = 48
    cooldown: int = 6
    min_evidence_count: int = 2
    min_evidence_weight: float = 2.5
    ewma_alpha: float = 0.08
    z_clip: float = 6.0
    baseline_weight: float = 0.9
    trust_init: float = 0.70
    trust_min: float = 0.10
    trust_max: float = 0.95
    trust_slope: float = 1.2
    trust_tp_delta: float = -0.18
    trust_fp_delta: float = 0.05
    tom_weight: float = 2.0
    tom_threshold: float = 0.30
    forensics_weight: float = 1.5
    role_adj: Dict[str, float] = field(default_factory=lambda: {"staff": 0, "analyst": 0.8, "admin": 1.2})
    ml_min: int = 25
    ml_refit: int = 15
    w: Dict[str, float] = field(default_factory=lambda: {
        "anchor_email": 1.5, "anchor_login": 1.5, "export_large": 2.5, "export_small": 1.0,
        "staging_export": 2.0, "after_hours": 1.2, "sens_burst": 1.2, "login_burst": 1.0,
        "unapproved": 3.0, "email_burst": 1.0, "exfil_chain": 4.0,
        "tom_intent": 3.0, "tom_plan": 2.5, "forensics_phishing": 2.5, "peer_dev": 2.0, "irreg": 1.5,
        "stealth_pattern": 3.0, "ext_email_count": 0.0})


# ============================================================================
# Agents
# ============================================================================

class BenignEmployee(mesa.Agent):
    def __init__(self, model, role="staff"):
        super().__init__(model)
        self.role = role
        ah = model.random.random() < 0.10
        approved = [f"partner{i}@approved.com" for i in range(model.random.randint(0, 2))]
        model.register_user(self.unique_id, role, ah, approved)

    def act(self):
        s = self.model.steps
        ah = self.model.random.random() < 0.06
        if self.model.random.random() < 0.35:
            self.model.emit_action(Event(s, "login", self.unique_id, meta={"role": self.role, "after_hours": ah}))
        if self.model.random.random() < 0.30:
            res = self.model.random.choice(["patients_table", "inventory", "logs", "configs"])
            self.model.emit_action(Event(s, "db_query", self.unique_id, resource=res, action="SELECT",
                bytes=self.model.random.randint(50, 6000), meta={"after_hours": ah}))
        if self.model.random.random() < 0.08:
            self.model.emit_action(Event(s, "email_send", self.unique_id, dst="colleague@internal",
                meta={"attachment_kb": self.model.random.randint(0, 80), "content": "Quick update on the project status.", "after_hours": ah}, label="benign"))


class BenignPowerUser(mesa.Agent):
    def __init__(self, model, role="analyst", cycle=36):
        super().__init__(model)
        self.role, self.cycle, self.phase, self.in_cycle = role, cycle, 0, False
        model.register_user(self.unique_id, role, True, [f"partner{i}@approved.com" for i in range(3)])

    def act(self):
        s = self.model.steps
        if self.model.random.random() < 0.45:
            self.model.emit_action(Event(s, "login", self.unique_id, meta={"role": self.role, "after_hours": self.model.random.random() < 0.18}))
        if self.model.random.random() < 0.40:
            self.model.emit_action(Event(s, "db_query", self.unique_id, resource="patients_table", action="SELECT",
                bytes=self.model.random.randint(3000, 28000), meta={"after_hours": self.model.random.random() < 0.12}))
        if s % self.cycle == 0 and s > 0:
            self.in_cycle, self.phase = True, 1
        if self.in_cycle:
            if self.phase == 1:
                self.model.emit_action(Event(s, "data_export", self.unique_id, resource="patients_table", action="EXPORT",
                    bytes=self.model.random.randint(60000, 180000), meta={"after_hours": False}, label="benign"))
                self.phase = 2
            elif self.phase == 2:
                self.model.emit_action(Event(s, "email_send", self.unique_id, dst="partner0@approved.com", meta={
                    "attachment_kb": self.model.random.randint(220, 2500), "content": "Scheduled partner report attached."}, label="benign"))
                self.in_cycle, self.phase = False, 0


class MaliciousInsider(mesa.Agent):
    def __init__(self, model, scenario, start_step, repeat_every=48):
        super().__init__(model)
        self.scenario, self.start_step, self.repeat_every, self.phase = scenario, start_step, repeat_every, 0
        self.model.register_user(self.unique_id, "staff", False, set())

    def _ah(self, s): 
        h = s % 24
        return h < 7 or h > 19

    def act(self):
        if not getattr(self.model, "attack_enabled", True): 
            return
        s, ah = self.model.steps, self._ah(self.model.steps)
        
        if self.phase == 0:
            if s >= self.start_step: 
                self.phase = 1
            else: 
                return
        
        if self.scenario == "exfil":
            if self.phase == 1:
                self.model.emit_action(Event(s, "db_query", self.unique_id, resource="patients_table", action="SELECT",
                    bytes=self.random.randint(8000, 70000), meta={"after_hours": ah}, label="malicious", scenario=self.scenario))
                if self.random.random() < 0.6: 
                    self.phase = 2
            elif self.phase == 2:
                self.model.emit_action(Event(s, "data_export", self.unique_id, resource="patients_table", action="EXPORT",
                    bytes=self.random.randint(80000, 260000), meta={"after_hours": ah}, label="malicious", scenario=self.scenario))
                self.phase = 3
            elif self.phase == 3:
                self.model.emit_action(Event(s, "email_send", self.unique_id, dst="external@outside",
                    meta={"attachment_kb": self.random.randint(600, 4000), "content": "Confidential patient data. Confirm immediately.", "after_hours": ah},
                    label="malicious", scenario=self.scenario))
                self.phase = 4
        
        elif self.scenario == "stealth":
            if self.phase == 1:
                if self.random.random() < 0.35:
                    self.model.emit_action(Event(s, "db_query", self.unique_id, resource="patients_table", action="SELECT",
                        bytes=self.random.randint(800, 5000), meta={"after_hours": ah}, label="malicious", scenario=self.scenario))
                if self.random.random() < 0.08: 
                    self.phase = 3
            elif self.phase == 3:
                self.model.emit_action(Event(s, "email_send", self.unique_id, dst="external@outside",
                    meta={"attachment_kb": self.random.randint(300, 2500), "content": "Summary attached.", "after_hours": ah},
                    label="malicious", scenario=self.scenario))
                self.phase = 4
        
        elif self.scenario == "acct_takeover":
            if ah and self.random.random() < 0.7:
                self.model.emit_action(Event(s, "login", self.unique_id, meta={"role": "admin", "after_hours": True, "takeover": True, "device_change": True},
                    label="malicious", scenario=self.scenario))
                self.model.emit_action(Event(s, "db_query", self.unique_id, resource="patients_table", action="SELECT",
                    bytes=self.random.randint(1500, 9000), meta={"after_hours": True}, label="malicious", scenario=self.scenario))
        
        elif self.scenario == "staging_exfil":
            if self.phase == 1:
                for _ in range(3):
                    self.model.emit_action(Event(s, "db_query", self.unique_id, resource="patients_table", action="SELECT",
                        bytes=self.random.randint(5000, 45000), meta={"after_hours": ah}, label="malicious", scenario=self.scenario))
                if self.random.random() < 0.55: 
                    self.phase = 2
            elif self.phase == 2:
                self.model.emit_action(Event(s, "data_export", self.unique_id, resource="patients_table", action="EXPORT_INTERNAL",
                    bytes=self.random.randint(60000, 220000), meta={"after_hours": ah, "destination": "internal_staging"},
                    label="malicious", scenario=self.scenario))
                self.phase = 3
            elif self.phase == 3:
                self.model.emit_action(Event(s, "email_send", self.unique_id, dst="external@outside",
                    meta={"attachment_kb": self.random.randint(600, 4500), "content": "Private data. Do not share.", "after_hours": ah},
                    label="malicious", scenario=self.scenario))
                self.phase = 4
        
        elif self.scenario == "email_only":
            if self.random.random() < 0.2:
                self.model.emit_action(Event(s, "email_send", self.unique_id, dst="external@outside",
                    meta={"attachment_kb": self.random.randint(250, 2500), "content": "Confidential docs. Act now!", "after_hours": ah},
                    label="malicious", scenario=self.scenario))
        
        if self.phase == 4 and self.repeat_every and (s - self.start_step) % self.repeat_every == 0 and s > self.start_step:
            self.phase = 1


# ============================================================================
# Monitors
# ============================================================================

class DBMonitor(mesa.Agent):
    def __init__(self, model): 
        super().__init__(model)
    
    def monitor(self):
        for e in self.model.action_events:
            if e.event_type in {"db_query", "data_export"}:
                self.model.emit_ingest(Event(self.model.steps, "siem_ingest", e.actor_id, 
                    meta={"source": "db", "record": e.to_dict()}, label=e.label, scenario=e.scenario))


class EmailMonitor(mesa.Agent):
    """Email monitor using combined Enron-trained forensics"""
    
    def __init__(self, model, forensics_agent: CombinedForensicsAgent = None):
        super().__init__(model)
        self.forensics = forensics_agent or CombinedForensicsAgent()
    
    def monitor(self):
        for e in self.model.action_events:
            if e.event_type == "email_send":
                fr = {}
                if getattr(self.model, 'siem_cfg', None) and self.model.siem_cfg.use_forensics:
                    fr = self.forensics.analyze_email(e.to_dict(), self.model.steps)
                rec = e.to_dict()
                rec["forensics"] = fr
                self.model.emit_ingest(Event(self.model.steps, "siem_ingest", e.actor_id, 
                    meta={"source": "email", "record": rec, "forensics": fr}, label=e.label, scenario=e.scenario))


class AuthMonitor(mesa.Agent):
    def __init__(self, model): 
        super().__init__(model)
    
    def monitor(self):
        for e in self.model.action_events:
            if e.event_type == "login":
                self.model.emit_ingest(Event(self.model.steps, "siem_ingest", e.actor_id, 
                    meta={"source": "auth", "record": e.to_dict()}, label=e.label, scenario=e.scenario))


# ============================================================================
# SIEM Agent
# ============================================================================

class SIEMAgent(mesa.Agent):
    def __init__(self, model, cfg: SIEMConfig):
        super().__init__(model)
        self.cfg = cfg
        self.recent = defaultdict(list)
        self.trust = defaultdict(lambda: cfg.trust_init)
        self.export_history = defaultdict(list)
        self.peer_export = defaultdict(lambda: {"count": 0, "bytes": []})
        self.last_alert_step = {}
        self.export_intervals = defaultdict(list)
        self.tom = TomAbdAgent(self.unique_id) if cfg.use_tom else None
        self.forensics_cache = defaultdict(list)

    def _is_external(self, dst):
        return dst and "outside" in dst.lower()

    def _is_email_anchor(self, r):
        return r.get("event_type") == "email_send" and self._is_external(r.get("dst", "")) and (r.get("meta") or {}).get("attachment_kb", 0) > 200

    def _is_login_anchor(self, r):
        if r.get("event_type") != "login": 
            return False
        m = r.get("meta") or {}
        return (m.get("after_hours") and m.get("role") == "admin") or m.get("takeover") or m.get("device_change")

    def _score(self, f):
        return sum(self.cfg.w.get(k, 0) * v for k, v in f.items())

    def confirmed_threshold(self, u):
        role = self.model.user_profile.get(u, {}).get("role", "staff")
        adj = self.cfg.role_adj.get(role, 0)
        return self.cfg.base_confirmed_threshold + self.cfg.trust_slope * (self.trust[u] - 0.5) + adj

    def calc_evidence(self, feats, u):
        c, w = 0, 0.0
        if feats.get("export_large", 0) > 0 or feats.get("staging_export", 0) > 0:
            c += 1; w += 1.5
        if feats.get("unapproved", 0) > 0:
            c += 1; w += 2.0
        if feats.get("exfil_chain", 0) > 0:
            c += 1; w += 2.5
        if feats.get("tom_intent", 0) > 0.4:
            c += 1; w += 2.0
        if feats.get("forensics_phishing", 0) > 0.5:
            c += 1; w += 1.5
        if feats.get("after_hours", 0) > 0 and not self.model.user_profile.get(u, {}).get("allowed_after_hours"):
            c += 1; w += 1.0
        if feats.get("email_burst", 0) > 0 and feats.get("unapproved", 0) > 0:
            c += 1; w += 1.5
        if feats.get("forensics_phishing", 0) > 0.3 and feats.get("unapproved", 0) > 0:
            c += 1; w += 2.0
        if feats.get("ext_email_count", 0) >= 2:
            c += 1; w += 1.5
        if feats.get("stealth_pattern", 0) > 0 and feats.get("unapproved", 0) > 0:
            c += 1; w += 2.5
        return c, w

    def correlate(self):
        s = self.model.steps
        
        for ing in self.model.ingest_events:
            u = ing.actor_id
            rec = (ing.meta or {}).get("record", {})
            if not rec: 
                continue
            
            self.recent[u].append(rec)
            self.recent[u] = self.recent[u][-500:]
            
            if rec.get("event_type") == "data_export":
                self.export_history[u].append(s)
                self.export_history[u] = self.export_history[u][-50:]
                role = self.model.user_profile.get(u, {}).get("role", "staff")
                self.peer_export[role]["count"] += 1
                self.peer_export[role]["bytes"].append(rec.get("bytes", 0))
                self.peer_export[role]["bytes"] = self.peer_export[role]["bytes"][-200:]
                if len(self.export_history[u]) >= 2:
                    self.export_intervals[u].append(s - self.export_history[u][-2])
            
            if self.cfg.use_tom and self.tom:
                self.tom.observe_action(s, u, rec)
            
            if self.cfg.use_forensics and (ing.meta or {}).get("forensics"):
                self.forensics_cache[u].append((ing.meta or {}).get("forensics"))
                self.forensics_cache[u] = self.forensics_cache[u][-30:]
        
        for ing in self.model.ingest_events:
            u = ing.actor_id
            rec = (ing.meta or {}).get("record", {})
            if not rec: 
                continue
            
            if not (self._is_email_anchor(rec) or self._is_login_anchor(rec)):
                continue
            
            recent = self.recent.get(u, [])[-self.cfg.window:]
            reasons = []
            f = self.extract_features(recent, u, s)
            f["anchor_email"] = 1.0 if self._is_email_anchor(rec) else 0
            f["anchor_login"] = 1.0 if self._is_login_anchor(rec) else 0
            
            score = self._score(f)
            mult, off = self.apply_policy(u, recent, f, reasons) if self.cfg.use_policy else (1.0, 0)
            score = (score + off) * mult
            
            if self.cfg.use_tom and self.tom and f.get("tom_intent", 0) > 0:
                score += self.cfg.tom_weight * f["tom_intent"]
                reasons.append(f"tom:{f['tom_intent']:.2f}")
            
            if self.cfg.use_forensics and f.get("forensics_phishing", 0) > 0:
                score += self.cfg.forensics_weight * f["forensics_phishing"]
                reasons.append(f"forensics:{f['forensics_phishing']:.2f}")
            
            early_thr = self.cfg.early_threshold
            conf_thr = self.confirmed_threshold(u)
            
            if score >= early_thr and self._can_fire(u, "early", s):
                self.model.emit_alert(Event(s, "alert_early", u, 
                    meta={"score": score, "threshold": early_thr, "reasons": reasons}, 
                    label=rec.get("label"), scenario=rec.get("scenario")))
                self.last_alert_step[(u, "early")] = s
            
            if self.cfg.use_evidence_gate:
                ev_c, ev_w = self.calc_evidence(f, u)
                if ev_c < self.cfg.min_evidence_count or ev_w < self.cfg.min_evidence_weight:
                    continue
            
            if score >= conf_thr and self._can_fire(u, "confirmed", s):
                self.model.emit_alert(Event(s, "alert_confirmed", u, 
                    meta={"score": score, "threshold": conf_thr, "reasons": reasons}, 
                    label=rec.get("label"), scenario=rec.get("scenario")))
                self.last_alert_step[(u, "confirmed")] = s
                self._update_trust(u, rec.get("label") == "malicious")

    def extract_features(self, recent, u, s):
        f = {k: 0.0 for k in self.cfg.w}
        
        logins = sum(1 for r in recent if r.get("event_type") == "login")
        sens = sum(1 for r in recent if r.get("event_type") == "db_query" and r.get("resource") == "patients_table")
        ext_emails = sum(1 for r in recent if r.get("event_type") == "email_send" and self._is_external(r.get("dst", "")) and (r.get("meta") or {}).get("attachment_kb", 0) > 200)
        exports = sum(1 for r in recent if r.get("event_type") == "data_export")
        
        f["login_burst"] = 1.0 if logins >= 4 else 0
        f["sens_burst"] = 1.0 if sens >= 6 else 0
        f["email_burst"] = 1.0 if ext_emails >= 3 else 0
        f["ext_email_count"] = ext_emails
        
        # Stealth pattern
        if sens >= 1 and ext_emails >= 1 and exports == 0:
            f["stealth_pattern"] = 1.0
        
        f["after_hours"] = 1.0 if any((r.get("meta") or {}).get("after_hours") for r in recent) else 0
        
        exp_bytes = [r.get("bytes", 0) for r in recent if r.get("event_type") == "data_export" and r.get("resource") == "patients_table" and r.get("action") == "EXPORT"]
        if exp_bytes:
            mx = max(exp_bytes)
            f["export_large"] = 1.0 if mx >= 120000 else 0
            f["export_small"] = 1.0 if mx < 120000 else 0
        
        f["staging_export"] = 1.0 if any(r.get("event_type") == "data_export" and r.get("action") in {"EXPORT_INTERNAL", "EXPORT_STAGING"} for r in recent) else 0
        
        # Exfil chain
        for i in range(len(recent) - 2):
            if (recent[i].get("event_type") == "db_query" and recent[i].get("resource") == "patients_table" and
                recent[i+1].get("event_type") == "data_export" and
                recent[i+2].get("event_type") == "email_send" and self._is_external(recent[i+2].get("dst", ""))):
                if recent[i+2].get("step", 0) - recent[i].get("step", 0) <= 5:
                    f["exfil_chain"] = 1.0
                    break
        
        # ToM
        if self.cfg.use_tom and self.tom:
            ta = self.tom.get_threat_assessment(u)
            if ta["tom_intent_score"] > self.cfg.tom_threshold:
                f["tom_intent"] = ta["tom_intent_score"]
            if ta["tom_plan_score"] > 0.4:
                f["tom_plan"] = ta["tom_plan_score"]
        
        # Forensics
        if self.cfg.use_forensics:
            fc = self.forensics_cache.get(u, [])[-10:]
            if fc:
                avg_phishing = np.mean([x.get("forensics_phishing_score", 0) for x in fc])
                if avg_phishing > 0.3:
                    f["forensics_phishing"] = avg_phishing
        
        return f

    def apply_policy(self, u, recent, f, reasons):
        mult, off = 1.0, 0.0
        profile = self.model.user_profile.get(u, {})
        approved = profile.get("approved_partners", set())
        
        for r in recent:
            if r.get("event_type") == "email_send" and self._is_external(r.get("dst", "")):
                dst = r.get("dst", "")
                if dst not in approved:
                    f["unapproved"] = 1.0
                    reasons.append("unapproved_partner")
                    off += 1.5
        
        return mult, off

    def _can_fire(self, u, level, s):
        last = self.last_alert_step.get((u, level), -999)
        return s - last >= self.cfg.cooldown

    def _update_trust(self, u, was_malicious):
        delta = self.cfg.trust_tp_delta if was_malicious else self.cfg.trust_fp_delta
        self.trust[u] = max(self.cfg.trust_min, min(self.cfg.trust_max, self.trust[u] + delta))


# ============================================================================
# Model
# ============================================================================

class InsiderModel(mesa.Model):
    def __init__(self, n_emp=30, n_power=4, n_exfil=3, n_stealth=2, n_takeover=1, 
                 n_staging=1, n_email=1, seed=42, siem_cfg=None, warmup=60,
                 forensics_model_path=None):
        super().__init__(seed=seed)
        
        self.steps = 0
        self.action_events, self.ingest_events, self.alert_events = [], [], []
        self.event_log = []
        self.user_profile: Dict[int, Dict] = {}
        self.siem_cfg = siem_cfg or SIEMConfig()
        self.warmup_steps = warmup
        self.phase = "train"
        self.attack_enabled = False
        
        # Load forensics model
        forensics_agent = CombinedForensicsAgent(forensics_model_path)
        
        # Create agents
        for _ in range(n_emp):
            BenignEmployee(self, self.random.choices(["staff", "analyst", "admin"], [0.78, 0.18, 0.04])[0])
        for _ in range(n_power):
            BenignPowerUser(self, "analyst", self.random.choice([24, 36, 48]))
        
        base = 16 + warmup
        for i in range(n_exfil):
            MaliciousInsider(self, "exfil", base + i*12, 48)
        for j in range(n_stealth):
            MaliciousInsider(self, "stealth", 28 + warmup + j*18, 48)
        for k in range(n_takeover):
            MaliciousInsider(self, "acct_takeover", 22 + warmup + k*20, None)
        for k in range(n_staging):
            MaliciousInsider(self, "staging_exfil", 34 + warmup + k*24, 72)
        for k in range(n_email):
            MaliciousInsider(self, "email_only", 18 + warmup + k*26, None)
        
        # Create monitors with forensics
        DBMonitor(self)
        EmailMonitor(self, forensics_agent)
        AuthMonitor(self)
        SIEMAgent(self, self.siem_cfg)

    def register_user(self, u, role, ah, approved):
        self.user_profile.setdefault(u, {}).update({
            "role": role, "allowed_after_hours": ah, "approved_partners": set(approved)})

    def emit_action(self, e):
        e.phase = self.phase
        self.action_events.append(e)
        self.event_log.append(e)

    def emit_ingest(self, e):
        e.phase = self.phase
        self.ingest_events.append(e)
        self.event_log.append(e)

    def emit_alert(self, e):
        e.phase = self.phase
        self.alert_events.append(e)
        self.event_log.append(e)

    def step(self):
        self.steps += 1
        if self.steps >= self.warmup_steps:
            self.phase = "test"
            self.attack_enabled = True
        
        self.action_events, self.ingest_events, self.alert_events = [], [], []
        
        for a in self.agents:
            if hasattr(a, 'act'): a.act()
        for a in self.agents:
            if hasattr(a, 'monitor'): a.monitor()
        for a in self.agents:
            if isinstance(a, SIEMAgent): a.correlate()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(event_log, warmup):
    test = [e for e in event_log if e.phase == "test"]
    mal_actors, det_actors, ttd_map, first_mal = set(), set(), {}, {}
    
    conf = [e for e in test if e.event_type == "alert_confirmed"]
    conf_tp = sum(1 for a in conf if a.label == "malicious")
    conf_fp = sum(1 for a in conf if a.label != "malicious")
    
    for e in test:
        if e.label == "malicious" and e.event_type not in {"alert_early", "alert_confirmed"}:
            mal_actors.add(e.actor_id)
            if e.actor_id not in first_mal:
                first_mal[e.actor_id] = e.step
    
    for a in conf:
        if a.label == "malicious":
            det_actors.add(a.actor_id)
            if a.actor_id not in ttd_map and a.actor_id in first_mal:
                ttd_map[a.actor_id] = a.step - first_mal[a.actor_id]
    
    n_mal, n_det = len(mal_actors), len(det_actors)
    p = n_det / (n_det + conf_fp) if (n_det + conf_fp) else 0
    r = n_det / n_mal if n_mal else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    ttd_vals = list(ttd_map.values())
    
    tom_det = sum(1 for a in conf if "tom:" in str(a.meta.get("reasons", [])))
    
    return {"precision": p, "recall": r, "f1": f1, 
            "ttd_avg": np.mean(ttd_vals) if ttd_vals else 0,
            "ttd_max": max(ttd_vals) if ttd_vals else 0,
            "conf_total": len(conf), "conf_prec": conf_tp/len(conf) if conf else 0,
            "conf_fp": conf_fp, "actors_detected": n_det, "actors_total": n_mal,
            "tom_detections": tom_det}


def run_experiment(T=240, warmup=60, runs=10, forensics_model_path=None):
    """Run the EG-SIEM experiment with combined forensics model"""
    
    print("="*70)
    print("EG-SIEM WITH COMBINED ENRON-TRAINED FORENSICS")
    print("="*70)
    
    if forensics_model_path:
        print(f"Forensics model: {forensics_model_path}")
    else:
        print("Forensics model: None (using keyword-only detection)")
    
    print(f"Runs: {runs}, Steps: {T}, Warmup: {warmup}")
    print()
    
    cfg = SIEMConfig(
        use_policy=True, use_baseline=True, use_trust=True, use_ml=True,
        use_tom=True, use_forensics=True, use_evidence_gate=True,
        use_peer_norm=True, use_regularity=True)
    
    results = []
    
    for r in range(runs):
        model = InsiderModel(seed=42+r, siem_cfg=cfg, warmup=warmup,
                            forensics_model_path=forensics_model_path)
        for _ in range(T):
            model.step()
        metrics = evaluate(model.event_log, warmup)
        results.append(metrics)
        
        print(f"Run {r+1}: F1={metrics['f1']:.3f} (P={metrics['precision']:.3f}, R={metrics['recall']:.3f}), "
              f"Confirmed={metrics['conf_total']}, FP={metrics['conf_fp']}, ToM={metrics['tom_detections']}")
    
    # Calculate averages
    print()
    print("="*70)
    print("AVERAGED RESULTS")
    print("="*70)
    
    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    std = {k: np.std([r[k] for r in results]) for k in results[0]}
    
    print(f"""
Actor-level Performance:
  Precision:  {avg['precision']:.4f} ± {std['precision']:.4f}
  Recall:     {avg['recall']:.4f} ± {std['recall']:.4f}
  F1:         {avg['f1']:.4f} ± {std['f1']:.4f}

Time-to-Detection:
  Average:    {avg['ttd_avg']:.2f} steps
  Maximum:    {avg['ttd_max']:.2f} steps

Alert Statistics:
  Confirmed alerts/run:  {avg['conf_total']:.1f}
  Confirmed precision:   {avg['conf_prec']:.4f}
  False positives/run:   {avg['conf_fp']:.1f}
  ToM-assisted/run:      {avg['tom_detections']:.1f}

Actor Detection:
  Detected: {avg['actors_detected']:.1f}/{avg['actors_total']:.1f}
""")
    
    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EG-SIEM with Combined Enron Forensics")
    parser.add_argument('--model', type=str, default='combined_forensics_model.pkl',
                       help='Path to trained forensics model (default: combined_forensics_model.pkl)')
    parser.add_argument('--runs', type=int, default=10, help='Number of simulation runs')
    parser.add_argument('--steps', type=int, default=240, help='Steps per run')
    parser.add_argument('--warmup', type=int, default=60, help='Warmup steps')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = args.model if os.path.exists(args.model) else None
    
    print()
    print("╔" + "═"*68 + "╗")
    print("║" + " INSIDER THREAT DETECTION - EG-SIEM + ENRON FORENSICS ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    print()
    
    run_experiment(T=args.steps, warmup=args.warmup, runs=args.runs, 
                   forensics_model_path=model_path)
