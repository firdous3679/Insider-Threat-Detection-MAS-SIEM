# mini_mesa_V5.py - Performance Enhanced Insider Threat Detection
# Enhancements: Evidence accumulation, Peer normalization, Regularity detection,
# Contradicting evidence in ToM, Role-based thresholds

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Tuple, Set
from collections import Counter, defaultdict
from enum import Enum
import json, math, re
import numpy as np
import mesa
from mesa.datacollection import DataCollector

try:
    from sklearn.ensemble import IsolationForest
except:
    IsolationForest = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    HAS_NLP = True
except:
    HAS_NLP = False

# ============ TomAbd Framework ============
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
        meta = action.get("meta", {}) or {}
        if meta.get("ticket_id"): contras.append("has_ticket")
        if meta.get("approved_partner"): contras.append("approved_partner")
        if meta.get("cycle") == "partner_reporting": contras.append("scheduled_cycle")
        if not meta.get("after_hours"): contras.append("work_hours")
        return contras

    def _abductive_inference(self, step: int, actor_id: int):
        recent = [c for _, c in self.observed_actions[actor_id][-20:]]
        for pattern, goal_type, boost in self.ABDUCTIVE_RULES:
            if self._pattern_matches(recent, pattern):
                if goal_type not in self.inferred_goals[actor_id]:
                    self.inferred_goals[actor_id][goal_type] = InferredGoal(goal_type=goal_type, probability=boost, supporting_evidence=[str(pattern)], last_updated=step)
                else:
                    g = self.inferred_goals[actor_id][goal_type]
                    g.probability = min(0.95, g.probability + boost * (1 - g.probability))
                    g.last_updated = step
        for gt, g in list(self.inferred_goals[actor_id].items()):
            age = step - g.last_updated
            if age > 20: g.probability *= 0.92 ** (age - 20)
            if g.probability < 0.08: del self.inferred_goals[actor_id][gt]

    def _pattern_matches(self, actions: List[str], pattern: List[str]) -> bool:
        if len(pattern) > len(actions): return False
        idx = 0
        for a in actions:
            if idx >= len(pattern): return True
            p = pattern[idx]
            if a.startswith(p.split(":")[0]) and (":" not in p or p.split(":")[1] in a):
                idx += 1
        return idx >= len(pattern)

    def _update_plans(self, step: int, actor_id: int, code: str):
        if not self.plan_hypotheses[actor_id]:
            for pid, goal, exp in self.PLAN_TEMPLATES.values():
                self.plan_hypotheses[actor_id].append(Plan(pid, goal, exp.copy(), [], 0.0, 0.1))
        for p in self.plan_hypotheses[actor_id]:
            idx = len(p.observed_actions)
            if idx < len(p.expected_actions):
                exp = p.expected_actions[idx]
                if code.startswith(exp.split(":")[0]) and (":" not in exp or exp.split(":")[1] in code):
                    p.observed_actions.append(code)
                    p.completion_ratio = len(p.observed_actions) / len(p.expected_actions)
                    p.confidence = min(0.9, p.confidence + 0.15)
                else:
                    p.confidence = max(0.05, p.confidence - 0.03)

    def get_threat_assessment(self, actor_id: int) -> Dict[str, Any]:
        result = {"has_malicious_intent": False, "intent_confidence": 0.0, "primary_goal": "none", "contradictions": [], "plan_completion": 0.0}
        goals = self.inferred_goals.get(actor_id, {})
        if goals:
            top = max(goals.values(), key=lambda x: x.probability)
            if top.probability > 0.25:
                result["has_malicious_intent"] = True
                result["intent_confidence"] = top.probability
                result["primary_goal"] = top.goal_type.value
                result["contradictions"] = top.contradicting_evidence[-3:]
        plans = self.plan_hypotheses.get(actor_id, [])
        if plans:
            top_p = max(plans, key=lambda x: x.confidence * x.completion_ratio)
            result["plan_completion"] = top_p.completion_ratio
        return result

    def get_cognitive_features(self, actor_id: int) -> Dict[str, float]:
        a = self.get_threat_assessment(actor_id)
        return {"tom_intent_score": a["intent_confidence"], "tom_plan_completion": a["plan_completion"], "tom_contradictions": len(a["contradictions"])}

# ============ Forensics ============
class EmailForensicsAgent:
    PHISHING_KW = ["urgent", "immediately", "verify", "suspend", "password", "click here", "confirm", "expire", "act now"]
    SENSITIVE_KW = ["confidential", "secret", "private", "patient", "medical", "financial"]

    def __init__(self):
        self.author_profiles: Dict[int, Dict] = defaultdict(lambda: {"avg_len": [], "vocab": [], "count": 0})
        self.word_freq: Counter = Counter()
        self.total_words = 0
        self.classifier = None
        self.vectorizer = None
        if HAS_NLP:
            ph = ["Urgent: Account suspended. Click here.", "Congratulations! Claim prize now.", "Password expires. Update immediately."]
            leg = ["Please find attached the report.", "Meeting at 3pm tomorrow.", "Here are the documents."]
            self.vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
            X = self.vectorizer.fit_transform(ph + leg)
            self.classifier = MultinomialNB()
            self.classifier.fit(X, [1]*len(ph) + [0]*len(leg))

    def analyze_email(self, event: Dict, step: int) -> Dict[str, Any]:
        meta = event.get("meta", {}) or {}
        content = meta.get("content", meta.get("subject", ""))
        actor = event.get("actor_id", 0)
        nlp = self._nlp_analysis(content)
        ai = self._ai_analysis(content, actor)
        self._update_profile(actor, content)
        content_risk = nlp["phishing"] * 0.4 + nlp["urgency"] * 0.3 + (0.2 if nlp["sensitive"] else 0)
        auth_risk = ai["ai_prob"] * 0.5 + ai["deviation"] * 0.3
        attach = meta.get("attachment_kb", 0)
        attach_risk = min(1.0, attach / 5000) if attach > 500 else 0
        return {"forensics_phishing_score": nlp["phishing"], "forensics_urgency": nlp["urgency"],
                "forensics_sensitive": 1.0 if nlp["sensitive"] else 0, "forensics_ai_prob": ai["ai_prob"],
                "forensics_content_risk": content_risk, "forensics_combined_risk": (content_risk + auth_risk + attach_risk) / 3}

    def _nlp_analysis(self, content: str) -> Dict:
        if not content: return {"phishing": 0, "urgency": 0, "sensitive": False}
        cl = content.lower()
        ph_count = sum(1 for k in self.PHISHING_KW if k in cl)
        ph_score = min(1.0, ph_count / 4)
        urg_count = sum(1 for w in ["urgent", "immediately", "asap", "now"] if w in cl)
        urg = min(1.0, urg_count / 2)
        sens = any(k in cl for k in self.SENSITIVE_KW)
        ml_prob = 0
        if self.classifier and content.strip():
            try: ml_prob = self.classifier.predict_proba(self.vectorizer.transform([content]))[0][1]
            except: pass
        return {"phishing": max(ph_score, ml_prob), "urgency": urg, "sensitive": sens}

    def _ai_analysis(self, content: str, actor: int) -> Dict:
        if not content or len(content) < 20: return {"ai_prob": 0, "deviation": 0}
        words = re.findall(r'\b\w+\b', content.lower())
        for w in words: self.word_freq[w] += 1; self.total_words += 1
        avg_p = sum(self.word_freq[w]/max(1,self.total_words) for w in words) / max(len(words),1)
        ppl = min(100, 1/max(avg_p, 0.001))
        ai_prob = 0.3 if ppl < 30 else 0
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        if len(sentences) >= 2:
            cv = np.std([len(s.split()) for s in sentences]) / (np.mean([len(s.split()) for s in sentences]) + 0.1)
            if cv < 0.3: ai_prob += 0.3
        prof = self.author_profiles[actor]
        dev = 0
        if prof["count"] >= 3 and sentences:
            curr_len = np.mean([len(s.split()) for s in sentences])
            curr_voc = len(set(words)) / max(len(words), 1)
            if prof["avg_len"]: dev += abs(curr_len - np.mean(prof["avg_len"][-10:])) / max(np.mean(prof["avg_len"][-10:]), 1)
            if prof["vocab"]: dev += abs(curr_voc - np.mean(prof["vocab"][-10:])) / max(np.mean(prof["vocab"][-10:]), 0.1)
            dev /= 2
            if dev > 0.5: ai_prob += 0.2
        return {"ai_prob": min(1, ai_prob), "deviation": dev}

    def _update_profile(self, actor: int, content: str):
        prof = self.author_profiles[actor]
        words = re.findall(r'\b\w+\b', content.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        if sentences: prof["avg_len"].append(np.mean([len(s.split()) for s in sentences]))
        if words: prof["vocab"].append(len(set(words)) / len(words))
        prof["count"] += 1
        prof["avg_len"] = prof["avg_len"][-50:]
        prof["vocab"] = prof["vocab"][-50:]

# ============ Event & Config ============
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

def _clip(x, lo, hi): return max(lo, min(hi, x))

# ============ ML ============
class MLModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.buffers: Dict[str, List] = defaultdict(list)
        self.models: Dict[str, Any] = {}
        self.last_fit: Dict[str, int] = defaultdict(lambda: -10000)
    def extract(self, recent, w):
        if not recent: return [0]*12
        lo = sum(1 for r in recent if r.get("event_type") == "login")
        db = sum(1 for r in recent if r.get("event_type") == "db_query")
        se = sum(1 for r in recent if r.get("event_type") == "db_query" and r.get("resource") == "patients_table")
        ex = sum(1 for r in recent if r.get("event_type") == "data_export")
        em = sum(1 for r in recent if r.get("event_type") == "email_send" and (r.get("dst") or "").endswith("outside"))
        ah = sum(1 for r in recent if (r.get("meta") or {}).get("after_hours"))
        by = sum(r.get("bytes", 0) for r in recent if r.get("event_type") in ["db_query", "data_export"])
        return [lo, db, se, ex, em, ah, se/max(db,1), ex/max(db,1), em/max(sum(1 for r in recent if r.get("event_type")=="email_send"),1), ah/max(len(recent),1), by/10000, 0]
    def add(self, role, vec):
        self.buffers[role].append(vec)
        self.buffers[role] = self.buffers[role][-500:]
    def fit(self, role, step):
        if not IsolationForest or len(self.buffers[role]) < self.cfg.ml_min or step - self.last_fit[role] < self.cfg.ml_refit: return
        clf = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
        clf.fit(self.buffers[role])
        self.models[role] = clf
        self.last_fit[role] = step
    def score(self, role, vec):
        if not IsolationForest or role not in self.models: return 0
        return min(max(0, -float(self.models[role].decision_function([vec])[0])) * 1.5, 5.0)

# ============ Agents ============
class BenignEmployee(mesa.Agent):
    def __init__(self, model, role="staff"):
        super().__init__(model)
        self.role = role
        self.model.register_user(self.unique_id, role, role in {"analyst", "admin"}, set() if role == "staff" else {"partner@outside"})
    def act(self):
        s, h = self.model.steps, self.model.steps % 24
        at_work = 8 <= h <= 17
        if self.random.random() < (0.08 if at_work else 0.01):
            self.model.emit_action(Event(s, "login", self.unique_id, meta={"role": self.role, "after_hours": not at_work}, label="benign"))
        if at_work and self.random.random() < 0.28:
            self.model.emit_action(Event(s, "db_query", self.unique_id, resource=self.random.choice(["public_table", "patients_table"]),
                action="SELECT", bytes=self.random.randint(200, 6000), meta={"role": self.role, "after_hours": False}, label="benign"))
        if at_work and self.random.random() < 0.08:
            self.model.emit_action(Event(s, "email_send", self.unique_id, dst="someone@org",
                meta={"attachment_kb": 0, "content": "Following up. Thanks", "role": self.role}, label="benign"))

class BenignPowerUser(mesa.Agent):
    def __init__(self, model, role="analyst", report_every=36):
        super().__init__(model)
        self.role, self.report_every = role, report_every
        self.in_cycle, self.phase, self.ticket = False, 0, None
        self.model.register_user(self.unique_id, role, True, {"partner@outside"})
    def act(self):
        s, h = self.model.steps, self.model.steps % 24
        at_work = 8 <= h <= 17
        if at_work and self.random.random() < 0.22:
            self.model.emit_action(Event(s, "db_query", self.unique_id, resource="patients_table", action="SELECT",
                bytes=self.random.randint(2000, 16000), meta={"role": self.role, "after_hours": False}, label="benign"))
        if s % self.report_every == 0 and not self.in_cycle:
            self.in_cycle, self.phase, self.ticket = True, 1, f"TCKT-{self.unique_id*1000+s:05d}"
        if self.in_cycle:
            if self.phase == 1:
                self.model.emit_action(Event(s, "db_query", self.unique_id, resource="patients_table", action="SELECT",
                    bytes=self.random.randint(3000, 25000), meta={"role": self.role, "cycle": "partner_reporting", "ticket_id": self.ticket}, label="benign"))
                self.phase = 2
            elif self.phase == 2:
                self.model.emit_action(Event(s, "data_export", self.unique_id, resource="patients_table", action="EXPORT",
                    bytes=self.random.randint(25000, 180000), meta={"role": self.role, "cycle": "partner_reporting", "ticket_id": self.ticket}, label="benign"))
                self.phase = 3
            elif self.phase == 3:
                self.model.emit_action(Event(s, "email_send", self.unique_id, dst="partner@outside",
                    meta={"role": self.role, "cycle": "partner_reporting", "ticket_id": self.ticket, "approved_partner": True,
                          "attachment_kb": self.random.randint(220, 2500), "content": "Scheduled partner report."}, label="benign"))
                self.in_cycle, self.phase = False, 0

class MaliciousInsider(mesa.Agent):
    def __init__(self, model, scenario, start_step, repeat_every=48):
        super().__init__(model)
        self.scenario, self.start_step, self.repeat_every, self.phase = scenario, start_step, repeat_every, 0
        self.model.register_user(self.unique_id, "staff", False, set())
    def _ah(self, s): h = s % 24; return h < 7 or h > 19
    def act(self):
        if not getattr(self.model, "attack_enabled", True): return
        s, ah = self.model.steps, self._ah(self.model.steps)
        if self.phase == 0:
            if s >= self.start_step: self.phase = 1
            else: return
        if self.scenario == "exfil":
            if self.phase == 1:
                self.model.emit_action(Event(s, "db_query", self.unique_id, resource="patients_table", action="SELECT",
                    bytes=self.random.randint(8000, 70000), meta={"after_hours": ah}, label="malicious", scenario=self.scenario))
                if self.random.random() < 0.6: self.phase = 2
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
                if self.random.random() < 0.08: self.phase = 3
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
                if self.random.random() < 0.55: self.phase = 2
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

# ============ Monitors ============
class DBMonitor(mesa.Agent):
    def __init__(self, model): super().__init__(model)
    def monitor(self):
        for e in self.model.action_events:
            if e.event_type in {"db_query", "data_export"}:
                self.model.emit_ingest(Event(self.model.steps, "siem_ingest", e.actor_id, meta={"source": "db", "record": e.to_dict()}, label=e.label, scenario=e.scenario))

class EmailMonitor(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.forensics = EmailForensicsAgent()
    def monitor(self):
        for e in self.model.action_events:
            if e.event_type == "email_send":
                fr = self.forensics.analyze_email(e.to_dict(), self.model.steps) if getattr(self.model, 'siem_cfg', None) and self.model.siem_cfg.use_forensics else {}
                rec = e.to_dict()
                rec["forensics"] = fr
                self.model.emit_ingest(Event(self.model.steps, "siem_ingest", e.actor_id, meta={"source": "email", "record": rec, "forensics": fr}, label=e.label, scenario=e.scenario))

class AuthMonitor(mesa.Agent):
    def __init__(self, model): super().__init__(model)
    def monitor(self):
        for e in self.model.action_events:
            if e.event_type == "login":
                self.model.emit_ingest(Event(self.model.steps, "siem_ingest", e.actor_id, meta={"source": "auth", "record": e.to_dict()}, label=e.label, scenario=e.scenario))

# ============ Enhanced SIEM ============
class SIEMAgent(mesa.Agent):
    def __init__(self, model, cfg: SIEMConfig):
        super().__init__(model)
        self.cfg = cfg
        self.last_alert: Dict[Tuple[int, str], int] = {}
        self.recent: Dict[int, List[Dict]] = {}
        self.mu: Dict[int, Dict[str, float]] = {}
        self.var: Dict[int, Dict[str, float]] = {}
        self.trust: Dict[int, float] = {}
        self.ml = MLModel(cfg)
        self.tom: Dict[int, TomAbdAgent] = {}
        self.forensics_cache: Dict[int, List[Dict]] = defaultdict(list)
        self.export_hist: Dict[int, List[int]] = defaultdict(list)

    def _get_tom(self, u): 
        if u not in self.tom: self.tom[u] = TomAbdAgent(u)
        return self.tom[u]

    def _is_ext(self, d): return (d or "").lower().endswith("outside")
    def _is_email_anchor(self, r): return r.get("event_type") == "email_send" and self._is_ext(r.get("dst", "")) and (r.get("meta") or {}).get("attachment_kb", 0) > 200
    def _is_login_anchor(self, r):
        if r.get("event_type") != "login": return False
        m = r.get("meta") or {}
        return (m.get("after_hours") and m.get("role") == "admin") or m.get("takeover")

    def calc_evidence(self, feats, u):
        c, w = 0, 0.0
        if feats.get("export_large", 0) > 0 or feats.get("staging_export", 0) > 0: c += 1; w += 1.5
        if feats.get("unapproved", 0) > 0: c += 1; w += 2.0
        if feats.get("exfil_chain", 0) > 0: c += 1; w += 2.5
        if feats.get("tom_intent", 0) > 0.4: c += 1; w += 2.0
        if feats.get("forensics_phishing", 0) > 0.5: c += 1; w += 1.5
        if feats.get("after_hours", 0) > 0 and not self.model.user_profile.get(u, {}).get("allowed_after_hours", False): c += 1; w += 1.0
        # Detect external emails to unapproved partners (email-only exfil)
        if feats.get("email_burst", 0) > 0 and feats.get("unapproved", 0) > 0: c += 1; w += 1.5
        # Detect even single external email with high-risk content (phishing + unapproved)
        if feats.get("forensics_phishing", 0) > 0.3 and feats.get("unapproved", 0) > 0: c += 1; w += 2.0
        # Count external emails directly for email-only attackers
        if feats.get("ext_email_count", 0) >= 2: c += 1; w += 1.5
        # NEW: Stealth pattern - queries + external email without export
        if feats.get("stealth_pattern", 0) > 0 and feats.get("unapproved", 0) > 0: c += 1; w += 2.5
        return c, w

    def peer_deviation(self, u, recent):
        if not self.cfg.use_peer_norm: return 0
        role = self.model.user_profile.get(u, {}).get("role", "staff")
        peers = [uid for uid, p in self.model.user_profile.items() if p.get("role") == role and uid != u]
        if len(peers) < 3: return 0
        my_exp = sum(1 for r in recent if r.get("event_type") == "data_export")
        peer_exp = [sum(1 for r in self.recent.get(p, [])[-48:] if r.get("event_type") == "data_export") for p in peers]
        z = (my_exp - np.mean(peer_exp)) / (np.std(peer_exp) + 0.1)
        return max(0, z - 2.0)

    def regularity(self, u):
        if not self.cfg.use_regularity: return 0.5
        exp = self.export_hist.get(u, [])
        if len(exp) < 4: return 0.5
        intervals = [exp[i+1] - exp[i] for i in range(len(exp)-1)]
        cv = np.std(intervals) / (np.mean(intervals) + 0.1)
        return 1.0 - min(cv / 1.5, 1.0)

    def extract_features(self, recent, u):
        f = {k: 0.0 for k in self.cfg.w}
        lo = sum(1 for r in recent if r.get("event_type") == "login")
        se = sum(1 for r in recent if r.get("event_type") == "db_query" and r.get("resource") == "patients_table")
        ext = sum(1 for r in recent if r.get("event_type") == "email_send" and self._is_ext(r.get("dst", "")) and (r.get("meta") or {}).get("attachment_kb", 0) > 200)
        exports = sum(1 for r in recent if r.get("event_type") == "data_export")
        f["login_burst"] = 1.0 if lo >= 3 else 0
        f["sens_burst"] = 1.0 if se >= 6 else 0
        f["email_burst"] = 1.0 if ext >= 3 else 0
        f["ext_email_count"] = ext  # Store actual count for evidence calculation
        # NEW: Detect stealth pattern - sensitive queries + external email but NO export
        if se >= 1 and ext >= 1 and exports == 0:
            f["stealth_pattern"] = 1.0
        f["after_hours"] = 1.0 if any((r.get("meta") or {}).get("after_hours") for r in recent) else 0
        exp_b = [r.get("bytes", 0) for r in recent if r.get("event_type") == "data_export" and r.get("resource") == "patients_table" and r.get("action") == "EXPORT"]
        if exp_b:
            mx = max(exp_b)
            f["export_large"] = 1.0 if mx >= 120000 else 0
            f["export_small"] = 1.0 if mx < 120000 else 0
        f["staging_export"] = 1.0 if any(r.get("event_type") == "data_export" and r.get("action") in {"EXPORT_INTERNAL", "EXPORT_STAGING"} for r in recent) else 0
        for i in range(len(recent)-2):
            if recent[i].get("event_type") == "db_query" and recent[i].get("resource") == "patients_table" and recent[i+1].get("event_type") == "data_export" and recent[i+2].get("event_type") == "email_send" and self._is_ext(recent[i+2].get("dst", "")):
                if recent[i+2].get("step", 0) - recent[i].get("step", 0) <= 5:
                    f["exfil_chain"] = 1.0
                    break
        if self.cfg.use_tom:
            tf = self._get_tom(u).get_cognitive_features(u)
            if tf["tom_intent_score"] > self.cfg.tom_threshold: f["tom_intent"] = tf["tom_intent_score"]
            if tf["tom_plan_completion"] > 0.4: f["tom_plan"] = tf["tom_plan_completion"]
        if self.cfg.use_forensics:
            fc = self.forensics_cache.get(u, [])[-10:]
            if fc:
                avg_ph = np.mean([x.get("forensics_phishing_score", 0) for x in fc])
                if avg_ph > 0.4: f["forensics_phishing"] = avg_ph
        f["peer_dev"] = self.peer_deviation(u, recent)
        reg = self.regularity(u)
        if reg < 0.3: f["irreg"] = 1.0 - reg
        return f

    def _score(self, f): return sum(self.cfg.w.get(k, 0) * f.get(k, 0) for k in f)

    def apply_policy(self, u, recent, f, reasons):
        mult, off = 1.0, 0.0
        in_cyc = any((r.get("meta") or {}).get("cycle") == "partner_reporting" for r in recent)
        if in_cyc and any((r.get("meta") or {}).get("ticket_id") for r in recent):
            mult *= 0.45; off -= 1.5; reasons.append("policy:cycle_ticket")
            if any((r.get("meta") or {}).get("approved_partner") for r in recent if r.get("event_type") == "email_send"):
                mult *= 0.35; off -= 1.0; reasons.append("policy:full_compliance")
        approved = self.model.user_profile.get(u, {}).get("approved_partners", set()) or set()
        for r in recent:
            if r.get("event_type") == "email_send" and self._is_ext(r.get("dst", "")):
                if r.get("dst", "") not in approved and not (r.get("meta") or {}).get("approved_partner"):
                    f["unapproved"] = 1.0; reasons.append("policy:unapproved"); break
        return mult, off

    def baseline(self, u, f, reasons):
        keys = ["export_large", "export_small", "staging_export", "after_hours", "sens_burst", "login_burst"]
        self.mu.setdefault(u, {k: 0 for k in keys})
        self.var.setdefault(u, {k: 1 for k in keys})
        zsum = 0
        for k in keys:
            x, m, v = f.get(k, 0), self.mu[u][k], self.var[u][k]
            z = max(0, min(self.cfg.z_clip, (x - m) / (math.sqrt(v) + 1e-6)))
            zsum += z
            nm = (1-self.cfg.ewma_alpha)*m + self.cfg.ewma_alpha*x
            self.mu[u][k] = nm
            self.var[u][k] = (1-self.cfg.ewma_alpha)*v + self.cfg.ewma_alpha*(x-nm)**2
        if zsum > 0: reasons.append(f"baseline:{zsum:.1f}")
        return zsum * self.cfg.baseline_weight

    def conf_threshold(self, u):
        base = self.cfg.base_confirmed_threshold
        role = self.model.user_profile.get(u, {}).get("role", "staff")
        role_adj = self.cfg.role_adj.get(role, 0)
        t = self.trust.get(u, self.cfg.trust_init)
        return base + role_adj + self.cfg.trust_slope * (t - 0.5)

    def update_trust(self, u, is_mal):
        t = self.trust.get(u, self.cfg.trust_init)
        t += self.cfg.trust_tp_delta if is_mal else self.cfg.trust_fp_delta
        self.trust[u] = _clip(t, self.cfg.trust_min, self.cfg.trust_max)

    def _can_fire(self, u, tier, s): return s - self.last_alert.get((u, tier), -10000) >= self.cfg.cooldown

    def correlate(self):
        s = self.model.steps
        for ing in self.model.ingest_events:
            u = ing.actor_id
            rec = (ing.meta or {}).get("record", {})
            if not rec: continue
            self.recent.setdefault(u, []).append(rec)
            self.recent[u] = self.recent[u][-2000:]
            if rec.get("event_type") == "data_export":
                self.export_hist[u].append(s)
                self.export_hist[u] = self.export_hist[u][-50:]
            if self.cfg.use_tom: self._get_tom(u).observe_action(s, u, rec)
            if self.cfg.use_forensics and (ing.meta or {}).get("forensics"):
                self.forensics_cache[u].append((ing.meta or {}).get("forensics"))
                self.forensics_cache[u] = self.forensics_cache[u][-50:]
        if self.cfg.use_ml and s % 2 == 0:
            for u, hist in self.recent.items():
                if hist:
                    role = self.model.user_profile.get(u, {}).get("role", "staff")
                    self.ml.add(role, self.ml.extract(hist[-self.cfg.window:], self.cfg.window))
                    self.ml.fit(role, s)
        for ing in self.model.ingest_events:
            u = ing.actor_id
            rec = (ing.meta or {}).get("record", {})
            if not rec or not (self._is_email_anchor(rec) or self._is_login_anchor(rec)): continue
            recent = self.recent.get(u, [])[-self.cfg.window:]
            reasons = []
            f = self.extract_features(recent, u)
            f["anchor_email"] = 1.0 if self._is_email_anchor(rec) else 0
            f["anchor_login"] = 1.0 if self._is_login_anchor(rec) else 0
            score = self._score(f)
            mult, off = self.apply_policy(u, recent, f, reasons) if self.cfg.use_policy else (1.0, 0)
            score = (score + off) * mult
            if self.cfg.use_baseline: score += self.baseline(u, f, reasons)
            if self.cfg.use_tom and f.get("tom_intent", 0) > 0:
                score += self.cfg.tom_weight * f["tom_intent"]
                reasons.append(f"tom:{f['tom_intent']:.2f}")
            if self.cfg.use_forensics and f.get("forensics_phishing", 0) > 0:
                score += self.cfg.forensics_weight * f["forensics_phishing"]
                reasons.append(f"forensics:{f['forensics_phishing']:.2f}")
            early_thr, conf_thr = self.cfg.early_threshold, self.conf_threshold(u)
            tom_a = self._get_tom(u).get_threat_assessment(u) if self.cfg.use_tom else {}
            if score >= early_thr and self._can_fire(u, "early", s):
                self.model.emit_alert(Event(s, "alert_early", u, meta={"score": score, "threshold": early_thr, "reasons": reasons, "tom": tom_a}, label=rec.get("label"), scenario=rec.get("scenario")))
                self.last_alert[(u, "early")] = s
            if self.cfg.use_evidence_gate:
                ev_c, ev_w = self.calc_evidence(f, u)
                if ev_c < self.cfg.min_evidence_count or ev_w < self.cfg.min_evidence_weight: continue
            if score >= conf_thr and self._can_fire(u, "confirmed", s):
                self.model.emit_alert(Event(s, "alert_confirmed", u, meta={"score": score, "threshold": conf_thr, "reasons": reasons, "tom": tom_a}, label=rec.get("label"), scenario=rec.get("scenario")))
                self.last_alert[(u, "confirmed")] = s
                self.update_trust(u, rec.get("label") == "malicious")
                self.model.event_log.append(Event(s, "analyst_verdict", u, meta={"verdict": "malicious" if rec.get("label") == "malicious" else "benign"}, label=rec.get("label")))

# ============ Model ============
class InsiderModel(mesa.Model):
    def __init__(self, n_emp=30, n_power=4, n_exfil=3, n_stealth=2, n_takeover=1, n_staging=1, n_email=1, seed=42, siem_cfg=None, warmup=60):
        super().__init__(seed=seed)
        self.steps, self.action_events, self.ingest_events, self.alert_events, self.event_log = 0, [], [], [], []
        self.user_profile: Dict[int, Dict] = {}
        self.siem_cfg = siem_cfg or SIEMConfig()
        self.warmup_steps, self.phase, self.attack_enabled = warmup, "train", False
        for _ in range(n_emp): BenignEmployee(self, self.random.choices(["staff", "analyst", "admin"], [0.78, 0.18, 0.04])[0])
        for _ in range(n_power): BenignPowerUser(self, "analyst", self.random.choice([24, 36, 48]))
        base = 16 + warmup
        for i in range(n_exfil): MaliciousInsider(self, "exfil", base + i*12, 48)
        for j in range(n_stealth): MaliciousInsider(self, "stealth", 28 + warmup + j*18, 48)
        for k in range(n_takeover): MaliciousInsider(self, "acct_takeover", 22 + warmup + k*20, None)
        for k in range(n_staging): MaliciousInsider(self, "staging_exfil", 34 + warmup + k*24, 72)
        for k in range(n_email): MaliciousInsider(self, "email_only", 18 + warmup + k*26, None)
        DBMonitor(self); EmailMonitor(self); AuthMonitor(self); SIEMAgent(self, self.siem_cfg)
        self.datacollector = DataCollector(model_reporters={"alerts": lambda m: len(m.alert_events), "phase": lambda m: m.phase})

    def register_user(self, u, role, ah, approved):
        self.user_profile.setdefault(u, {}).update({"role": role, "allowed_after_hours": ah, "approved_partners": set(approved)})

    def _tag(self, e):
        e.phase = self.phase
        if e.meta is None: e.meta = {}
        e.meta["phase"] = self.phase

    def emit_action(self, e): self._tag(e); self.action_events.append(e); self.event_log.append(e)
    def emit_ingest(self, e): self._tag(e); self.ingest_events.append(e); self.event_log.append(e)
    def emit_alert(self, e): self._tag(e); self.alert_events.append(e); self.event_log.append(e)

    def step(self):
        self.steps += 1
        if self.steps >= self.warmup_steps: self.phase, self.attack_enabled = "test", True
        self.action_events, self.ingest_events, self.alert_events = [], [], []
        for a in self.agents:
            if isinstance(a, (BenignEmployee, BenignPowerUser, MaliciousInsider)): a.act()
        for a in self.agents:
            if isinstance(a, (DBMonitor, EmailMonitor, AuthMonitor)): a.monitor()
        for a in self.agents:
            if isinstance(a, SIEMAgent): a.correlate()
        self.datacollector.collect(self)

# ============ Evaluation ============
def evaluate(log, warmup):
    test = [e for e in log if e.phase == "test"]
    conf = [e for e in test if e.event_type == "alert_confirmed"]
    early = [e for e in test if e.event_type == "alert_early"]
    mal = [e for e in test if e.label == "malicious" and e.event_type not in {"siem_ingest", "analyst_verdict"}]
    mal_actors = set(e.actor_id for e in mal)
    conf_actors = set(e.actor_id for e in conf)
    detected = conf_actors & mal_actors
    tp = sum(1 for e in conf if e.label == "malicious")
    fp = sum(1 for e in conf if e.label != "malicious")
    prec = len(detected) / len(conf_actors) if conf_actors else 0
    rec = len(detected) / len(mal_actors) if mal_actors else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0
    ttd = []
    for a in detected:
        fm = min((e.step for e in mal if e.actor_id == a), default=None)
        fa = min((e.step for e in conf if e.actor_id == a), default=None)
        if fm and fa: ttd.append(fa - fm)
    tom_det = sum(1 for e in conf if (e.meta or {}).get("tom", {}).get("has_malicious_intent"))
    return {"f1": f1, "precision": prec, "recall": rec, "ttd_avg": np.mean(ttd) if ttd else 0, "ttd_max": max(ttd) if ttd else 0,
            "conf_total": len(conf), "conf_fp": fp, "conf_prec": tp/len(conf) if conf else 0,
            "early_total": len(early), "actors_total": len(mal_actors), "actors_detected": len(detected), "tom_detected": tom_det}

def run(T=240, cfg=None, warmup=60, runs=10, out="events_v5.jsonl"):
    results = []
    for r in range(runs):
        print(f"\n=== Run {r+1}/{runs} ===")
        m = InsiderModel(seed=42+r, siem_cfg=cfg, warmup=warmup)
        for _ in range(T): m.step()
        metrics = evaluate(m.event_log, warmup)
        results.append(metrics)
        print(f"F1: {metrics['f1']:.3f} (P={metrics['precision']:.3f}, R={metrics['recall']:.3f})")
        print(f"Confirmed: {metrics['conf_total']} (prec={metrics['conf_prec']:.3f}, FP={metrics['conf_fp']})")
        print(f"ToM detections: {metrics['tom_detected']}")
    with open(out, "w") as f:
        for e in m.event_log: f.write(json.dumps(e.to_dict()) + "\n")
    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    print("\n" + "="*65)
    print(f"AVERAGED RESULTS ACROSS {runs} RUNS (V5 Enhanced)")
    print("="*65)
    print(f"Actor F1:    {avg['f1']:.4f} (P={avg['precision']:.4f}, R={avg['recall']:.4f})")
    print(f"TTD:         avg={avg['ttd_avg']:.2f}, max={avg['ttd_max']:.2f}")
    print(f"Confirmed:   {avg['conf_total']:.1f} alerts, precision={avg['conf_prec']:.4f}, FP={avg['conf_fp']:.1f}")
    print(f"ToM:         {avg['tom_detected']:.1f} detections")
    print(f"Actors:      {avg['actors_detected']:.1f}/{avg['actors_total']:.1f} detected")
    return avg

if __name__ == "__main__":
    cfg = SIEMConfig(
        use_policy=True, use_baseline=True, use_trust=True, use_ml=True,
        use_tom=True, use_forensics=True,
        use_evidence_gate=True, use_peer_norm=True, use_regularity=True,
        early_threshold=2.0, base_confirmed_threshold=4.0,
        min_evidence_count=2, min_evidence_weight=2.5,
        tom_weight=2.0, tom_threshold=0.30, forensics_weight=1.5)
    print("="*70)
    print("INSIDER THREAT DETECTION V5 - Performance Enhanced")
    print("="*70)
    print("\nEnhancements over V4:")
    print("  + Evidence accumulation (require 2+ independent signals)")
    print("  + Peer group normalization (compare to role peers)")
    print("  + Regularity detection (whitelist scheduled behavior)")
    print("  + Contradicting evidence in ToM (reduce false intent)")
    print("  + Role-based adaptive thresholds")
    print("  + Enhanced policy compliance detection")
    print("="*70 + "\n")
    run(T=240, cfg=cfg, warmup=60, runs=10, out="events_v5.jsonl")
