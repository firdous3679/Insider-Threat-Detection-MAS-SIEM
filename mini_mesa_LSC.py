# mini_mas_mesa_optionB_v5_adaptive_layers_fixed.py
# A compact Mesa mini-MAS insider-threat simulator with:
# - benign + adversarial-benign power users
# - 5 attacker scenarios
# - SIEM with 4 toggleable layers: policy, EWMA baseline, trust-adaptive threshold, online learning
# - optional role-based ML anomaly model trained during warmup phase
#
# Fix 3: ML anomaly is suppressed during known partner_reporting cycles
# Fix 4: CONFIRMED alerts require an evidence gate (stricter than EARLY)

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Tuple, Set
from collections import Counter, defaultdict
import json
import math
import mesa
from mesa.datacollection import DataCollector

# Optional ML
try:
    from sklearn.ensemble import IsolationForest
except Exception:
    IsolationForest = None


# =========================
# Event Schema
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
    label: Optional[str] = None          # "benign" or "malicious"
    scenario: Optional[str] = None       # attacker scenario name
    phase: Optional[str] = None          # "train" or "test"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================
# SIEM Config (Layer Toggles)
# =========================
@dataclass
class SIEMConfig:
    # Layer toggles (for ablations)
    use_policy: bool = True
    use_baseline: bool = True
    use_trust: bool = True
    use_online_learning: bool = True
    use_ml: bool = True

    # thresholds (early fixed, confirmed base is swept)
    early_threshold: float = 3.0
    base_confirmed_threshold: float = 4.0

    # window + cooldown
    window: int = 48
    cooldown: int = 6

    # EWMA baseline params
    ewma_alpha: float = 0.08
    ewma_var_alpha: float = 0.08
    z_clip: float = 6.0
    baseline_weight: float = 0.9

    # trust params
    trust_init: float = 0.70
    trust_min: float = 0.10
    trust_max: float = 0.95
    trust_slope: float = 1.2
    trust_decay: float = 0.00            # small decay toward trust_init each step (e.g., 0.002)
    trust_tp_delta: float = -0.15        # TP decreases trust (more suspicious)
    trust_fp_delta: float = +0.04        # FP increases trust (more trusted)

    # online learning params (simple logistic SGD)
    lr: float = 0.04
    l2: float = 1e-4
    w_clip: float = 6.0

    # ML params (role-based IsolationForest)
    ml_weight: float = 2.0
    ml_score_threshold: float = 0.0    # ignore small anomaly scores
    # Only let ML nudge borderline cases near thresholds
    ml_margin: float = 3.0
    # Make ML influence EARLY more than CONFIRMED
    ml_weight_confirmed_scale: float = 1.0
    ml_train_sample_every: int = 1
    ml_min_samples: int = 25
    ml_refit_every: int = 15
    ml_freeze_after_warmup: bool = True # fit in warmup then freeze

    # feature weights (start "rule-like")
    w: Dict[str, float] = field(default_factory=lambda: {
        "anchor_email": 2.0,
        "anchor_login": 2.0,
        "export_large": 2.0,
        "export_small": 1.0,
        "staging_export": 1.0,
        "after_hours": 1.0,
        "sens_read_burst": 1.0,
        "login_burst": 1.0,
        "unapproved_partner": 2.0,
        "email_burst": 1.0,
    })


# =========================
# Helper: safe clip
# =========================
def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =========================
# Simple role-based ML anomaly model
# =========================
class EnhancedRoleAnomalyModel:
    """
    Enhanced ML model with richer features and smarter training.
    Key improvements:
    1. Rich temporal features (velocity, burst patterns, ratios)
    2. Clean training (exclude adversarial benign users)
    3. Separate models per role AND behavior pattern
    4. Better anomaly scoring with confidence
    """
    def __init__(self, cfg: SIEMConfig):
        self.cfg = cfg
        # Separate buffers for clean baseline vs all data
        self.clean_buffers: Dict[str, List[List[float]]] = defaultdict(list)
        self.all_buffers: Dict[str, List[List[float]]] = defaultdict(list)
        self.models: Dict[str, Any] = {}
        self.last_fit_step: Dict[str, int] = defaultdict(lambda: -10**9)
        
        # Track user patterns to identify adversarial benign users
        self.user_cycle_patterns: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
            "has_partner_cycle": False,
            "export_count": 0,
            "large_export_count": 0,
            "external_email_count": 0,
        })

    def extract_rich_features(self, recent: List[Dict[str, Any]], window_size: int) -> List[float]:
        """
        Extract 20+ features capturing temporal patterns, ratios, and anomalies.
        These features are much more discriminative than simple counts.
        """
        if not recent:
            return [0.0] * 24
        
        # Time-based grouping (first half vs second half of window)
        mid = len(recent) // 2
        first_half = recent[:mid]
        second_half = recent[mid:]
        
        # === Basic counts ===
        login_count = sum(1 for r in recent if r.get("event_type") == "login")
        db_count = sum(1 for r in recent if r.get("event_type") == "db_query")
        sens_reads = sum(1 for r in recent if r.get("event_type") == "db_query" and r.get("resource") == "patients_table")
        exports = sum(1 for r in recent if r.get("event_type") == "data_export")
        ext_emails = sum(1 for r in recent if r.get("event_type") == "email_send" and (r.get("dst") or "").endswith("outside"))
        after_hours = sum(1 for r in recent if (r.get("meta") or {}).get("after_hours"))
        
        # === Velocity features (acceleration/deceleration) ===
        login_first = sum(1 for r in first_half if r.get("event_type") == "login")
        login_second = sum(1 for r in second_half if r.get("event_type") == "login")
        login_velocity = (login_second - login_first) / max(1, mid)  # acceleration
        
        sens_first = sum(1 for r in first_half if r.get("event_type") == "db_query" and r.get("resource") == "patients_table")
        sens_second = sum(1 for r in second_half if r.get("event_type") == "db_query" and r.get("resource") == "patients_table")
        sens_velocity = (sens_second - sens_first) / max(1, mid)
        
        # === Ratio features (unusual proportions) ===
        sens_ratio = sens_reads / max(1, db_count)  # % of queries that are sensitive
        export_ratio = exports / max(1, db_count)  # export after read ratio
        external_ratio = ext_emails / max(1, sum(1 for r in recent if r.get("event_type") == "email_send"))
        after_hours_ratio = after_hours / max(1, len(recent))
        
        # === Byte volume features ===
        total_bytes = sum(r.get("bytes", 0) for r in recent if r.get("event_type") in ["db_query", "data_export"])
        max_bytes = max((r.get("bytes", 0) for r in recent if r.get("event_type") in ["db_query", "data_export"]), default=0)
        avg_bytes = total_bytes / max(1, db_count + exports)
        
        # Large exports (>100KB)
        large_exports = sum(1 for r in recent if r.get("event_type") == "data_export" and r.get("bytes", 0) > 100000)
        
        # === Email features ===
        large_attachments = sum(1 for r in recent if r.get("event_type") == "email_send" 
                                and (r.get("meta") or {}).get("attachment_kb", 0) > 500)
        
        # Unapproved external destinations
        unapproved_emails = sum(1 for r in recent if r.get("event_type") == "email_send" 
                                and (r.get("dst") or "").endswith("outside")
                                and not (r.get("meta") or {}).get("approved_partner", False))
        
        # === Sequential pattern features ===
        # Check for read -> export -> email pattern (classic exfil)
        has_read_export_email = 0
        for i in range(len(recent) - 2):
            if (recent[i].get("event_type") == "db_query" and
                recent[i+1].get("event_type") == "data_export" and
                recent[i+2].get("event_type") == "email_send"):
                has_read_export_email = 1
                break
        
        # Burst detection (events clustered in small time window)
        steps = [r.get("step", 0) for r in recent]
        if steps:
            step_range = max(steps) - min(steps)
            event_density = len(recent) / max(1, step_range)  # events per step
        else:
            event_density = 0.0
        
        # === Diversity features ===
        unique_resources = len(set(r.get("resource") for r in recent if r.get("resource")))
        unique_dsts = len(set(r.get("dst") for r in recent if r.get("dst")))
        
        # === Context anomalies ===
        # Missing ticket IDs when there should be one
        exports_without_ticket = sum(1 for r in recent if r.get("event_type") == "data_export" 
                                     and not (r.get("meta") or {}).get("ticket_id"))
        
        # Device changes during activity
        device_changes = sum(1 for r in recent if (r.get("meta") or {}).get("device_change"))
        
        return [
            float(login_count),
            float(db_count),
            float(sens_reads),
            float(exports),
            float(ext_emails),
            float(after_hours),
            float(login_velocity),
            float(sens_velocity),
            float(sens_ratio),
            float(export_ratio),
            float(external_ratio),
            float(after_hours_ratio),
            float(total_bytes) / 10000.0,  # scale down
            float(max_bytes) / 10000.0,
            float(avg_bytes) / 1000.0,
            float(large_exports),
            float(large_attachments),
            float(unapproved_emails),
            float(has_read_export_email),
            float(event_density),
            float(unique_resources),
            float(unique_dsts),
            float(exports_without_ticket),
            float(device_changes),
        ]
    def is_clean_training_sample(self, actor_id: int, recent: List[Dict[str, Any]]) -> bool:
            """
            Determine if this sample should be used for clean baseline training.
            Exclude adversarial benign users (power users doing partner reporting).
            """
            # Check for partner reporting cycle markers
            in_cycle = any(((r.get("meta") or {}).get("cycle") == "partner_reporting") for r in recent)
            if in_cycle:
                self.user_cycle_patterns[actor_id]["has_partner_cycle"] = True
                return False
            
            # Check for suspicious patterns even without cycle markers
            exports = sum(1 for r in recent if r.get("event_type") == "data_export")
            large_exports = sum(1 for r in recent if r.get("event_type") == "data_export" and r.get("bytes", 0) > 100000)
            ext_emails = sum(1 for r in recent if r.get("event_type") == "email_send" and (r.get("dst") or "").endswith("outside"))
            
            self.user_cycle_patterns[actor_id]["export_count"] += exports
            self.user_cycle_patterns[actor_id]["large_export_count"] += large_exports
            self.user_cycle_patterns[actor_id]["external_email_count"] += ext_emails
            
            # If user frequently does large exports + external emails, likely power user
            if (self.user_cycle_patterns[actor_id]["large_export_count"] > 3 and
                self.user_cycle_patterns[actor_id]["external_email_count"] > 5):
                return False
            
            # This looks like normal staff activity
            return True

    def add(self, role: str, vec: List[float], actor_id: int, recent: List[Dict[str, Any]]):
        """Add sample to appropriate buffer(s)."""
        self.all_buffers[role].append(vec)
        
        if self.is_clean_training_sample(actor_id, recent):
            self.clean_buffers[role].append(vec)

    def maybe_fit(self, role: str, step: int):
        if IsolationForest is None:
            return
        
        # Prefer clean buffer, fall back to all if insufficient
        buffer = self.clean_buffers[role]
        if len(buffer) < self.cfg.ml_min_samples:
            buffer = self.all_buffers[role]
        
        if len(buffer) < self.cfg.ml_min_samples:
            return
        if (step - self.last_fit_step[role]) < self.cfg.ml_refit_every:
            return

        # Use more recent samples, larger window
        X = buffer[-min(500, len(buffer)):]
        
        clf = IsolationForest(
            n_estimators=250,  # More trees
            max_samples='auto',
            contamination=0.01,  # Expect fewer anomalies in clean data
            random_state=42,
            bootstrap=True,
        )
        clf.fit(X)
        self.models[role] = clf
        self.last_fit_step[role] = step

    def score(self, role: str, vec: List[float]) -> float:
        """
        Returns non-negative anomaly score with better calibration.
        Higher = more anomalous.
        """
        if IsolationForest is None:
            return 0.0
        clf = self.models.get(role)
        if clf is None:
            return 0.0

        # Get anomaly score and confidence
        raw_score = -float(clf.decision_function([vec])[0])
        raw_score = max(0.0, raw_score)
        
        # More aggressive thresholding - only flag clear anomalies
        if raw_score < self.cfg.ml_score_threshold:
            return 0.0
        
        # Exponential scaling for strong anomalies
        excess = raw_score - self.cfg.ml_score_threshold
        scaled = excess * (1.0 + excess * 0.5)  # quadratic boost
        
        return min(scaled, 5.0)  # Cap at 5.0
 #=====================================================
# Updated SIEMAgent methods
# =====================================================

def enhanced_siem_init(self, model: mesa.Model, cfg: SIEMConfig):
    """Replace SIEMAgent.__init__ with this."""
    super(SIEMAgent, self).__init__(model)
    self.cfg = cfg

    self.last_alert_step: Dict[Tuple[int, str], int] = {}
    self.recent_by_user: Dict[int, List[Dict[str, Any]]] = {}

    self.mu: Dict[int, Dict[str, float]] = {}
    self.var: Dict[int, Dict[str, float]] = {}
    self.trust: Dict[int, float] = {}

    # Use enhanced ML model
    self.ml = EnhancedRoleAnomalyModel(cfg)


def enhanced_ml_vector(self, recent: List[Dict[str, Any]]) -> List[float]:
    """Replace SIEMAgent.ml_vector with this."""
    return self.ml.extract_rich_features(recent, self.cfg.window)


def enhanced_correlate(self):
    """
    Replace the ML training section in SIEMAgent.correlate.
    Find this section and replace it:
    
    # OLD CODE:
    if train_ml_now and (s % self.cfg.ml_train_sample_every == 0):
        for user, hist in self.recent_by_user.items():
            recent = hist[-self.cfg.window:]
            if not recent:
                continue
            role = self.model.user_profile.get(user, {}).get("role", "staff")
            vec = self.ml_vector(recent)
            self.ml.add(role, vec)
            self.ml.maybe_fit(role, s)
    """
    s = self.model.steps
    self.decay_trust()

    # Update history from ingest stream
    for ing in self.model.ingest_events:
        user = ing.actor_id
        rec = (ing.meta or {}).get("record", {})
        if not rec:
            continue
        self.recent_by_user.setdefault(user, []).append(rec)
        self.recent_by_user[user] = self.recent_by_user[user][-2000:]

    # === ENHANCED ML TRAINING ===
    phase = getattr(self.model, "phase", "test")
    train_ml_now = (self.cfg.use_ml and (phase == "train" or not self.cfg.ml_freeze_after_warmup))
    if train_ml_now and (s % self.cfg.ml_train_sample_every == 0):
        for user, hist in self.recent_by_user.items():
            recent = hist[-self.cfg.window:]
            if not recent:
                continue
            role = self.model.user_profile.get(user, {}).get("role", "staff")
            vec = self.ml_vector(recent)  # Now uses rich features
            self.ml.add(role, vec, user, recent)  # Pass actor_id and recent for filtering
            self.ml.maybe_fit(role, s)

    # [Rest of correlate() continues as before...]
# =====================================================
# Updated SIEMConfig with better ML defaults
# =====================================================

def get_enhanced_siem_config():
    """
    Return a SIEMConfig with ML-friendly settings.
    Use this when creating your model.
    """
    cfg = SIEMConfig(
        # Layer toggles
        use_policy=True,
        use_baseline=True,
        use_trust=True,
        use_online_learning=True,
      
        
        # Thresholds
        early_threshold=3.0,
        base_confirmed_threshold=4.0,
        
        # ML settings - ENHANCED
         use_ml=True,
         ml_weight=2.5,              # Increased from 1.2
         ml_score_threshold=0.02,    # Lowered from 0.03
         ml_margin=1.5,              # Widened from 0.75
         ml_weight_confirmed_scale=0.6,  # Increased from 0.25
         ml_train_sample_every=2,    # More frequent (was 3)
         ml_min_samples=80,          # More samples needed (was 60)
         ml_refit_every=12,          # Refit more often (was 15)
         ml_freeze_after_warmup=False,  # Continue learning (was True)
        
        # Other params
        window=48,
        cooldown=6,
        ewma_alpha=0.08,
        ewma_var_alpha=0.08,
        z_clip=6.0,
        baseline_weight=0.9,
        trust_init=0.70,
        trust_min=0.10,
        trust_max=0.95,
        trust_slope=1.2,
        trust_decay=0.00,
        trust_tp_delta=-0.15,
        trust_fp_delta=+0.04,
        lr=0.04,
        l2=1e-4,
        w_clip=6.0,
    )
    return cfg
# =========================
# Benign Agents
# =========================
class BenignEmployee(mesa.Agent):
    """
    Everyday user activity, plus a tiny bit of realism:
      - a few staff are "on-call" but not always correctly marked as allowed after-hours
      - occasional external emails with attachments (vendor/conference/etc.)
      - occasional device changes
      - a small number of admins doing after-hours maintenance (can look suspicious)
    """
    def __init__(self, model: mesa.Model, role: str = "staff"):
        super().__init__(model)  # Mesa 3.x: just pass model, unique_id is auto-assigned
        self.role = role

        # some staff are legitimately on-call; some are "shadow on-call" (policy mismatch)
        self.on_call = (self.role == "staff" and self.random.random() < 0.12)
        self.shadow_on_call = (self.on_call and self.random.random() < 0.25)

        # device identity can change (reimage, loaner, travel laptop)
        self.device_id = f"corp-laptop-{self.unique_id}"
        self.device_change_prob = 0.010 if self.role != "admin" else 0.030

        # low-rate external email (benign), sometimes to unapproved destinations
        self.external_email_prob = 0.003 if self.role == "staff" else 0.008
        self.external_unapproved_prob = 0.55  # when emailing outside, chance it's not on allowlist

        approved = set() if self.role == "staff" else {"partner@outside"}
        allowed_after = (self.role in {"analyst", "admin"}) or (self.on_call and not self.shadow_on_call)

        # register context profile
        self.model.register_user(
            self.unique_id,
            role=self.role,
            allowed_after_hours=allowed_after,
            approved_partners=approved,
        )

    def act(self):
        s = self.model.steps
        hour = s % 24
        at_work = 8 <= hour <= 17
        after_hours = not at_work

        # device change event (silent): next login may look "new"
        device_change = False
        if self.random.random() < self.device_change_prob:
            self.device_id = f"corp-laptop-{self.unique_id}-r{s}"
            device_change = True

        # login (admins and on-call folks do more after-hours)
        base_login = 0.08 if at_work else 0.01
        if after_hours and self.role == "admin":
            base_login = 0.05
        elif after_hours and self.on_call:
            base_login = 0.03

        if self.random.random() < base_login:
            burst = 1
            # admin maintenance can generate "bursty" auth logs
            if after_hours and self.role == "admin" and self.random.random() < 0.35:
                burst = 2 + (1 if self.random.random() < 0.5 else 0)  # 2-3 logins
            for _ in range(burst):
                self.model.emit_action(Event(
                    step=s, event_type="login", actor_id=self.unique_id,
                    meta={
                        "role": self.role,
                        "after_hours": after_hours,
                        "device_id": self.device_id,
                        "device_change": device_change,
                        "on_call": self.on_call,
                    },
                    label="benign"
                ))

        # DB query
        if at_work and self.random.random() < 0.28:
            table = self.random.choice(["public_table", "project_table", "patients_table"])
            self.model.emit_action(Event(
                step=s, event_type="db_query", actor_id=self.unique_id,
                resource=table, action="SELECT",
                bytes=self.random.randint(200, 6000),
                meta={"role": self.role, "after_hours": after_hours, "on_call": self.on_call},
                label="benign"
            ))

        # admin after-hours maintenance reads (benign but scary-looking)
        if after_hours and self.role == "admin" and self.random.random() < 0.25:
            reads = 2 + (1 if self.random.random() < 0.6 else 0)
            for _ in range(reads):
                self.model.emit_action(Event(
                    step=s, event_type="db_query", actor_id=self.unique_id,
                    resource="patients_table", action="SELECT",
                    bytes=self.random.randint(1500, 12000),
                    meta={"role": self.role, "after_hours": True, "maintenance": True},
                    label="benign"
                ))

        # internal email
        if at_work and self.random.random() < 0.10:
            self.model.emit_action(Event(
                step=s, event_type="email_send", actor_id=self.unique_id,
                dst="someone@org",
                meta={"attachment_kb": 0, "subject": "quick question", "role": self.role, "after_hours": after_hours},
                label="benign"
            ))

        # occasional external email with attachment (benign, but SIEMs hate it)
        if self.random.random() < self.external_email_prob:
            unapproved = (self.random.random() < self.external_unapproved_prob)
            dst = "newpartner@outside" if unapproved else "partner@outside"
            attach = self.random.randint(220, 900) if unapproved else self.random.randint(210, 600)
            self.model.emit_action(Event(
                step=s, event_type="email_send", actor_id=self.unique_id,
                dst=dst,
                meta={
                    "attachment_kb": attach,
                    "subject": "document",
                    "role": self.role,
                    "after_hours": after_hours,
                    "approved_partner": (not unapproved),
                    "ad_hoc_external": True,
                },
                label="benign"
            ))


class BenignPowerUser(mesa.Agent):
    """
    Benign but adversarial: does partner reporting cycles that can resemble exfil.
    Realism knobs added:
      - sometimes ticket_id is missing (people forget)
      - sometimes the "partner" is a new recipient not on the allowlist (process drift)
      - sometimes there is an ad-hoc external send outside the formal cycle
    """
    def __init__(
        self,
        model: mesa.Model,
        role: str = "analyst",
        report_every: int = 36,
        fp_probability: float = 0.70,
        after_hours_probability: float = 0.30,
        ticket_missing_probability: float = 0.15,
        new_partner_probability: float = 0.12,
        ad_hoc_external_probability: float = 0.02
    ):
        super().__init__(model)  # Mesa 3.x: just pass model, unique_id is auto-assigned
        self.role = role
        self.report_every = report_every
        self.fp_probability = fp_probability
        self.after_hours_probability = after_hours_probability

        self.ticket_missing_probability = ticket_missing_probability
        self.new_partner_probability = new_partner_probability
        self.ad_hoc_external_probability = ad_hoc_external_probability

        self.in_cycle = False
        self.cycle_step_start = 0
        self.force_after_hours = False
        self.suspicious_mode = False
        self.phase = 0

        # per-cycle context
        self.cycle_ticket_id: Optional[str] = None
        self.cycle_partner_dst: str = "partner@outside"

        self.model.register_user(
            self.unique_id,
            role=self.role,
            allowed_after_hours=True,
            approved_partners={"partner@outside"},  # allowlisted "known" partner
        )

    def act(self):
        s = self.model.steps
        hour = s % 24
        at_work = 8 <= hour <= 17
        after_hours = not at_work

        # normal sensitive work
        if at_work and self.random.random() < 0.22:
            self.model.emit_action(Event(
                step=s, event_type="db_query", actor_id=self.unique_id,
                resource="patients_table", action="SELECT",
                bytes=self.random.randint(2000, 16000),
                meta={"role": self.role, "reason": "audit/analysis", "after_hours": after_hours},
                label="benign"
            ))

        # normal internal email
        if at_work and self.random.random() < 0.12:
            self.model.emit_action(Event(
                step=s, event_type="email_send", actor_id=self.unique_id,
                dst="team@org",
                meta={"attachment_kb": 0, "subject": "update", "role": self.role, "after_hours": after_hours},
                label="benign"
            ))

        # ad-hoc external send (benign, but sometimes not pre-approved)
        if self.random.random() < self.ad_hoc_external_probability:
            dst = "partner@outside" if self.random.random() < 0.75 else "newpartner@outside"
            self.model.emit_action(Event(
                step=s, event_type="email_send", actor_id=self.unique_id,
                dst=dst,
                meta={"role": self.role, "after_hours": after_hours,
                      "ad_hoc_external": True,
                      "approved_partner": (dst == "partner@outside"),
                      "attachment_kb": self.random.randint(220, 1800),
                      "subject": "requested data"},
                label="benign"
            ))

        # start reporting cycle
        if (s % self.report_every == 0) and (not self.in_cycle):
            self.in_cycle = True
            self.cycle_step_start = s
            self.phase = 1
            self.force_after_hours = (self.random.random() < self.after_hours_probability)
            self.suspicious_mode = (self.random.random() < self.fp_probability)

            # choose partner destination (sometimes "new partner" appears)
            self.cycle_partner_dst = "newpartner@outside" if (self.random.random() < self.new_partner_probability) else "partner@outside"

            # ticket_id sometimes missing
            if self.random.random() < self.ticket_missing_probability:
                self.cycle_ticket_id = None
            else:
                self.cycle_ticket_id = f"TCKT-{(self.unique_id*1000 + s) % 99999:05d}"

        if not self.in_cycle:
            return

        if self.force_after_hours and not after_hours:
            return

        # Phase 1: reads
        if self.phase == 1:
            reads = 2 + (1 if self.random.random() < 0.6 else 0)
            for _ in range(reads):
                self.model.emit_action(Event(
                    step=s, event_type="db_query", actor_id=self.unique_id,
                    resource="patients_table", action="SELECT",
                    bytes=self.random.randint(3000, 25000),
                    meta={"role": self.role, "after_hours": after_hours,
                          "cycle": "partner_reporting", "ticket_id": self.cycle_ticket_id},
                    label="benign"
                ))
            if s - self.cycle_step_start >= 1:
                self.phase = 2
            return

        # Phase 2: export (sometimes looks like exfil)
        if self.phase == 2:
            if self.suspicious_mode:
                action = "EXPORT"  # looks like exfil
                dest = "partner_share"
            else:
                action = "EXPORT_INTERNAL"
                dest = "internal_share"

            self.model.emit_action(Event(
                step=s, event_type="data_export", actor_id=self.unique_id,
                resource="patients_table", action=action,
                bytes=self.random.randint(25000, 180000),
                meta={"role": self.role, "destination": dest, "after_hours": after_hours,
                      "cycle": "partner_reporting", "ticket_id": self.cycle_ticket_id},
                label="benign"
            ))
            self.phase = 3
            return

        # Phase 3: external partner email (with or without ticket)
        if self.phase == 3:
            self.model.emit_action(Event(
                step=s, event_type="email_send", actor_id=self.unique_id,
                dst=self.cycle_partner_dst,
                meta={"role": self.role, "after_hours": after_hours, "cycle": "partner_reporting",
                      "ticket_id": self.cycle_ticket_id,
                      "approved_partner": (self.cycle_partner_dst == "partner@outside"),
                      "attachment_kb": self.random.randint(220, 2500),
                      "subject": "partner report"},
                label="benign"
            ))
            self.in_cycle = False
            self.phase = 0
            self.force_after_hours = False
            self.suspicious_mode = False
            self.cycle_ticket_id = None
            self.cycle_partner_dst = "partner@outside"
            return
class MaliciousInsider(mesa.Agent):
    """
    scenarios:
      - exfil: reads -> EXPORT -> external email attach
      - stealth: low-and-slow reads -> external email attach (no export)
      - acct_takeover: odd-hour admin-like logins + bursts + sensitive reads (no export needed)
      - staging_exfil: reads -> EXPORT_INTERNAL (staging) -> later external email attach
      - email_only: repeated external emails with attachments, minimal exports
    """
    def __init__(self, model: mesa.Model, scenario: str, start_step: int, repeat_every: Optional[int] = 48):
        super().__init__(model)  # Mesa 3.x: just pass model, unique_id is auto-assigned
        self.scenario = scenario
        self.start_step = start_step
        self.repeat_every = repeat_every
        self.phase = 0

        # register as "staff" (attack behavior is in events)
        self.model.register_user(
            self.unique_id,
            role="staff",
            allowed_after_hours=False,
            approved_partners=set(),
        )

        # some adversaries avoid obvious anchors (small attachments, alternate channels, etc.)
        # This is intentionally imperfect: real attackers are rarely polite enough to be obvious.
        self.evade_email = (self.random.random() < 0.15)

    def _after_hours(self, step: int) -> bool:
        hour = step % 24
        return (hour < 7) or (hour > 19)


    def _pick_attachment(self, lo_kb: int, hi_kb: int) -> int:
        # If evasion is enabled, often keep attachments below the email-anchor threshold.
        if self.evade_email and (self.random.random() < 0.70):
            return self.random.randint(20, 190)
        return self.random.randint(lo_kb, hi_kb)
    def act(self):
        # disabled during warmup
        if not getattr(self.model, "attack_enabled", True):
            return

        s = self.model.steps
        after_hours = self._after_hours(s)

        if self.phase == 0:
            if s >= self.start_step:
                self.phase = 1
            else:
                return

        # 1) Classic EXFIL
        if self.scenario == "exfil":
            if self.phase == 1:
                reads = 1 + (1 if self.random.random() < 0.5 else 0)
                for _ in range(reads):
                    self.model.emit_action(Event(
                        step=s, event_type="db_query", actor_id=self.unique_id,
                        resource="patients_table", action="SELECT",
                        bytes=self.random.randint(8000, 70000),
                        meta={"after_hours": after_hours},
                        label="malicious", scenario=self.scenario
                    ))
                if self.random.random() < 0.60:
                    self.phase = 2
                return

            if self.phase == 2:
                self.model.emit_action(Event(
                    step=s, event_type="data_export", actor_id=self.unique_id,
                    resource="patients_table", action="EXPORT",
                    bytes=self.random.randint(80000, 260000),
                    meta={"after_hours": after_hours},
                    label="malicious", scenario=self.scenario
                ))
                self.phase = 3
                return

            if self.phase == 3:
                self.model.emit_action(Event(
                    step=s, event_type="email_send", actor_id=self.unique_id,
                    dst="external@outside",
                    meta={"attachment_kb": self._pick_attachment(600, 4000), "subject": "report", "after_hours": after_hours},
                    label="malicious", scenario=self.scenario
                ))
                self.phase = 4
                return

        # 2) STEALTH (low-and-slow)
        if self.scenario == "stealth":
            if self.phase == 1:
                if self.random.random() < (0.85 if after_hours else 0.35):
                    self.model.emit_action(Event(
                        step=s, event_type="db_query", actor_id=self.unique_id,
                        resource="patients_table", action="SELECT",
                        bytes=self.random.randint(800, 5000),
                        meta={"after_hours": after_hours, "low_slow": True},
                        label="malicious", scenario=self.scenario
                    ))
                if self.random.random() < 0.08:
                    self.phase = 3
                return

            if self.phase == 3:
                self.model.emit_action(Event(
                    step=s, event_type="email_send", actor_id=self.unique_id,
                    dst="external@outside",
                    meta={"attachment_kb": self._pick_attachment(300, 2500), "subject": "summary", "after_hours": after_hours},
                    label="malicious", scenario=self.scenario
                ))
                self.phase = 4
                return

        # 3) ACCOUNT TAKEOVER
        if self.scenario == "acct_takeover":
            if after_hours and self.random.random() < 0.70:
                burst = 1 + (1 if self.random.random() < 0.6 else 0)
                for _ in range(burst):
                    self.model.emit_action(Event(
                        step=s, event_type="login", actor_id=self.unique_id,
                        meta={"role": "admin", "after_hours": True, "takeover": True, "device_id": "unknown-device"},
                        label="malicious", scenario=self.scenario
                    ))

                reads = 2 + (1 if self.random.random() < 0.7 else 0)
                for _ in range(reads):
                    self.model.emit_action(Event(
                        step=s, event_type="db_query", actor_id=self.unique_id,
                        resource="patients_table", action="SELECT",
                        bytes=self.random.randint(1500, 9000),
                        meta={"after_hours": True, "takeover": True},
                        label="malicious", scenario=self.scenario
                    ))

            if after_hours and self.random.random() < 0.10:
                self.model.emit_action(Event(
                    step=s, event_type="email_send", actor_id=self.unique_id,
                    dst="external@outside",
                    meta={"attachment_kb": self._pick_attachment(250, 1800), "subject": "notes",
                          "after_hours": True, "takeover": True},
                    label="malicious", scenario=self.scenario
                ))
            return

        # 4) STAGING THEN EXFIL
        if self.scenario == "staging_exfil":
            if self.phase == 1:
                reads = 3 + (1 if self.random.random() < 0.6 else 0)
                for _ in range(reads):
                    self.model.emit_action(Event(
                        step=s, event_type="db_query", actor_id=self.unique_id,
                        resource="patients_table", action="SELECT",
                        bytes=self.random.randint(5000, 45000),
                        meta={"after_hours": after_hours, "staging": True},
                        label="malicious", scenario=self.scenario
                    ))
                if self.random.random() < 0.55:
                    self.phase = 2
                return

            if self.phase == 2:
                self.model.emit_action(Event(
                    step=s, event_type="data_export", actor_id=self.unique_id,
                    resource="patients_table", action="EXPORT_INTERNAL",
                    bytes=self.random.randint(60000, 220000),
                    meta={"after_hours": after_hours, "staging": True, "destination": "internal_staging"},
                    label="malicious", scenario=self.scenario
                ))
                self.phase = 3
                return

            if self.phase == 3:
                self.model.emit_action(Event(
                    step=s, event_type="email_send", actor_id=self.unique_id,
                    dst="external@outside",
                    meta={"attachment_kb": self._pick_attachment(600, 4500), "subject": "report",
                          "after_hours": after_hours, "staging": True},
                    label="malicious", scenario=self.scenario
                ))
                self.phase = 4
                return

        # 5) EMAIL-ONLY LEAKAGE
        if self.scenario == "email_only":
            if self.random.random() < (0.30 if after_hours else 0.12):
                self.model.emit_action(Event(
                    step=s, event_type="email_send", actor_id=self.unique_id,
                    dst="external@outside",
                    meta={"attachment_kb": self._pick_attachment(250, 2500), "subject": "doc",
                          "after_hours": after_hours, "email_only": True},
                    label="malicious", scenario=self.scenario
                ))
            if self.random.random() < 0.20:
                self.model.emit_action(Event(
                    step=s, event_type="db_query", actor_id=self.unique_id,
                    resource="patients_table", action="SELECT",
                    bytes=self.random.randint(500, 3500),
                    meta={"after_hours": after_hours, "email_only": True},
                    label="malicious", scenario=self.scenario
                ))
            return

        # repeat for phased scenarios
        if self.phase == 4 and self.repeat_every is not None:
            if (s - self.start_step) % self.repeat_every == 0 and s > self.start_step:
                self.phase = 1


# =========================
# Monitoring Agents -> SIEM ingest
# =========================
class DBMonitor(mesa.Agent):
    def __init__(self, model: mesa.Model):
        super().__init__(model)  # Mesa 3.x: just pass model, unique_id is auto-assigned

    def monitor(self):
        s = self.model.steps
        for e in self.model.action_events:
            if e.event_type in {"db_query", "data_export"}:
                self.model.emit_ingest(Event(
                    step=s, event_type="siem_ingest", actor_id=e.actor_id,
                    meta={"source": "db", "record": e.to_dict()},
                    label=e.label, scenario=e.scenario
                ))


class EmailMonitor(mesa.Agent):
    def __init__(self, model: mesa.Model):
        super().__init__(model)  # Mesa 3.x: just pass model, unique_id is auto-assigned

    def monitor(self):
        s = self.model.steps
        for e in self.model.action_events:
            if e.event_type == "email_send":
                self.model.emit_ingest(Event(
                    step=s, event_type="siem_ingest", actor_id=e.actor_id,
                    meta={"source": "email", "record": e.to_dict()},
                    label=e.label, scenario=e.scenario
                ))


class AuthMonitor(mesa.Agent):
    def __init__(self, model: mesa.Model):
        super().__init__(model)  # Mesa 3.x: just pass model, unique_id is auto-assigned

    def monitor(self):
        s = self.model.steps
        for e in self.model.action_events:
            if e.event_type == "login":
                self.model.emit_ingest(Event(
                    step=s, event_type="siem_ingest", actor_id=e.actor_id,
                    meta={"source": "auth", "record": e.to_dict()},
                    label=e.label, scenario=e.scenario
                ))


# =========================
# SIEM Agent (4-layer + optional ML)
# =========================
class SIEMAgent(mesa.Agent):
    def __init__(self, model: mesa.Model, cfg: SIEMConfig):
        super().__init__(model)  # Mesa 3.x: just pass model, unique_id is auto-assigned
        self.cfg = cfg

        self.last_alert_step: Dict[Tuple[int, str], int] = {}
        self.recent_by_user: Dict[int, List[Dict[str, Any]]] = {}

        # baseline state (EWMA mean/var per actor per feature)
        self.mu: Dict[int, Dict[str, float]] = {}
        self.var: Dict[int, Dict[str, float]] = {}

        # trust state
        self.trust: Dict[int, float] = {}

        # ML
        self.ml = EnhancedRoleAnomalyModel(cfg)


    def _is_external(self, dst: str) -> bool:
        """Return True if destination looks like an external address in this simulation."""
        d = (dst or "").lower()
        # Simulation uses addresses like "partner@outside" and "external@outside"
        return d.endswith("outside") or d.endswith("@outside")
    # -------- anchors --------
    def _is_email_anchor(self, rec: Dict[str, Any]) -> bool:
        if rec.get("event_type") != "email_send":
            return False
        meta = rec.get("meta") or {}
        attach = meta.get("attachment_kb", 0)
        dst = rec.get("dst") or ""
        return dst.endswith("outside") and attach > 200

    def _is_login_anchor(self, rec: Dict[str, Any]) -> bool:
        if rec.get("event_type") != "login":
            return False
        meta = rec.get("meta") or {}
        role = meta.get("role")
        after_hours = bool(meta.get("after_hours"))
        takeover = bool(meta.get("takeover"))
        return (after_hours and role == "admin") or takeover

    # -------- features --------
    def extract_features(self, recent: List[Dict[str, Any]], actor_id: int) -> Dict[str, float]:
        feats = {k: 0.0 for k in self.cfg.w.keys()}

        login_count = sum(1 for r in recent if r.get("event_type") == "login")
        sens_reads = sum(1 for r in recent if r.get("event_type") == "db_query" and r.get("resource") == "patients_table")
        external_emails = sum(
            1 for r in recent
            if r.get("event_type") == "email_send"
            and (r.get("dst") or "").endswith("outside")
            and ((r.get("meta") or {}).get("attachment_kb", 0) > 200)
        )

        feats["login_burst"] = 1.0 if login_count >= 3 else 0.0
        feats["sens_read_burst"] = 1.0 if sens_reads >= 6 else 0.0
        feats["email_burst"] = 1.0 if external_emails >= 3 else 0.0

        # after-hours evidence
        feats["after_hours"] = 1.0 if any((r.get("meta") or {}).get("after_hours") for r in recent) else 0.0

        # exports (bytes-aware, explicit EXPORT only)
        export_bytes = [
            r.get("bytes", 0) for r in recent
            if r.get("event_type") == "data_export"
            and r.get("resource") == "patients_table"
            and r.get("action") == "EXPORT"
        ]
        if export_bytes:
            mx = max(export_bytes)
            feats["export_large"] = 1.0 if mx >= 120000 else 0.0
            feats["export_small"] = 1.0 if mx < 120000 else 0.0

        # staging export
        feats["staging_export"] = 1.0 if any(
            r.get("event_type") == "data_export"
            and r.get("resource") == "patients_table"
            and r.get("action") in {"EXPORT_INTERNAL", "EXPORT_STAGING"}
            for r in recent
        ) else 0.0

        feats["unapproved_partner"] = 0.0
        return feats

    # -------- ML vector --------
    def ml_vector(self, recent: List[Dict[str, Any]]) -> List[float]:
        return self.ml.extract_rich_features(recent, self.cfg.window)
    # -------- policy layer --------
    def apply_policy(self, actor_id: int, recent: List[Dict[str, Any]], feats: Dict[str, float], reasons: List[str]) -> Tuple[float, float]:
        """
        Policy layer: discounts known benign cycles and applies allowlist / after-hours rules.
        Returns (multiplier, additive_offset) applied to the raw score.
        """
        mult = 1.0
        offset = 0.0

        # Known scheduled partner reporting cycle: discount, but DO NOT early-return.
        in_partner_cycle = any(((r.get("meta") or {}).get("cycle") == "partner_reporting") for r in recent)
        if in_partner_cycle:
            has_ticket = any(bool((r.get("meta") or {}).get("ticket_id")) for r in recent)
            if has_ticket:
                mult *= 0.70
                offset += -0.60
                reasons.append("policy:partner_reporting_ticket")
            else:
                mult *= 0.85
                offset += -0.30
                reasons.append("policy:partner_reporting_no_ticket")

        # External email allowlist policy.
        approved = self.model.user_profile.get(actor_id, {}).get("approved_partners", set()) or set()

        external_emails = []
        for r in recent:
            if r.get("event_type") != "email_send":
                continue
            dst = r.get("dst") or ""
            if self._is_external(dst):
                external_emails.append(r)

        if external_emails:
            has_unapproved = False
            has_approved = False
            for em in external_emails:
                dst = em.get("dst") or ""
                approved_flag = bool((em.get("meta") or {}).get("approved_partner", False))
                if (dst in approved) or approved_flag:
                    has_approved = True
                else:
                    has_unapproved = True

            if has_unapproved:
                feats["unapproved_partner"] = 1.0
                reasons.append("policy:unapproved_partner")
            elif has_approved:
                feats["anchor_email"] = 0.0
                reasons.append("policy:approved_partner_email_not_anchor")

        # After-hours policy mismatch
        allowed_after = bool(self.model.user_profile.get(actor_id, {}).get("allowed_after_hours", False))
        if feats.get("after_hours", 0.0) == 1.0 and not allowed_after:
            mult *= 1.15
            offset += 0.50
            reasons.append("policy:after_hours_not_allowed")

        return mult, offset
    # -------- baseline layer (EWMA z-score) --------
    def baseline_anomaly(self, actor_id: int, feats: Dict[str, float], reasons: List[str]) -> float:
        baseline_keys = ["export_large", "export_small", "staging_export", "after_hours", "sens_read_burst", "login_burst", "email_burst"]

        a = actor_id
        self.mu.setdefault(a, {k: 0.0 for k in baseline_keys})
        self.var.setdefault(a, {k: 1.0 for k in baseline_keys})

        zsum = 0.0
        eps = 1e-6

        for k in baseline_keys:
            x = float(feats.get(k, 0.0))
            m = self.mu[a][k]
            v = self.var[a][k]

            z = (x - m) / (math.sqrt(v) + eps)
            z = max(0.0, min(self.cfg.z_clip, z))
            zsum += z

            # update EWMA mean/var
            alpha = self.cfg.ewma_alpha
            new_m = (1 - alpha) * m + alpha * x
            self.mu[a][k] = new_m

            beta = self.cfg.ewma_var_alpha
            dv = (x - new_m) ** 2
            self.var[a][k] = (1 - beta) * v + beta * dv

        if zsum > 0.0:
            reasons.append(f"baseline:zsum={zsum:.2f}")

        return zsum * self.cfg.baseline_weight

    # -------- trust layer --------
    def confirmed_threshold(self, actor_id: int) -> float:
        base = self.cfg.base_confirmed_threshold
        if not self.cfg.use_trust:
            return base
        t = self.trust.get(actor_id, self.cfg.trust_init)
        return base + self.cfg.trust_slope * (t - 0.5)

    def decay_trust(self):
        if not self.cfg.use_trust:
            return
        d = float(self.cfg.trust_decay)
        if d <= 0.0:
            return
        for a, t in list(self.trust.items()):
            self.trust[a] = _clip((1 - d) * t + d * self.cfg.trust_init, self.cfg.trust_min, self.cfg.trust_max)

    def update_trust(self, actor_id: int, is_malicious: bool):
        if not self.cfg.use_trust:
            return
        t = self.trust.get(actor_id, self.cfg.trust_init)
        t += (self.cfg.trust_tp_delta if is_malicious else self.cfg.trust_fp_delta)
        self.trust[actor_id] = _clip(t, self.cfg.trust_min, self.cfg.trust_max)

    # -------- online learning layer --------
    def online_update(self, feats: Dict[str, float], y: int):
        if not self.cfg.use_online_learning:
            return

        # logistic score
        s = 0.0
        for k, w in self.cfg.w.items():
            s += w * float(feats.get(k, 0.0))
        p = 1.0 / (1.0 + math.exp(-s))

        for k in self.cfg.w.keys():
            x = float(feats.get(k, 0.0))
            grad = (p - y) * x + self.cfg.l2 * self.cfg.w[k]
            self.cfg.w[k] -= self.cfg.lr * grad
            self.cfg.w[k] = _clip(self.cfg.w[k], -self.cfg.w_clip, self.cfg.w_clip)

    # -------- utility --------
    def _can_fire(self, actor_id: int, tier: str, step: int) -> bool:
        last = self.last_alert_step.get((actor_id, tier), -10**9)
        return (step - last) >= self.cfg.cooldown

    def _score(self, feats: Dict[str, float]) -> float:
        return sum(self.cfg.w[k] * float(feats.get(k, 0.0)) for k in self.cfg.w)

    def correlate(self):
        s = self.model.steps
        self.decay_trust()

        # update history from ingest stream
        for ing in self.model.ingest_events:
            user = ing.actor_id
            rec = (ing.meta or {}).get("record", {})
            if not rec:
                continue

            self.recent_by_user.setdefault(user, []).append(rec)
            self.recent_by_user[user] = self.recent_by_user[user][-2000:]

        # ML training during warmup (or if not freezing)
        phase = getattr(self.model, "phase", "test")
        train_ml_now = (self.cfg.use_ml and (phase == "train" or not self.cfg.ml_freeze_after_warmup))
        if train_ml_now and (s % self.cfg.ml_train_sample_every == 0):
            for user, hist in self.recent_by_user.items():
                recent = hist[-self.cfg.window:]
                if not recent:
                    continue
                role = self.model.user_profile.get(user, {}).get("role", "staff")
                vec = self.ml_vector(recent)
                self.ml.add(role, vec, user, recent)
                self.ml.maybe_fit(role, s)

        # score whenever anchor seen in this step
        for ing in self.model.ingest_events:
            user = ing.actor_id
            rec = (ing.meta or {}).get("record", {})

            if not rec:

                continue


            email_anchor = self._is_email_anchor(rec)
            login_anchor = self._is_login_anchor(rec)
            if not (email_anchor or login_anchor):
                continue

            recent = self.recent_by_user.get(user, [])[-self.cfg.window:]
            reasons: List[str] = []

            feats = self.extract_features(recent, user)
            feats["anchor_email"] = 1.0 if email_anchor else 0.0
            feats["anchor_login"] = 1.0 if login_anchor else 0.0

            # base score (rules/weights)
            score = self._score(feats)

            # policy adjustments (may also suppress anchor_email)
            mult, offset = 1.0, 0.0
            if self.cfg.use_policy:
                mult, offset = self.apply_policy(user, recent, feats, reasons)

            # If policy suppressed the only anchor, skip correlation entirely.
            if feats.get("anchor_email", 0.0) == 0.0 and feats.get("anchor_login", 0.0) == 0.0:
                reasons.append("policy:anchor_suppressed")
                continue

            # apply policy scaling/offset
            score = (score + offset) * mult

            # baseline anomaly
            if self.cfg.use_baseline:
                score += self.baseline_anomaly(user, feats, reasons)

            # thresholds (before ML, for "borderline" gating)
            early_thr = self.cfg.early_threshold
            conf_thr = self.confirmed_threshold(user)

            # keep the score before ML so we can do tier-specific nudging
            score_base = score
            score_early = score_base
            score_conf = score_base

            reasons_early = list(reasons)
            reasons_conf = list(reasons)

            # Fix 3: don't let ML add risk during known partner_reporting cycles
            in_partner_cycle = any(((r.get("meta") or {}).get("cycle") == "partner_reporting") for r in recent)

            # ML anomaly: influence EARLY more than CONFIRMED, and only move borderline cases
            if self.cfg.use_ml:
                if in_partner_cycle:
                    reasons_early.append("ml:skipped_partner_cycle")
                    reasons_conf.append("ml:skipped_partner_cycle")
                else:
                    role = self.model.user_profile.get(user, {}).get("role", "staff")
                    vec = self.ml_vector(recent)
                    ml_raw = float(self.ml.score(role, vec))

                    reasons_early.append(f"ml:raw={ml_raw:.3f}")
                    reasons_conf.append(f"ml:raw={ml_raw:.3f}")

                    # Use a small gate: only "excess" over threshold counts as risk
                    ml_excess = max(0.0, ml_raw - self.cfg.ml_score_threshold)

                    if ml_excess > 0.0:
                        near_early = (score_base >= (early_thr - self.cfg.ml_margin))
                        near_conf  = (score_base >= (conf_thr  - self.cfg.ml_margin))

                        if near_early:
                            term = self.cfg.ml_weight * ml_excess
                            score_early += term
                            reasons_early.append(f"ml:risk={term:.2f}")

                        # CONFIRMED gets less ML influence
                        if near_conf and self.cfg.ml_weight_confirmed_scale > 0.0:
                            term = (self.cfg.ml_weight * self.cfg.ml_weight_confirmed_scale) * ml_excess
                            score_conf += term
                            reasons_conf.append(f"ml:risk={term:.2f}")

            # EARLY tier
            if score_early >= early_thr and self._can_fire(user, "early", s):
                self.model.emit_alert(Event(
                    step=s, event_type="alert_early", actor_id=user,
                    meta={"score": score_early, "threshold": early_thr, "window": self.cfg.window,
                          "reasons": list(reasons_early), "feats": dict(feats),
                          "policy_mult": mult, "policy_offset": offset,
                          "trust": self.trust.get(user, self.cfg.trust_init),
                          "anchor": "email" if feats.get("anchor_email", 0.0) == 1.0 else "login",
                          "score_base": score_base},
                    label=rec.get("label"),
                    scenario=rec.get("scenario")
                ))
                self.last_alert_step[(user, "early")] = s

            # Fix 4: CONFIRMED must be stricter than EARLY (evidence gate)
            confirmed_gate = (
                feats.get("unapproved_partner", 0.0) == 1.0
                or feats.get("export_large", 0.0) == 1.0
                or feats.get("anchor_login", 0.0) == 1.0
            )

            if confirmed_gate and score_conf >= conf_thr and self._can_fire(user, "confirmed", s):
                self.model.emit_alert(Event(
                    step=s, event_type="alert_confirmed", actor_id=user,
                    meta={"score": score_conf, "threshold": conf_thr, "base_threshold": self.cfg.base_confirmed_threshold,
                          "window": self.cfg.window, "reasons": list(reasons_conf), "feats": dict(feats),
                          "policy_mult": mult, "policy_offset": offset,
                          "trust": self.trust.get(user, self.cfg.trust_init),
                          "anchor": "email" if feats.get("anchor_email", 0.0) == 1.0 else "login",
                          "weights": dict(self.cfg.w),
                          "confirmed_gate": True,
                          "score_base": score_base},
                    label=rec.get("label"),
                    scenario=rec.get("scenario")
                ))
                self.last_alert_step[(user, "confirmed")] = s

                is_mal = (rec.get("label") == "malicious")
                self.update_trust(user, is_mal)

                # online update uses ground truth (simulated analyst verdict)
                y = 1 if is_mal else 0
                self.online_update(feats, y)

                self.model.event_log.append(Event(
                    step=s, event_type="analyst_verdict", actor_id=user,
                    meta={"verdict": "malicious" if is_mal else "benign",
                          "used_online_learning": self.cfg.use_online_learning},
                    label=rec.get("label"),
                    scenario=rec.get("scenario")
                ))

class InsiderModel(mesa.Model):
    def __init__(
        self,
        n_employees: int = 30,
        n_power_users: int = 4,
        n_malicious_exfil: int = 3,
        n_malicious_stealth: int = 2,
        n_malicious_acct_takeover: int = 1,
        n_malicious_staging_exfil: int = 1,
        n_malicious_email_only: int = 1,
        seed: int = 42,
        siem_cfg: Optional[SIEMConfig] = None,
        warmup_steps: int = 60,
    ):
        super().__init__(seed=seed)

        # Mesa 2.x does not guarantee a public `steps` counter on Model;
        # track our own simulation step count for warmup/test transitions.
        self.steps = 0

        self.action_events: List[Event] = []
        self.ingest_events: List[Event] = []
        self.alert_events: List[Event] = []
        self.event_log: List[Event] = []

        # per-actor context profiles (policy layer uses this)
        self.user_profile: Dict[int, Dict[str, Any]] = {}

        self.siem_cfg = siem_cfg or SIEMConfig()

        # train/test phase
        self.warmup_steps = int(warmup_steps)
        self.phase = "train"
        self.attack_enabled = False

        # benign employees
        for _ in range(n_employees):
            role = self.random.choices(["staff", "analyst", "admin"], weights=[0.78, 0.18, 0.04], k=1)[0]
            BenignEmployee(self, role=role)

        # adversarial benign power users
        for _ in range(n_power_users):
            BenignPowerUser(
                self,
                role="analyst",
                report_every=self.random.choice([24, 36, 48]),
                fp_probability=0.70,
                after_hours_probability=0.30,
                ticket_missing_probability=0.15,
                new_partner_probability=0.12,
                ad_hoc_external_probability=0.02
            )

        # malicious: exfil
        base_start = 16 + self.warmup_steps
        gap = 12
        for i in range(n_malicious_exfil):
            MaliciousInsider(self, scenario="exfil", start_step=base_start + i * gap, repeat_every=48)

        # malicious: stealth
        stealth_base = 28 + self.warmup_steps
        stealth_gap = 18
        for j in range(n_malicious_stealth):
            MaliciousInsider(self, scenario="stealth", start_step=stealth_base + j * stealth_gap, repeat_every=48)

        # malicious: account takeover
        takeover_base = 22 + self.warmup_steps
        takeover_gap = 20
        for k in range(n_malicious_acct_takeover):
            MaliciousInsider(self, scenario="acct_takeover", start_step=takeover_base + k * takeover_gap, repeat_every=None)

        # malicious: staging exfil
        staging_base = 34 + self.warmup_steps
        staging_gap = 24
        for k in range(n_malicious_staging_exfil):
            MaliciousInsider(self, scenario="staging_exfil", start_step=staging_base + k * staging_gap, repeat_every=72)

        # malicious: email-only
        email_only_base = 18 + self.warmup_steps
        email_only_gap = 26
        for k in range(n_malicious_email_only):
            MaliciousInsider(self, scenario="email_only", start_step=email_only_base + k * email_only_gap, repeat_every=None)

        # monitors + SIEM (Mesa 3.x: just pass self, unique_id is auto-assigned)
        DBMonitor(self)
        EmailMonitor(self)
        AuthMonitor(self)
        SIEMAgent(self, cfg=self.siem_cfg)

        self.datacollector = DataCollector(
            model_reporters={
                "alerts_this_step": lambda m: len(m.alert_events),
                "malicious_actions_this_step": lambda m: sum(1 for e in m.action_events if e.label == "malicious"),
                "total_events_logged": lambda m: len(m.event_log),
                "phase": lambda m: m.phase,
            }
        )

    # context profile registration
    def register_user(self, actor_id: int, role: str, allowed_after_hours: bool, approved_partners: Set[str]):
        self.user_profile.setdefault(actor_id, {})
        self.user_profile[actor_id]["role"] = role
        self.user_profile[actor_id]["allowed_after_hours"] = allowed_after_hours
        self.user_profile[actor_id]["approved_partners"] = set(approved_partners)

    def _tag_phase(self, e: Event):
        e.phase = self.phase
        if e.meta is None:
            e.meta = {}
        e.meta["phase"] = self.phase

    # event emission
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
        # advance simulation step counter
        self.steps += 1

        # switch from warmup -> test
        if self.steps >= self.warmup_steps:
            self.phase = "test"
            self.attack_enabled = True

        self.action_events = []
        self.ingest_events = []
        self.alert_events = []

        # Stage 1: user activity
        self.agents_by_type[BenignEmployee].shuffle_do("act")
        self.agents_by_type[BenignPowerUser].shuffle_do("act")
        self.agents_by_type[MaliciousInsider].shuffle_do("act")

        # Stage 2: monitoring
        self.agents_by_type[DBMonitor].do("monitor")
        self.agents_by_type[EmailMonitor].do("monitor")
        self.agents_by_type[AuthMonitor].do("monitor")

        # Stage 3: SIEM correlation
        self.agents_by_type[SIEMAgent].do("correlate")

        self.datacollector.collect(self)


# =========================
# Run + Output
# =========================
def run(
    T: int = 240,
    out_jsonl: str = "events.jsonl",
    siem_cfg: Optional[SIEMConfig] = None,
    warmup_steps: int = 60,
):
    model = InsiderModel(
        n_employees=30,
        n_power_users=4,
        n_malicious_exfil=3,
        n_malicious_stealth=2,
        n_malicious_acct_takeover=1,
        n_malicious_staging_exfil=1,
        n_malicious_email_only=1,
        seed=42,
        siem_cfg=siem_cfg,
        warmup_steps=warmup_steps,
    )

    for _ in range(T):
        model.step()

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for e in model.event_log:
            f.write(json.dumps(e.to_dict()) + "\n")

    c = Counter(e.event_type for e in model.event_log)
    phase_counts = Counter((e.phase or (e.meta or {}).get("phase")) for e in model.event_log)

    alerts = [e for e in model.event_log if e.event_type in {"alert_early", "alert_confirmed"}]
    tp = sum(1 for e in alerts if e.label == "malicious")
    fp = sum(1 for e in alerts if e.label != "malicious")

    print("event_type counts:", c)
    print(f"phase events: " + " ".join(f"{k}={v}" for k, v in phase_counts.items()) + f" total={len(model.event_log)}")
    print("alerts:", len(alerts), f"(TP={tp}, FP={fp})")
    print(f"Wrote {len(model.event_log)} events to {out_jsonl}")


if __name__ == "__main__":
    run(T=240, out_jsonl="events.jsonl", siem_cfg=SIEMConfig(), warmup_steps=60)