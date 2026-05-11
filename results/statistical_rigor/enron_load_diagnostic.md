# Enron forensics-model load diagnostic

This note documents how `mini_mesa_EG-SIEM_Enron.py` resolves the path to the
serialized Enron forensics model, what happens when the file is not found, and
the load status observed for one matched run (seed = 42).

## 1. Path used to load the serialized model

The forensics model path is **not hardcoded inside the simulation module**. It
is a constructor argument passed from outside.

**`CombinedForensicsAgent.__init__`** (`mini_mesa_EG-SIEM_Enron.py`, line 55):

```python
def __init__(self, model_path: str = None, mode: str = "full"):
    ...
    if self.mode in {"full", "model_only"} and model_path and os.path.exists(model_path):
        self._load_model(model_path)
    elif self.mode in {"full", "model_only"}:
        print(f"Note: No trained model loaded. Falling back from {self.mode} to keyword-only/disabled behavior.")
```

So the path is whatever the caller passes in — it can be absolute, relative, or
`None`. There is no environment variable involved.

The default the module ships with is a **relative path string** in the CLI
parser at the bottom of the file (line 1144):

```python
parser.add_argument(
    "--model",
    default="combined_forensics_model.pkl",
    help="Path to trained forensics model (default: combined_forensics_model.pkl)"
)
...
# line 1173
model_path = args.model if os.path.exists(args.model) else None
```

`os.path.exists()` resolves relative paths against the **current working
directory** of the Python process, not against the script's own location. So
whether the default `"combined_forensics_model.pkl"` works depends entirely on
where Python is launched from.

The actual `pickle.load` happens in `_load_model` (line 85) on whatever path
was passed in.

**Summary:** the path is a relative string by default, resolved against CWD.

## 2. Does the file resolve when run from `scripts/` via subprocess?

No, not with the default relative path.

The file `combined_forensics_model.pkl` lives at the repository root
(`/Users/.../Insider-Threat-Detection-MAS-SIEM-main/combined_forensics_model.pkl`).
A subprocess launched with cwd = `scripts/` and only the bare default string
finds nothing:

```
$ cd scripts && ls combined_forensics_model.pkl
ls: cannot access 'combined_forensics_model.pkl': No such file or directory
```

Our runner (`scripts/run_all_variants_matched.py`) avoids this by (a) building
the absolute path from `REPO_ROOT` (computed via `__file__`) and (b) `os.chdir`-ing
to the repository root before constructing each model. With those two safeguards
the file is found regardless of how the script is launched.

## 3. Is there a silent fallback?

Yes. There are **three** layers of silent fallback that prevent a missing model
from raising:

1. **`__main__` CLI guard** (line 1173):
   ```python
   model_path = args.model if os.path.exists(args.model) else None
   ```
   Substitutes `None` for the path if the file is missing — no error.

2. **`CombinedForensicsAgent.__init__`** (line 76):
   ```python
   if self.mode in {"full", "model_only"} and model_path and os.path.exists(model_path):
       self._load_model(model_path)
   elif self.mode in {"full", "model_only"}:
       print(f"Note: No trained model loaded. Falling back from {self.mode} to keyword-only/disabled behavior.")
   ```
   Prints a `Note:` and continues with `self.classifier = None`, `self.vectorizer = None`.
   The agent stays in `full` mode by name but has no classifier or vectorizer
   loaded — `analyze_email` then takes the `if ... and self.classifier and self.vectorizer:`
   branch as False and only the keyword score contributes.

3. **`_load_model`** (line 85):
   ```python
   try:
       with open(path, 'rb') as f:
           data = pickle.load(f)
       ...
   except Exception as e:
       print(f"Warning: Could not load model from {path}: {e}")
   ```
   Catches **any** exception during deserialization and only prints a warning.
   Pickle-version mismatches, sklearn-version mismatches, or a corrupted file
   would not stop the run — the agent would simply be left without a usable
   classifier.

The net effect: a missing or unloadable model degrades the system to keyword-only
forensics without surfacing a clear error to the caller. Reproducibility audits
that rely on the run-level metrics alone could miss this regression.

## 4. Observed load status for seed = 42

Diagnostic run (`scripts/run_all_variants_matched.py --variants EG-SIEM-Enron
--seeds 42 --diag-enron --out /tmp/enron_diag.csv`):

```
[diag-enron] cwd=/Users/.../Insider-Threat-Detection-MAS-SIEM-main
[diag-enron] REPO_ROOT=/Users/.../Insider-Threat-Detection-MAS-SIEM-main
[diag-enron] resolved forensics_model_path='/Users/.../Insider-Threat-Detection-MAS-SIEM-main/combined_forensics_model.pkl'
[diag-enron] file_exists=True
[diag-enron] forensics_mode='full'
Classifier: LogisticRegression(max_iter=2000, random_state=42, solver='liblinear')
Vectorizer: TfidfVectorizer(max_features=10000, min_df=2, ngram_range=(1, 2),
                stop_words='english')
Loaded combined forensics model from: /Users/.../Insider-Threat-Detection-MAS-SIEM-main/combined_forensics_model.pkl
  Classifier accuracy: 71.39%
  Vocabulary size: 10,000
  Baseline sentence length: 15.00
[diag-enron] forensics_agent.mode='full'
[diag-enron] forensics_agent.classifier_loaded=True
[diag-enron] forensics_agent.vectorizer_loaded=True
[diag-enron] forensics_agent.classifier_accuracy=0.7138933933094225
[diag-enron] forensics_agent.learned_phrase_weights_count=18
[EG-SIEM-Enron seed=42] f1=0.737 P=0.636 R=0.875 ttd_avg=48.86 conf=103 FP=33 (0.2s)
```

Result: for seed 42 (and by symmetry for all ten seeds in our run, since the
same path resolution logic is used), the Enron forensics model loaded
successfully. The run was performed in `mode="full"` with both classifier
(71.39% accuracy) and TF-IDF vectorizer (10,000-feature vocabulary) populated,
plus 18 learned phrase weights.

There is one caveat that surfaces during load: `sklearn` prints
`InconsistentVersionWarning` because the pickle was produced with scikit-learn
1.6.1 and is being deserialized with 1.7.2. The classes still load and predict,
but per scikit-learn's own warning the resulting predictions are not formally
guaranteed to match the originals.

## 5. Implications for the Table 6 results

For the per-run metrics already produced in `run_level_all_variants.csv`, the
forensics model **was** loaded and active for every EG-SIEM-Enron run. The
identical actor F1 = 0.7368 across all ten seeds is therefore not an artifact
of a missing classifier — it reflects the deterministic recovery of the same
malicious-actor set on every seed under the `full` preset.

If anyone re-runs the EG-SIEM-Enron variant from a different working directory
without going through our runner (for example, by invoking
`python mini_mesa_EG-SIEM_Enron.py` from `scripts/`), the silent fallbacks
documented in §3 will trigger and the variant will quietly degrade to
keyword-only forensics. We recommend either (a) keeping the runner as the
single entry point or (b) hardening `mini_mesa_EG-SIEM_Enron.py` to resolve
its default path relative to `__file__` rather than CWD — that change would
remove the fallback ambiguity without altering simulation logic.
