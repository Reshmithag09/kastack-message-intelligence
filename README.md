# KaStack Labs — Message Intelligence System

## Overview

This project implements a privacy-first message processing system for the KaStack Labs AI/ML Engineer Intern assignment.

The pipeline:

1. Reads the supplied CSV.
2. Sorts messages chronologically using the supplied timestamp.
3. Classifies every message into:
   - Action Required
   - Meeting or Event
   - Personal Information
   - General Information
   - Promotional
   - Sensitive Information
4. Extracts tasks and events.
5. Detects sensitive information and masks it.
6. Produces structured CSV/JSON outputs.
7. Provides a Streamlit demo for the video and cloud deployment.

## Important privacy rule

The original candidate dataset must **not** be committed to GitHub.

Keep `messages.csv` and other supplied dataset files outside the public repository. The `.gitignore` also ignores local data and generated outputs.

## Classification approach

The dataset has no answer labels, so a conventional supervised classifier cannot honestly be trained on ground-truth labels.

Instead, this implementation uses an explainable local hybrid NLP approach:

- High-priority deterministic patterns for sensitive information.
- Promotional lexical signals such as discount/sale/coupon codes.
- Meeting/event signals such as scheduled, reminder, seminar, appointment, and "are you available for".
- Action signals such as deadlines, "please submit", "review", "reply", and "don't forget".
- Personal-information signals such as preferences and profile facts.
- General Information as the fallback category.

The classifier is deliberately deterministic so every decision can be explained in the video.

### Confidence

The confidence value is a **heuristic confidence score**, not a calibrated probability. This is explicitly stated because the supplied dataset has no ground-truth labels.

## Task and event extraction

The extractor uses local regular expressions and phrase patterns to find:

- ISO dates (`YYYY-MM-DD`)
- Times (`HH:MM`, AM/PM)
- Event titles from phrases such as "Calendar update", "Reminder", "Please join", and "scheduled"
- Task titles from explicit requests and deadlines
- Explicitly named people only

It never invents a date, time, person, or deadline. If the value cannot be resolved, the structured field is `null` or the title is `Unresolved`.

Priority is inferred only from explicit urgency markers such as "urgent", "important", "for today", or an explicit deadline; otherwise it is left null.

## Sensitive-information detection

The detector checks for:

- OTPs
- Passwords
- Authentication/access tokens
- Recovery codes
- Payment card numbers
- Bank account numbers
- Government/identification numbers
- Home addresses
- Phone/mobile numbers
- Health/test-result information
- Email addresses

Sensitive values are masked before display in the Streamlit demo.

Recommended actions:

- Credentials, OTPs, tokens, recovery codes, payment/bank data → `do_not_store`
- Personal address/contact data → `ask_for_confirmation`

Never display raw sensitive values in the recording.

## Run locally

Install:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python scripts/run_pipeline.py --input "PATH_TO/messages.csv" --output outputs
```

Generated files:

- `outputs/classifications.csv`
- `outputs/tasks_events.csv`
- `outputs/sensitive_findings.csv`
- `outputs/summary.json`

Run the demo:

```bash
streamlit run app.py
```

## Cloud deployment

The demo is designed for Streamlit Community Cloud.

Push only the code files to GitHub. Do **not** push the supplied dataset.

Deploy the repository as a Streamlit app with:

```text
Main file: app.py
```

The public demo should use a safe sample or an uploaded demonstration file with no raw sensitive values.

## Video demonstration plan

Target length: 7–10 minutes.

1. 0:00–0:45 — Explain the problem and architecture.
2. 0:45–1:20 — Show CSV structure with sensitive values hidden.
3. 1:20–2:30 — Run classification and show all six categories.
4. 2:30–3:30 — Show all 15 mandatory IDs.
5. 3:30–4:30 — Show at least three tasks.
6. 4:30–5:20 — Show at least three meetings/events.
7. 5:20–6:00 — Show one unresolved/missing-field example.
8. 6:00–6:50 — Show sensitive detection, masking, risk, action.
9. 6:50–7:40 — Explain three classification decisions.
10. 7:40–8:20 — Show one uncertain/incorrect result and explain why.
11. 8:20–9:00 — Explain the main code section.
12. 9:00–9:40 — Limitations and improvements.

## AI-tool usage declaration

Suggested disclosure:

> AI-assisted development tools were used for brainstorming, code scaffolding, debugging assistance, and documentation refinement. The final implementation, rule design, data-processing logic, outputs, and demonstration were reviewed and understood by the candidate. The supplied raw dataset was not sent to external AI services.

## Limitations

- No ground-truth labels were supplied, so the system cannot report true accuracy.
- Rule-based NLP can miss paraphrases and ambiguous language.
- Confidence values are heuristic.
- Regex extraction is less robust than a full local language model.
- Person extraction intentionally stays conservative to avoid guessing.

## Future improvements

- Build a manually reviewed labeled subset and train TF-IDF + Logistic Regression.
- Add a local embedding model for semantic similarity.
- Add a calibrated confidence model.
- Expand date/time normalization.
- Add entity recognition using a local NLP model.
- Add unit tests and an evaluation dashboard.
