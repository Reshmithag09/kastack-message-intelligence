from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class Classification:
    message_id: str
    category: str
    confidence: float
    reason: str


@dataclass
class ExtractedItem:
    item_id: str
    type: str
    title: str
    description: str
    date: Optional[str]
    time: Optional[str]
    person: Optional[str]
    priority: Optional[str]
    source_message_id: str


@dataclass
class SensitiveFinding:
    message_id: str
    sensitivity_type: str
    risk: str
    masked_text: str
    recommended_action: str


PREFIX_RE = re.compile(
    r"^(?:hi,?\s*|for today:\s*|important:\s*|quick update:\s*|"
    r"please note:\s*|just checking[—-]\s*|just checking:\s*|"
    r"fyi:\s*|one more thing:\s*|can you help\?\s*)+",
    re.I,
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def sensitive_type(text: str) -> Optional[str]:
    checks = [
        ("one_time_password", r"\botp\b"),
        ("password", r"\bpassword\b"),
        ("authentication_token", r"\b(?:temporary )?access token\b"),
        ("recovery_code", r"\baccount recovery code\b"),
        ("payment_card", r"\bcard number\b"),
        ("bank_account", r"\bbank account number\b"),
        ("government_id", r"\bidentification number\b"),
        ("private_address", r"\bhome address\b"),
        ("phone_number", r"\b(?:phone|mobile) number\b|\bcontact me on\b"),
        ("health_information", r"\b(?:medical|health|test) result\b.*\b(?:deficiency|diagnosis|condition)\b"),
        ("email_address", r"\bemail address\b"),
    ]
    for label, pattern in checks:
        if re.search(pattern, text, re.I):
            return label
    return None


def classify_message(text: str) -> tuple[str, float, str]:
    """Explainable local classifier.

    The dataset has no answer labels, so confidence is a heuristic score,
    not a calibrated probability.
    """
    t = normalize(text)
    st = sensitive_type(t)
    if st:
        return "Sensitive Information", 0.98, f"Contains a sensitive-data signal: {st.replace('_', ' ')}."

    if re.search(r"\b(?:discount|sale|offer|promo|promotion|use code|% off|deal|save \d+%)\b", t, re.I):
        return "Promotional", 0.97, "Contains a promotional offer, discount, sale, or coupon code."

    # Personal preferences/profile facts should not become events just because
    # the word "meeting" or "dinner" appears.
    if re.search(
        r"\b(?:prefer|favourite|favorite|usually|drink|vegetarian|personal note|"
        r"for my profile|my emergency contact|i live near|use dark mode|t-shirt size)\b",
        t, re.I
    ):
        if not re.search(
            r"\b(?:scheduled|calendar update|reminder:|are you available for|"
            r"please join .+ on 20\d{2})\b", t, re.I
        ):
            return "Personal Information", 0.91, "States a personal preference, profile detail, or personal fact."

    # Explicit requests/deadlines take precedence over the mere word "meeting".
    action = (
        r"(?:\bdeadline\b|"
        r"\bby\s+20\d{2}-\d{2}-\d{2}\b|"
        r"\bbefore\s+20\d{2}-\d{2}-\d{2}\b|"
        r"\bplease\s+(?:submit|review|reply|complete|confirm|send|upload|finish|"
        r"renew|pay|call|update|verify|prepare|register|check|fill|bring)\b|"
        r"\bdon['’]t forget\b|"
        r"\bi need you to\b|"
        r"\bis due on\b|"
        r"\bcould you send\b|"
        r"\bplease call\b|"
        r"\bif possible,\s+review\b)"
    )
    if re.search(action, t, re.I):
        return "Action Required", 0.93, "Contains an explicit request, action verb, or deadline."

    # Strong event signals.
    if re.search(r"\b(?:are you available for|scheduled|reminder:|calendar update:)\b", t, re.I):
        return "Meeting or Event", 0.95, "Contains a clear meeting/event signal with scheduling context."

    if re.search(r"\bplease join\b.*\b20\d{2}-\d{2}-\d{2}\b", t, re.I):
        return "Meeting or Event", 0.95, "Explicitly asks the recipient to join a dated event."

    # Unresolved but explicit meeting/event references are still events.
    if re.search(r"\blet us meet\b|\breview could be (?:friday|monday|tuesday|wednesday|thursday|saturday|sunday)\b", t, re.I):
        return "Meeting or Event", 0.88, "Mentions a meeting/event but leaves part of the schedule unresolved."

    # Named event types count when a date/time is actually present.
    if (
        re.search(
            r"\b(?:seminar|workshop|appointment|catch-up|discussion|orientation|"
            r"session|meeting|dinner|conference|sprint planning|team stand-up|"
            r"product demo|project review)\b", t, re.I
        )
        and re.search(r"\b20\d{2}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}\b", t)
    ):
        return "Meeting or Event", 0.93, "Contains a named event type together with scheduling information."

    return "General Information", 0.82, "Provides information or an update without a clear action, event, promotion, or sensitive-data signal."


def extract_date(text: str) -> Optional[str]:
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    return m.group(1) if m else None


def extract_time(text: str) -> Optional[str]:
    m = re.search(r"\b(\d{1,2}:\d{2})\b", text)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{1,2})\s*(AM|PM)\b", text, re.I)
    if m:
        return f"{m.group(1)} {m.group(2).upper()}"
    return None


def extract_title(text: str, item_type: str) -> str:
    t = PREFIX_RE.sub("", normalize(text))
    patterns = [
        r"Calendar update:\s*([^,.;]+)",
        r"Reminder:\s*([^,.;]+?)\s+happens\b",
        r"Please join the\s+(.+?)\s+on\s+20\d{2}-\d{2}-\d{2}",
        r"Are you available for the\s+(.+?)\s+at\s+\d",
        r"The\s+(.+?)\s+is scheduled for\s+20\d{2}-\d{2}-\d{2}",
        r"the\s+(.+?)\s+is scheduled for\s+20\d{2}-\d{2}-\d{2}",
        r"Reminder:\s*([^.;]+)",
    ]
    for p in patterns:
        m = re.search(p, t, re.I)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip(" .,:")
    # Action fallback: use the clause after common request markers.
    for p in [
        r"Please\s+(.+?)(?:\s+by\s+20\d{2}-\d{2}-\d{2}|[.;]|$)",
        r"I need you to\s+(.+?)(?:\s+by\s+20\d{2}-\d{2}-\d{2}|[.;]|$)",
        r"Don't forget to\s+(.+?)(?:;\s*deadline|[.;]|$)",
        r"Can you\s+(.+?)(?:\s+before\s+20\d{2}-\d{2}-\d{2}|[.;]|$)",
        r"(.+?)\s+is due on\s+20\d{2}-\d{2}-\d{2}",
        r"(.+?)\s+when you are free",
    ]:
        m = re.search(p, t, re.I)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip(" .,:")
    return "Unresolved"


def extract_description(text: str, title: str) -> str:
    if title != "Unresolved":
        return normalize(text)
    return normalize(text)


def extract_person(text: str) -> Optional[str]:
    # Only extract a person when explicitly named after a supported phrase.
    patterns = [
        r"\b(?:with|from|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        r"\bcall\s+([A-Z][a-z]+)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return None


def extract_priority(text: str) -> Optional[str]:
    t = text.lower()
    if re.search(r"\burgent\b|\bhigh priority\b|\bcritical\b", t):
        return "high"
    if "important:" in t or t.startswith("important"):
        return "high"
    if "for today:" in t:
        return "high"
    if "deadline" in t or re.search(r"\bby\s+20\d{2}-\d{2}-\d{2}\b", t):
        return "medium"
    return None


def extract_items(df: pd.DataFrame) -> list[ExtractedItem]:
    items = []
    counter = 1
    for row in df.itertuples(index=False):
        text = normalize(row.message)
        category, _, _ = classify_message(text)

        # Extract only from messages that clearly contain a task/event.
        is_event = category == "Meeting or Event"
        is_task = category == "Action Required"
        if not (is_event or is_task):
            continue

        item_type = "event" if is_event else "task"
        title = extract_title(text, item_type)
        date = extract_date(text)
        time = extract_time(text)

        # "before the meeting" has no resolvable date/time; keep null.
        if item_type == "task" and not date:
            date = None
        if item_type == "event" and not date:
            date = None

        items.append(
            ExtractedItem(
                item_id=f"{'EVENT' if is_event else 'TASK'}_{counter:03d}",
                type=item_type,
                title=title,
                description=extract_description(text, title),
                date=date,
                time=time,
                person=extract_person(text),
                priority=extract_priority(text),
                source_message_id=row.message_id,
            )
        )
        counter += 1
    return items


def mask_sensitive(text: str) -> str:
    t = normalize(text)
    replacements = [
        (r"(\bOTP\s+is\s+)([A-Za-z0-9-]+)", r"\1******"),
        (r"(\bpassword\s+)(\S+)", r"\1********"),
        (r"(\b(?:temporary )?access token\s+is\s+)(\S+)", r"\1********"),
        (r"(\baccount recovery code\s+is\s+)(\S+)", r"\1********"),
        (r"(\bcard number\s+is\s+)([0-9][0-9 -]*)", r"\1****************"),
        (r"(\bbank account number\s+)([0-9][0-9 -]*)", r"\1************"),
        (r"(\bidentification number\s+is\s+)(\S+)", r"\1[REDACTED ID]"),
        (r"(\bhome address\s+is\s+)(.+?)(?=[.;]|$)", r"\1[REDACTED ADDRESS]"),
        (r"(\bcontact me on\s+)([0-9+ -]+)", r"\1[REDACTED PHONE]"),
        (r"(\b(?:phone|mobile) number\s+is\s+)([0-9+ -]+)", r"\1[REDACTED PHONE]"),
        (r"(\bemail address\s+is\s+)(\S+)", r"\1[REDACTED EMAIL]"),
        (r"(\btest result says\s+)(.+?)(?=[.;]|$)", r"\1[REDACTED HEALTH DATA]"),
    ]
    for pattern, repl in replacements:
        t = re.sub(pattern, repl, t, flags=re.I)
    return t


def sensitive_risk_and_action(kind: str) -> tuple[str, str]:
    if kind in {"one_time_password", "password", "authentication_token", "recovery_code", "payment_card", "bank_account"}:
        return "high", "do_not_store"
    if kind in {"government_id", "health_information"}:
        return "high", "ask_for_confirmation"
    return "medium", "ask_for_confirmation"


def detect_sensitive(df: pd.DataFrame) -> list[SensitiveFinding]:
    findings = []
    for row in df.itertuples(index=False):
        kind = sensitive_type(row.message)
        if not kind:
            continue
        risk, action = sensitive_risk_and_action(kind)
        findings.append(
            SensitiveFinding(
                message_id=row.message_id,
                sensitivity_type=kind,
                risk=risk,
                masked_text=mask_sensitive(row.message),
                recommended_action=action,
            )
        )
    return findings


def classify_all(df: pd.DataFrame) -> list[Classification]:
    # Explicitly preserve chronological order from the supplied CSV.
    ordered = df.reset_index(drop=True)
    results = []
    for row in ordered.itertuples(index=False):
        category, confidence, reason = classify_message(row.message)
        results.append(Classification(row.message_id, category, confidence, reason))
    return results


def run_pipeline(input_csv: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    required = {"message_id", "timestamp", "sender", "message"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Stable chronological ordering; preserves original order for equal timestamps.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    classifications = classify_all(df)
    items = extract_items(df)
    sensitive = detect_sensitive(df)

    pd.DataFrame([asdict(x) for x in classifications]).to_csv(out/"classifications.csv", index=False)
    pd.DataFrame([asdict(x) for x in items]).to_csv(out/"tasks_events.csv", index=False)
    pd.DataFrame([asdict(x) for x in sensitive]).to_csv(out/"sensitive_findings.csv", index=False)

    summary = {
        "messages_processed": len(df),
        "categories": pd.Series([x.category for x in classifications]).value_counts().to_dict(),
        "tasks_and_events_extracted": len(items),
        "sensitive_messages_detected": len(sensitive),
        "note": "Confidence is heuristic because the supplied dataset contains no ground-truth labels.",
    }
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    run_pipeline(args.input, args.output)
