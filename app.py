import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pipeline import classify_all, extract_items, detect_sensitive, classify_message, mask_sensitive

st.set_page_config(page_title="Message Intelligence System", page_icon="🧠", layout="wide")

st.title("🧠 Message Intelligence System")
st.caption("Local, privacy-first message classification, task/event extraction, and sensitive-data masking")

st.info(
    "The demo processes the uploaded CSV locally in this app. "
    "Raw sensitive values are masked before they are displayed."
)

uploaded = st.file_uploader("Upload messages.csv", type=["csv"])

if uploaded is None:
    st.markdown("""
### What this demo does
1. Preserves chronological message order.
2. Classifies every message into six categories.
3. Extracts tasks and meetings/events without inventing missing fields.
4. Detects sensitive information and masks values.
5. Shows explainable reasons and heuristic confidence scores.

**Privacy rule:** Do not upload the original dataset to a public repository.
""")
    st.stop()

df = pd.read_csv(uploaded)
required = {"message_id", "timestamp", "sender", "message"}
if not required.issubset(df.columns):
    st.error(f"CSV must contain: {sorted(required)}")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

classifications = classify_all(df)
cls = pd.DataFrame([x.__dict__ for x in classifications])
items = pd.DataFrame([x.__dict__ for x in extract_items(df)])
sensitive = pd.DataFrame([x.__dict__ for x in detect_sensitive(df)])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Messages", len(df))
m2.metric("Tasks / Events", len(items))
m3.metric("Sensitive", len(sensitive))
m4.metric("Categories", cls["category"].nunique())

tab1, tab2, tab3, tab4 = st.tabs(
    ["Classification", "Tasks & Events", "Sensitive Detection", "Mandatory IDs"]
)

with tab1:
    st.subheader("All 900 messages")
    counts = cls["category"].value_counts().rename_axis("category").reset_index(name="count")
    st.bar_chart(counts.set_index("category"))
    st.dataframe(cls, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Extracted tasks and events")
    if len(items):
        st.dataframe(items, use_container_width=True, hide_index=True)
    else:
        st.warning("No task/event items detected.")

with tab3:
    st.subheader("Sensitive information — masked view only")
    if len(sensitive):
        st.dataframe(sensitive, use_container_width=True, hide_index=True)
    else:
        st.success("No sensitive findings.")

with tab4:
    st.subheader("Mandatory demonstration IDs")
    demo_text = st.text_input(
        "Paste the 15 IDs separated by commas",
        value="MSG_0002,MSG_0007,MSG_0001,MSG_0003,MSG_0009,MSG_0016,MSG_0004,MSG_0006,MSG_0014,MSG_0015,MSG_0012,MSG_0024,MSG_0037,MSG_0013,MSG_0005"
    )
    ids = [x.strip() for x in demo_text.split(",") if x.strip()]
    selected = df[df["message_id"].isin(ids)].copy()
    selected["category"] = selected["message"].apply(lambda x: classify_message(x)[0])
    selected["confidence"] = selected["message"].apply(lambda x: classify_message(x)[1])
    selected["reason"] = selected["message"].apply(lambda x: classify_message(x)[2])
    selected["display_message"] = selected.apply(
        lambda r: mask_sensitive(r["message"]) if r["message"] else "", axis=1
    )
    display_cols = ["message_id", "timestamp", "sender", "display_message", "category", "confidence", "reason"]
    st.dataframe(selected[display_cols], use_container_width=True, hide_index=True)

st.divider()
st.caption("Important: confidence scores are heuristic, not calibrated probabilities. Missing dates, times, or people remain null/unresolved.")
