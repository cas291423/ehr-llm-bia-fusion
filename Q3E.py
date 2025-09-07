from gradio_client import Client
import pandas as pd
import numpy as np
import json
import re

# -------------------------------
# Config
# -------------------------------
EMB_MODEL = "Qwen3-Embedding-4B"  # use 4B model
EMB_DIM = 32                      # embedding dimension (LLM guarantees 32)
INPUT_XLSX = "sampled_data_share.xlsx"
OUTPUT_XLSX = "embedding_sampled_data_share.xlsx"
MAX_TEXT_SENT = 23                # keep at most 23 sentences
ZERO_VEC = [0.0] * EMB_DIM        # fallback vector only for exceptions


def normalize_embedding(result):
    """
    Normalize gradio 'result' to a flat Python list[float] with shape (EMB_DIM,).
    LLM is assumed to output exactly EMB_DIM; we only flatten.
    """
    if isinstance(result, dict):
        for k in ("embedding", "vector", "data", "emb"):
            if k in result:
                result = result[k]
                break
    arr = np.asarray(result, dtype=float).reshape(-1)
    return arr.tolist()


def split_text_by_periods(text):
    """
    Split text by Chinese and English periods ('。' and '.').
    Keep non-empty trimmed sentences; do not include the period chars.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    s = str(text)
    parts = re.split(r"[。\.]", s)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def embed_once(client, content):
    """
    Call embedding endpoint and return list[float] of length EMB_DIM.
    No pad/truncate here since the LLM guarantees EMB_DIM.
    """
    result = client.predict(
        model_name=EMB_MODEL,
        query_text=content,
        dim=EMB_DIM,
        api_name="/predict"
    )
    vec = normalize_embedding(result)
    return vec


# -------------------------------
# Main
# -------------------------------
client = Client("http://127.0.0.1:7860/")
data = pd.read_excel(INPUT_XLSX)
df = pd.DataFrame(data)

# Basic info (quick sanity checks)
print(df.isnull().sum())
feature_columns = data.columns
print(feature_columns)

# ---------------------------------------------
# Fixed schema per your requirement:
# - First 31 columns: structured features
# - 32nd column: long text column
# - Last column: label (keep as is)
# ---------------------------------------------
assert len(feature_columns) >= 33, "Expect at least 33 columns: 31 structured + 1 text + 1 label."

structured_cols = list(feature_columns[:31])
text_col = feature_columns[31]     # 32nd column (0-based index 31)
label_col = feature_columns[-1]    # last column is label, kept unchanged

print("Structured columns:", structured_cols)
print("Text column:", text_col)
print("Label column:", label_col)

# ---------------------------------------------
# Create new columns to store embeddings
# For structured features: one embedding column per feature: {col}_emb
# For text: exactly 23 sentence columns: {text_col}_sent{i} (i=1..23)
#   - Real sentence -> store 32-dim vector (JSON string)
#   - Padded/masked -> store literal "MASK" (no model call)
# ---------------------------------------------
for c in structured_cols:
    emb_col = f"{c}_emb"
    if emb_col not in data.columns:
        data[emb_col] = None

text_sent_cols = []
for i in range(1, MAX_TEXT_SENT + 1):
    col_name = f"{text_col}_sent{i}"
    text_sent_cols.append(col_name)
    if col_name not in data.columns:
        data[col_name] = None

# ---------------------------------------------
# Iterate rows to compute embeddings
# - Structured: each column -> embed its cell value (cast to string)
# - Text: split by '。' and '.'; embed each real sentence up to 23;
#         for padded positions, write "MASK" (no model call)
# - Label: untouched
# ---------------------------------------------
for index, row in data.iterrows():
    # ---- Structured features ----
    for c in structured_cols:
        try:
            content = row[c]
            if content is None or (isinstance(content, float) and np.isnan(content)):
                content = ""
            vec = embed_once(client, str(content))
            data.at[index, f"{c}_emb"] = json.dumps(vec, ensure_ascii=False)
        except Exception as e:
            print(f"[Structured][Row {index}][Col {c}] embedding failed: {e}")
            data.at[index, f"{c}_emb"] = json.dumps(ZERO_VEC, ensure_ascii=False)

    # ---- Text feature ----
    try:
        raw_text = row[text_col] if text_col in data.columns else ""
        sents = split_text_by_periods(raw_text)
        sents = sents[:MAX_TEXT_SENT]

        real_n = len(sents)
        # Real sentences -> embed
        for i in range(real_n):
            col_i = text_sent_cols[i]
            try:
                vec_i = embed_once(client, sents[i])
                data.at[index, col_i] = json.dumps(vec_i, ensure_ascii=False)
            except Exception as e:
                print(f"[Text][Row {index}][Sent {i+1}] embedding failed: {e}")
                data.at[index, col_i] = json.dumps(ZERO_VEC, ensure_ascii=False)

        # Padded positions -> literal "MASK"
        for i in range(real_n, MAX_TEXT_SENT):
            col_i = text_sent_cols[i]
            data.at[index, col_i] = "MASK"

    except Exception as e:
        print(f"[Text][Row {index}] processing failed: {e}")
        # If the whole text processing fails, mark all 23 as "MASK"
        for col_i in text_sent_cols:
            data.at[index, col_i] = "MASK"

# ---------------------------------------------
# Save (label column remains untouched)
# ---------------------------------------------
data.to_excel(OUTPUT_XLSX, index=False)
print(f"Saved to {OUTPUT_XLSX}")
