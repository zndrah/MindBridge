import argparse
import json
import re
import hashlib
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Comment


DEFAULT_INPUT = Path("data/raw/counselchat/cleaned_counselchat_data.csv")
DEFAULT_OUTPUT = Path("data/processed/counselchat")
DATASET_NAME = "elvindavis/counselchat"


def clean_html(value):
    """Remove non-content HTML while preserving readable structure."""
    if pd.isna(value):
        return ""

    soup = BeautifulSoup(str(value), "html.parser")

    # Remove comments, including Microsoft Word metadata.
    for comment in soup.find_all(string=lambda item: isinstance(item, Comment)):
        comment.extract()

    # Remove non-text content.
    for tag in soup.find_all(["img", "script", "style", "iframe", "noscript"]):
        tag.decompose()

    # Preserve useful block boundaries and list structure.
    for tag in soup.find_all("br"):
        tag.replace_with("\n")

    for tag in soup.find_all("li"):
        tag.insert_before("\n• ")

    for tag in soup.find_all(
        ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]
    ):
        tag.insert_before("\n")
        tag.insert_after("\n")

    # Remove attributes such as Base64 image data and inline CSS.
    for tag in soup.find_all(True):
        tag.attrs = {}

    text = soup.get_text("", strip=False)
    text = text.replace("\xa0", " ")

    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)

    return "\n".join(lines)


def normalize_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def process_dataset(input_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required_columns = {
        "questionTitle",
        "questionText",
        "topics",
        "answerText",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    records = []
    seen_rows = set()

    for index, row in df.iterrows():
        title = normalize_text(row["questionTitle"])
        question = normalize_text(row["questionText"])
        topics = normalize_text(row["topics"])
        answer = clean_html(row["answerText"])

        flags = []

        if not title and not question:
            flags.append("missing_question_title_and_text")
        elif not question:
            flags.append("missing_question_text")

        if not answer:
            flags.append("missing_answer")
        elif len(answer) < 100:
            flags.append("short_answer_review")

        if len(answer) > 10000:
            flags.append("unusually_long_cleaned_answer_review")

        if not topics:
            flags.append("missing_topic_review")

        # Flag exact duplicate source rows without deleting them.
        row_signature = hashlib.sha256(
            json.dumps(
                [title, question, topics, answer],
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

        if row_signature in seen_rows:
            flags.append("exact_duplicate_record_review")
        seen_rows.add(row_signature)

        records.append({
            "record_id": f"counselchat_{index:04d}",
            "source_dataset": DATASET_NAME,
            "source_row_index": int(index),
            "question_title": title,
            "question_text": question,
            "topics": [topic.strip() for topic in topics.split(",") if topic.strip()],
            "answer_text": answer,
            "answer_characters": len(answer),
            "review_status": "pending_review",
            "quality_flags": flags,
        })

    jsonl_path = output_dir / "counselchat_cleaned.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    flag_counts = {}
    for record in records:
        for flag in record["quality_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    repeated_questions = (
        pd.Series([
            record["question_text"]
            for record in records
            if record["question_text"]
        ])
        .value_counts()
    )

    audit = {
        "dataset": DATASET_NAME,
        "input_file": str(input_path),
        "output_file": str(jsonl_path),
        "source_records": len(df),
        "output_records": len(records),
        "records_removed": 0,
        "unique_question_texts": int(
            len({r["question_text"] for r in records if r["question_text"]})
        ),
        "repeated_question_groups": int(
            (repeated_questions > 1).sum()
        ),
        "records_with_quality_flags": sum(
            bool(record["quality_flags"]) for record in records
        ),
        "quality_flag_counts": flag_counts,
        "clinical_approval": False,
        "all_records_pending_review": True,
        "faiss_indexed": False,
    }

    audit_path = output_dir / "counselchat_audit.json"

    with audit_path.open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2, ensure_ascii=False)

    print("\n=== COUNSELCHAT CLEANING COMPLETE ===")
    print(f"Source records:       {len(df)}")
    print(f"Output records:       {len(records)}")
    print("Records removed:      0")
    print(f"Repeated question groups: {audit['repeated_question_groups']}")
    print(f"Records flagged:      {audit['records_with_quality_flags']}")
    print(f"\nJSONL: {jsonl_path}")
    print(f"Audit: {audit_path}")
    print("\nReview status: All records remain pending_review.")
    print("Clinical approval: None.")
    print("FAISS indexing: Not performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean and audit the CounselChat dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    process_dataset(args.input, args.output)
