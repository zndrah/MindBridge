import argparse
import csv
import json
from pathlib import Path

DEFAULT_INPUT = Path(
    "data/processed/counselchat/counselchat_cleaned.jsonl"
)
DEFAULT_OUTPUT = Path(
    "data/processed/counselchat/counselchat_review_queue.csv"
)

REVIEW_STATUSES = {
    "pending_review",
    "approved",
    "restricted",
    "rejected",
}


def prepare_review_queue(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    records = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            topics = record.get("topics", [])
            if not isinstance(topics, list):
                topics = [topics] if topics else []

            quality_flags = record.get("quality_flags", [])
            if not isinstance(quality_flags, list):
                quality_flags = [quality_flags] if quality_flags else []

            records.append({
                "record_id": record["record_id"],
                "source_dataset": record["source_dataset"],
                "source_row_index": record["source_row_index"],
                "question_title": record["question_title"],
                "question_text": record["question_text"],
                "topics": json.dumps(topics, ensure_ascii=False),
                "answer_text": record["answer_text"],
                "quality_flags": json.dumps(quality_flags, ensure_ascii=False),
                "review_status": "pending_review",
                "retrieval_permission": "no",
                "review_notes": "",
                "reviewer": "",
                "reviewed_at": "",
                "clinical_approval": "not_assessed",
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(records[0].keys()) if records else []

    with output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("\n=== COUNSELCHAT REVIEW QUEUE CREATED ===")
    print(f"Records exported: {len(records)}")
    print(f"Output: {output_path}")
    print("Initial status: pending_review")
    print("Retrieval permission: no")
    print("Clinical approval: not assessed")
    print("FAISS indexing: not performed")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a human-review queue for CounselChat."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    prepare_review_queue(args.input, args.output)


if __name__ == "__main__":
    main()
