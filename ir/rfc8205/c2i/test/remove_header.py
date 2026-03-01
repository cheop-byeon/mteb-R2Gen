import json
import argparse
import sys
from pathlib import Path


def process_file(input_path: str, output_path: str):
    """Process JSONL file and remove header from text field."""
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        processed = 0
        skipped = 0
        
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                skipped += 1
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {lineno}: {e}", file=sys.stderr)
                skipped += 1
                continue

            # Get text, ensuring it's a string
            text = obj.get("text", "")
            if not isinstance(text, str):
                text = "" if text is None else str(text)

            # Remove first line (header) if present
            text_lines = "\n".join(text.split("\n", 1)[1:]) if "\n" in text else ""
            
            # Build output object; keep id/title for traceability
            out = {
                "_id": obj.get("_id"),
                "title": obj.get("title"),
                "text": text_lines
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            processed += 1
        
        print(f"Processed: {processed} records, Skipped: {skipped} records", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Remove header (first line) from text field in JSONL file")
    parser.add_argument("--input", type=str, default="corpus.386.jsonl", help="Input JSONL file")
    parser.add_argument("--output", type=str, default="corpus.386.no.head.jsonl", help="Output JSONL file")
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    # Verify input file exists
    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Process file
    process_file(input_path, output_path)


if __name__ == "__main__":
    main()

