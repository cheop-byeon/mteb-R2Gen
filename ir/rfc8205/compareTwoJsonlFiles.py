
import json

def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def compare_items(itemA, itemB, index):
    """Compare two JSON objects and print differences."""
    print(f"\n=== Item {index} ===")
    
    # Compare _id
    if itemA.get("_id") != itemB.get("_id"):
        print(f"Different _id:\n  A: {itemA.get('_id')}\n  B: {itemB.get('_id')}")
    
    # Compare title
    if itemA.get("title") != itemB.get("title"):
        print(f"Different title:\n  A: {itemA.get('title')}\n  B: {itemB.get('title')}")
    
    # Compare text
    if itemA.get("text") != itemB.get("text"):
        print(f"Different text:\n  A: {itemA.get('text')}\n  B: {itemB.get('text')}")
    
    # If no differences
    if (itemA.get("_id") == itemB.get("_id") and 
        itemA.get("title") == itemB.get("title") and 
        itemA.get("text") == itemB.get("text")):
        print("Items are identical.")


def compare_jsonl_files(fileA, fileB):
    A = load_jsonl(fileA)
    B = load_jsonl(fileB)

    if len(A) != len(B):
        print(f"ERROR: Different number of items! A={len(A)}, B={len(B)}")
        return

    for i, (a, b) in enumerate(zip(A, B), start=1):
        compare_items(a, b, i)


if __name__ == "__main__":
    compare_jsonl_files("rfc8205.386.corpus.jsonl", "./c2i/test/corpus.386.jsonl")

