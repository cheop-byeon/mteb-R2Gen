import os
import requests
from typing import List
import json
import sys
import argparse

def get_object_by_id(data, search_id):
    return next((item for item in data if item['_id'] == search_id), None)

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY') # or OPENROUTER_API_KEY="your-api-key-here"

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set")

MODEL = "meta-llama/llama-3.3-70b-instruct"

separator = "\n\n=== EMAIL START ===\n\n"
def answer_with_openrouter_llama3(query: str, contexts: list[str]) -> str:
    prompt = f"""
You are a technical analyst specializing in protocol design.
Write a concise, evidence-grounded rationale for the following design decision—focus only on the core reasoning (no opening restatement of the decision, no closing summary).

Design Decision:
{query}

Context:
{separator.join(contexts)}


Requirements:
- Begin directly with the reasoning; do not restate the design decision.
- End naturally when the reasoning is covered; do not add a concluding summary.
- Produce a concise, cohesive narrative that reflects the reasoning in the context, using comparisons of different choices, concrete examples, constraints, trade-offs, historical background, or operational/security considerations when relevant.
- Avoid unnecessary elaboration or filler; keep the explanation succinct.
- Keep the explanation high-level and avoid mentioning specific individuals; if differing viewpoints must be referenced, use neutral phrases such as “one said” or “another noted.”
- Do not invent facts, stakeholders, or external justifications not present in the context.
- Present the narrative in prose (not as a transcript, bullet list, or meeting minutes).
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QA data using RAG and LLM")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., llama_non, qwen_verbose)")
    args = parser.parse_args()
    
    model = args.model
    results_path = f'../results/stage1/test/c2i/rfc8205_{model}_default_predictions.json'
    query_path = './c2i/test/queries.jsonl'
    corpus_path = './c2i/test/corpus.386.jsonl'
    top_k = 3
    pairs = []

    with open(results_path, 'r') as file:
        results = json.load(file)

    with open(query_path, "r") as file:
        queries = [json.loads(line) for line in file]

    with open(corpus_path, "r") as file:
        corpus = [json.loads(line) for line in file]
    
    index = 0
    for qid, scores in results.items():
        docs = []
        index +=1
        print(index)
        top_three = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        query = get_object_by_id(queries, qid)['text']
        for t in top_three:
            docid, score = t
            doc = get_object_by_id(corpus, docid)
            print("doc", doc['_id'])
            docs.append(doc['text'])
        print(len(docs))

        reply = answer_with_openrouter_llama3(query, docs)
        pair = {"query": qid,"reply": reply}
        pairs.append(pair)

    with open(f'qa_data_{model}.jsonl', 'w') as jsonl_file:
        for pair in pairs:
            jsonl_file.write(json.dumps(pair) + '\n')
