import re
import requests
import json
import jsonlines
import argparse
from tqdm import tqdm

###############################################################################
# Helpers
###############################################################################

def extract_pubmed_id(url: str) -> str | None:
    """Extract the PubMed ID from a URL such as
    http://www.ncbi.nlm.nih.gov/pubmed/15858239"""
    match = re.search(r"/pubmed/(\d+)", url)
    return match.group(1) if match else None


def fetch_abstract_from_pubmed(pmid: str, api_key: str | None = None) -> str:
    """Fetch the abstract text for a given PMID using NCBI E‑utilities.
    Returns an empty string on failure or if no abstract is found."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
        "rettype": "abstract",
    }
    if api_key:
        params["api_key"] = api_key

    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        xml_text = resp.text
        parts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", xml_text, flags=re.DOTALL)
        parts_clean = [re.sub(r"<[^>]+>", "", p).strip() for p in parts]
        return " ".join(parts_clean).strip()
    except Exception as exc:
        print(f"[WARN] PMID {pmid}: {exc}")
        return ""

###############################################################################
# Main
###############################################################################

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BioASQ‑style data to question/answer/abstract JSONL")
    parser.add_argument("--input", default="data/training13b.json",
                        help="Path to source JSON (top‑level 'questions' list).")
    parser.add_argument("--output", default="new_database.jsonl",
                        help="Destination JSONL file.")
    parser.add_argument("--api_key", default=None,
                        help="Optional NCBI API key for higher rate limits.")
    args = parser.parse_args()

    # Load source JSON
    with open(args.input, "r", encoding="utf-8") as fp:
        src = json.load(fp)
    questions = src.get("questions", [])

    with jsonlines.open(args.output, mode="w") as writer:
        cache: dict[str, str] = {}

        for entry in tqdm(questions, desc="Processing questions"):
            q_text = entry.get("body", "").strip()
            ideal = entry.get("ideal_answer", [])
            answer = "\n".join(s.strip() for s in ideal if s).strip() or "N/A"

            abstracts: dict[str, str] = {}
            for url in entry.get("documents", []):
                pmid = extract_pubmed_id(url)
                if not pmid:
                    continue
                if pmid not in cache:
                    cache[pmid] = fetch_abstract_from_pubmed(pmid, api_key=args.api_key)
                if cache[pmid]:
                    abstracts[pmid] = cache[pmid]

            writer.write({
                "question": q_text,
                "answer": answer,
                "abstract": abstracts
            })

    print(f"✅ New database written to {args.output} ({len(questions)} entries).")


if __name__ == "__main__":
    main()
