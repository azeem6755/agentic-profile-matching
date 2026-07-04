"""
tools/compare_candidates.py
Phase 2, Step 2.2 — Candidate Comparator

Fetches profiles for a list of candidate names from ChromaDB and uses
google/gemma-4-e4b to produce a side-by-side Markdown comparison table.
"""
import json
import sys
from openai import OpenAI
import chromadb

# ── Configuration ──────────────────────────────────────────────────────────────
LM_STUDIO_URL   = "http://localhost:1234/v1"
LLM_MODEL       = "google/gemma-4-e4b"
CHROMA_DB_PATH  = "./data/chroma_db"
COLLECTION_NAME = "resume_collection"

SYSTEM_PROMPT = (
    "You are an expert technical recruiter. You will receive profiles for several candidates. "
    "Produce a clean Markdown comparison table with one row per candidate. "
    "Columns must be: Candidate Name | Role / Division | Years Exp | Key Skills | Education | Notable Strengths | Key Gaps. "
    "After the table, add a short 'Summary' paragraph (2-3 sentences) highlighting the top pick and why."
)


def _get_chroma_collection(chroma_path: str = CHROMA_DB_PATH):
    """Return the ChromaDB collection, raising if it doesn't exist or is empty."""
    if not __import__("os").path.exists(chroma_path):
        raise FileNotFoundError(
            f"ChromaDB not found at '{chroma_path}'. Run resume_rag.py first."
        )
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        col = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found. Run resume_rag.py first."
        )
    if col.count() == 0:
        raise RuntimeError("Collection is empty. Run resume_rag.py first.")
    return col


def _fetch_candidate_profile(collection, candidate_name: str) -> dict | None:
    """
    Pull all chunks for a candidate_name from ChromaDB and aggregate their metadata
    into a single profile dict.
    """
    results = collection.get(
        where={"candidate_name": {"$eq": candidate_name}},
        include=["metadatas", "documents"]
    )
    if not results["ids"]:
        return None

    # Aggregate across all chunks
    metadatas = results["metadatas"]
    documents  = results["documents"]

    # Use the first chunk's metadata as the base (skills/exp are the same across chunks)
    base_meta = metadatas[0]

    # Collect all unique text chunks for full-text context
    all_text = "\n\n".join(documents)

    return {
        "candidate_name":  candidate_name,
        "skills":          base_meta.get("skills", ""),
        "experience_years": base_meta.get("experience_years", 0),
        "education":       base_meta.get("education", "[]"),
        "full_text":       all_text[:3000]  # Truncate to keep prompt manageable
    }


def compare_candidates(
    candidate_names: list[str],
    jd_requirements: dict | None = None,
    client: OpenAI | None = None,
    chroma_path: str = CHROMA_DB_PATH
) -> str:
    """
    Fetch profiles for the given candidate names from ChromaDB and produce
    a side-by-side Markdown comparison using the LLM.

    Args:
        candidate_names:  List of exact candidate_name strings as stored in ChromaDB.
        jd_requirements:  Optional dict with must_have/nice_to_have for gap analysis.
        client:           Optional pre-built OpenAI client.
        chroma_path:      Path to the ChromaDB persistent storage.

    Returns:
        A Markdown string containing the comparison table and summary paragraph.
    """
    if client is None:
        client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

    # ── Fetch profiles from ChromaDB ───────────────────────────
    collection = _get_chroma_collection(chroma_path)
    profiles = []
    not_found = []

    for name in candidate_names:
        profile = _fetch_candidate_profile(collection, name)
        if profile:
            profiles.append(profile)
        else:
            not_found.append(name)

    if not_found:
        print(f"[compare_candidates] Warning: not found in DB: {not_found}", file=sys.stderr)

    if not profiles:
        return "No candidate profiles found in the database for the provided names."

    # ── Build prompt ──────────────────────────────────────────
    profiles_text = ""
    for p in profiles:
        edu_parsed = json.loads(p["education"]) if isinstance(p["education"], str) else p["education"]
        edu_str = "; ".join(
            f"{e.get('degree', '')} in {e.get('major', '')} from {e.get('institution', '')}"
            for e in edu_parsed
        ) if edu_parsed else "Not specified"

        profiles_text += (
            f"\n--- Candidate: {p['candidate_name']} ---\n"
            f"Years of Experience: {p['experience_years']}\n"
            f"Skills: {p['skills']}\n"
            f"Education: {edu_str}\n"
            f"Resume Excerpt:\n{p['full_text']}\n"
        )

    jd_context = ""
    if jd_requirements:
        jd_context = (
            f"\nJob Requirements for Gap Analysis:\n"
            f"  Must-Have: {', '.join(jd_requirements.get('must_have', []))}\n"
            f"  Nice-to-Have: {', '.join(jd_requirements.get('nice_to_have', []))}\n"
        )

    user_prompt = (
        f"Compare the following {len(profiles)} candidates:{jd_context}\n{profiles_text}"
    )

    # ── Call LLM ──────────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[compare_candidates] LLM error: {e}", file=sys.stderr)
        return f"Error generating comparison: {e}"


# ── Standalone test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import chromadb as _chroma  # noqa: F401 – just to verify import

    print("Running compare_candidates test...\n")
    print("NOTE: This test requires ChromaDB to be populated (run resume_rag.py first).\n")

    # List all candidate names available in the collection
    try:
        col = _get_chroma_collection()
        # Peek at some names to test with
        sample = col.peek(10)
        names = list({m["candidate_name"] for m in sample["metadatas"]})[:3]
        print(f"Testing with candidates: {names}\n")
    except Exception as e:
        print(f"Skipping live test: {e}")
        sys.exit(0)

    if len(names) < 2:
        print("Not enough candidates in DB to compare. Skipping.")
        sys.exit(0)

    result = compare_candidates(names)
    print(result)
    assert "| " in result, "Expected a Markdown table in output"
    print("\n✅ Test passed.")
