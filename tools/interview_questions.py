"""
tools/interview_questions.py
Phase 2, Step 2.3 — Interview Question Generator

Fetches a candidate's SKILLS and EXPERIENCE chunks from ChromaDB and uses
google/gemma-4-e4b to generate 5-10 targeted screening questions.
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

# JSON Schema for structured output
QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "A list of 5 to 10 targeted interview questions for this specific candidate. "
                "Each question should probe either a claimed skill, a potential gap relative to "
                "the job requirements, or a past experience listed in their resume."
            ),
            "minItems": 5,
            "maxItems": 10
        }
    },
    "required": ["questions"]
}

SYSTEM_PROMPT = (
    "You are a senior technical interviewer. Given a candidate's resume and job requirements, "
    "generate 5 to 10 targeted screening questions. "
    "Your questions must: "
    "(1) verify key skills from their resume, "
    "(2) probe potential gaps against the must-have requirements, "
    "(3) explore real examples from their stated work experience. "
    "Do NOT ask generic questions. Make every question specific to this candidate."
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


def _fetch_candidate_chunks(
    collection, candidate_name: str, section_types: list[str]
) -> tuple[str, dict]:
    """
    Fetch chunks of specific section types for a candidate.

    Returns:
        (combined_text, base_metadata)
    """
    results = collection.get(
        where={"candidate_name": {"$eq": candidate_name}},
        include=["metadatas", "documents"]
    )
    if not results["ids"]:
        return "", {}

    # Filter to relevant sections (skills, experience) for concise context
    relevant_docs  = []
    base_meta = {}

    for doc, meta in zip(results["documents"], results["metadatas"]):
        if not base_meta:
            base_meta = meta
        if meta.get("section_type", "") in section_types:
            relevant_docs.append(doc)

    # Fall back to all chunks if no section-specific ones found
    if not relevant_docs:
        relevant_docs = results["documents"]

    combined = "\n\n".join(relevant_docs)[:4000]  # Limit context size
    return combined, base_meta


def generate_interview_questions(
    candidate_name: str,
    jd_requirements: dict,
    client: OpenAI | None = None,
    chroma_path: str = CHROMA_DB_PATH
) -> list[str]:
    """
    Generate role-specific interview questions for a single candidate.

    Args:
        candidate_name:   Exact candidate_name string as stored in ChromaDB.
        jd_requirements:  Dict with must_have[], nice_to_have[], role_type keys.
        client:           Optional pre-built OpenAI client.
        chroma_path:      Path to the ChromaDB persistent storage.

    Returns:
        A list of 5–10 targeted interview question strings.
    """
    if client is None:
        client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

    # ── Fetch candidate sections from ChromaDB ─────────────────
    collection = _get_chroma_collection(chroma_path)
    resume_text, base_meta = _fetch_candidate_chunks(
        collection, candidate_name,
        section_types=["skills", "experience", "summary"]
    )

    if not resume_text:
        print(f"[interview_questions] Candidate '{candidate_name}' not found in DB.", file=sys.stderr)
        return []

    # ── Build prompt ───────────────────────────────────────────
    must_have_str    = ", ".join(jd_requirements.get("must_have", []))
    nice_to_have_str = ", ".join(jd_requirements.get("nice_to_have", []))
    role_type        = jd_requirements.get("role_type", "the role")

    user_prompt = (
        f"Candidate Name: {candidate_name}\n"
        f"Target Role: {role_type}\n\n"
        f"Job Requirements:\n"
        f"  Must-Have: {must_have_str or 'Not specified'}\n"
        f"  Nice-to-Have: {nice_to_have_str or 'Not specified'}\n\n"
        f"Candidate Resume (Skills & Experience):\n{resume_text}"
    )

    # ── Call LLM ───────────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "interview_questions",
                    "schema": QUESTIONS_SCHEMA
                }
            },
            temperature=0.4,  # Slight creativity for varied questions
            max_tokens=800
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("questions", [])
    except Exception as e:
        print(f"[interview_questions] LLM error: {e}", file=sys.stderr)
        return []


# ── Standalone test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running generate_interview_questions test...\n")
    print("NOTE: This test requires ChromaDB to be populated (run resume_rag.py first).\n")

    SAMPLE_REQUIREMENTS = {
        "role_type":    "Backend Engineer",
        "must_have":    ["Python", "REST APIs", "PostgreSQL", "AWS"],
        "nice_to_have": ["Kubernetes", "Kafka", "Go"]
    }

    # Pick a candidate name dynamically from the DB
    try:
        from tools.compare_candidates import _get_chroma_collection
        col = _get_chroma_collection()
        sample = col.peek(5)
        candidate = sample["metadatas"][0]["candidate_name"]
        print(f"Testing with candidate: '{candidate}'\n")
    except Exception as e:
        print(f"Skipping live test: {e}")
        import sys; sys.exit(0)

    questions = generate_interview_questions(candidate, SAMPLE_REQUIREMENTS)
    if not questions:
        print("No questions returned (DB may be empty). Skipping assertions.")
    else:
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q}")
        assert len(questions) >= 5, f"Expected at least 5 questions, got {len(questions)}"
        print(f"\n✅ Test passed — {len(questions)} questions generated.")
