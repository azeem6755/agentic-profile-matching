"""
tools/extract_requirements.py
Phase 2, Step 2.1 — JD Requirements Extractor

Parses a raw Job Description string into structured must-have /
nice-to-have requirements using google/gemma-4-e4b via LM Studio.
"""
import json
import sys
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
LM_STUDIO_URL = "http://localhost:1234/v1"
LLM_MODEL = "google/gemma-4-e4b"

# JSON Schema for structured output
REQUIREMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "role_type": {
            "type": "string",
            "description": "The job title or role category inferred from the JD (e.g. 'Backend Engineer', 'Marketing Manager')."
        },
        "must_have": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "List of hard requirements that a candidate MUST possess to be considered. "
                "Include mandatory skills, tools, minimum years of experience, required degrees, "
                "or certifications explicitly stated as required."
            )
        },
        "nice_to_have": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "List of optional or preferred qualifications that would strengthen a candidacy "
                "but are NOT blocking. Include items labelled 'preferred', 'bonus', or 'plus'."
            )
        }
    },
    "required": ["role_type", "must_have", "nice_to_have"]
}

SYSTEM_PROMPT = (
    "You are a senior technical recruiter. Given a job description, "
    "classify every stated requirement into either 'must_have' (hard, non-negotiable) "
    "or 'nice_to_have' (preferred, bonus). "
    "Also infer the target role_type. "
    "Output ONLY a valid JSON object that matches the requested schema."
)


def extract_requirements(jd: str, client: OpenAI | None = None) -> dict:
    """
    Parse a raw Job Description string into structured requirements.

    Args:
        jd:     Raw job description text.
        client: Optional pre-built OpenAI client (for reuse within agent).

    Returns:
        {
          "role_type":    "Backend Engineer",
          "must_have":    ["Python", "5+ years of experience", "AWS"],
          "nice_to_have": ["Kubernetes", "Go"]
        }
    """
    if client is None:
        client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

    # Truncate very long JDs to fit model context window
    truncated_jd = jd[:6000]

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Extract requirements from the following job description:\n\n{truncated_jd}"}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "job_requirements",
                    "schema": REQUIREMENTS_SCHEMA
                }
            },
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content)
        # Ensure all keys are present with safe defaults
        return {
            "role_type":    data.get("role_type", "Unknown"),
            "must_have":    data.get("must_have", []),
            "nice_to_have": data.get("nice_to_have", [])
        }
    except Exception as e:
        print(f"[extract_requirements] LLM error: {e}", file=sys.stderr)
        return {"role_type": "Unknown", "must_have": [], "nice_to_have": []}


# ── Standalone test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SAMPLE_JD = """
    We are looking for a Senior Backend Engineer to join our growing infrastructure team.

    Requirements:
    - 5+ years of professional software engineering experience
    - Strong proficiency in Python and at least one of: Go or Java
    - Experience designing and maintaining RESTful APIs and microservices
    - Proficiency with PostgreSQL and Redis
    - Hands-on AWS experience (EC2, S3, Lambda)
    - Familiarity with Docker and Kubernetes

    Nice to have:
    - Experience with Kafka or other event streaming platforms
    - Contributions to open-source projects
    - Prior experience in a fintech environment
    """

    print("Running extract_requirements test...\n")
    result = extract_requirements(SAMPLE_JD)
    print(json.dumps(result, indent=2))

    assert "role_type"    in result, "Missing 'role_type'"
    assert "must_have"    in result, "Missing 'must_have'"
    assert "nice_to_have" in result, "Missing 'nice_to_have'"
    assert len(result["must_have"]) > 0, "must_have list is empty"
    print("\n✅ Test passed.")
