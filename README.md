# Agentic Profile Matching System

An autonomous, multi-round AI recruiting assistant powered by LangGraph, ChromaDB, and local LLMs via LM Studio.

The system performs intelligent, agentic screening of candidates against a job description. It progresses through multiple screening rounds, narrowing down a large pool of applicants (e.g., 100 candidates) to a final shortlist of the best matches, and provides deep reasoning, gap analysis, and tailored interview questions for each top candidate.

## Architecture Highlights
- **LangGraph Agent**: Manages the conversational flow and multi-round screening logic.
- **RAG via ChromaDB**: Stores and searches resume chunks effectively using embeddings.
- **Agent Tools**:
  - `extract_requirements`: Parses a JD into structured must-haves and nice-to-haves.
  - `job_matcher`: Executes the semantic search and composite scoring (RAG).
  - `compare_candidates`: Produces a Markdown side-by-side comparison table.
  - `interview_questions`: Probes specific candidate gaps and claimed skills.

## Prerequisites & LM Studio Setup

1. **Python 3.10+** is required.
2. **LM Studio** must be installed and running locally.
3. Start the Local Inference Server in LM Studio on port `1234` (`http://localhost:1234/v1`).
4. Load the following models in LM Studio:
   - **LLM**: `google/gemma-4-e4b`
   - **Embeddings**: `text-embedding-embeddinggemma-300m-qat`

## Installation Steps

Clone the repository and set up your virtual environment:

```bash
# Initialize a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Verify your LM Studio connection before proceeding:
```bash
python diagnose.py
```
This script will test the chat completion, JSON schema mode, and embedding generation.

## How to Run the Agent

### 1. Ingest Resumes
Before running the agent, you need to populate the vector database with candidate resumes. If you don't have resumes, generate sample data first:

```bash
# Generate 100 realistic PDF resumes (if not already done)
python generate_resumes_pdf.py

# Ingest them into ChromaDB
python resume_rag.py
```
This will parse the PDFs, extract structured metadata, chunk the text, embed it, and store everything in `./data/chroma_db`.

### 2. Start the Agent
Run the main LangGraph agent:
```bash
python matching_agent.py
```

You will be prompted to paste a Job Description (JD). Press `Enter` twice to submit it.

## Sample Interaction Transcript

```text
============================================================
  Agentic Profile Matching System
  Powered by LangGraph + LM Studio (gemma-4-e4b)
============================================================

Paste the job description below (press Enter twice to submit):
Looking for a DevOps Engineer with strong experience in Kubernetes and Docker. Must have hands-on Terraform and Linux administration. Experience with Jenkins or Ansible is a plus. Minimum 3 years required.

============================================================
  🚀 Agentic Profile Matching — Starting
  JD: Looking for a DevOps Engineer with strong experience in Kubernetes and Docker. Must have hands-on Terraform and Linux ad...
============================================================

📋 Extracting requirements...
  Role: DevOps Engineer | Must-have: 5 items

🔍 [Round 1] Searching resumes (limit=10)...
  [Round 1] RAG search returned 10 candidates.

  📊 [Round 1] Ranked 10 candidates → advancing to round 2.
    1. Nancy Evans  score=77.5
    2. Matthew Alvarez  score=77.2
    3. Michelle Allen  score=77.1
    ...

🔍 [Round 2] Searching resumes (limit=5)...
  [Round 2] Narrowed shortlist 10 → 5.

  📊 [Round 2] Ranked 5 candidates → advancing to round 3.
    1. Nancy Evans  score=77.5
    2. Matthew Alvarez  score=77.2
    3. Michelle Allen  score=77.1
    4. William Cox  score=77.0
    5. Cynthia Williams  score=76.7

📄 Generating final report...
  Analysing Nancy Evans (score=77.5)...
  Analysing Matthew Alvarez (score=77.2)...
  ...
✅ Report complete.

============================================================
# Final Screening Report
**Role:** DevOps Engineer
**Must-Have:** Kubernetes, Docker, Terraform, Linux administration, Minimum 3 years
---

## 1. Nancy Evans  ✅ HIRE
**Match Score:** 77.5/100

**Strengths:**
- Exceptional technical alignment (Kubernetes, Docker, Terraform, Linux, CI/CD).
- Direct evidence of hands-on experience in core requirements.
- Experience significantly exceeds minimum tenure requirement (14 years vs. 3 years).

**Gaps:**
- None identified based on the provided criteria.

**Verdict:** HIRE — The candidate is an excellent fit.

**Suggested Interview Questions:**
1. Can you describe a complex Terraform module you wrote from scratch?
2. How do you handle secrets management in Kubernetes?
...
============================================================

📋 Top 5 candidates:
  1. Nancy Evans  (score: 77.5)
  2. Matthew Alvarez  (score: 77.2)
  3. Michelle Allen  (score: 77.1)
  4. William Cox  (score: 77.0)
  5. Cynthia Williams  (score: 76.7)

💬 Options:
  'compare top N'          — side-by-side comparison table
  'why did <Name> rank...' — explain ranking decision
  'add requirement: ...'   — refine and re-rank
  'show top N'             — adjust displayed count
  'done' / 'exit'          — finish

You: compare top 3

⚖️  Comparing ['Nancy Evans', 'Matthew Alvarez', 'Michelle Allen']...

| Candidate Name | Role / Division | Years Exp | Key Skills | Education | Notable Strengths | Key Gaps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Nancy Evans | DevOps Engineer | 14.0 | Terraform, Kubernetes, Docker, Jenkins, CI/CD, Linux | Master of Science in Cloud Computing from Univ. of Illinois | Extensive experience, deep expertise in all must-haves | None apparent |
| Matthew Alvarez | DevOps Engineer | 13.0 | Docker, Terraform, Ansible, Kubernetes, Bash, Linux | Master of Science in Information Technology from Cornell | Strong Ansible background, solid Linux/Bash skills | None apparent |
| Michelle Allen | DevOps Engineer | 6.0 | Kubernetes, Docker, Terraform, Linux, Jenkins, Ansible, Prometheus | Bachelor of Science in Information Technology from Texas A&M | Familiar with modern observability (Prometheus) | Less experience than the top 2 |

**Summary:**
All three candidates are highly qualified, but **Nancy Evans** is the top pick due to her extensive experience (14 years) and deep, proven expertise across the entire requested stack (Kubernetes, Docker, Terraform, Linux, CI/CD), making her an immediate, high-impact asset.

You: done
  [intent: done]

💾 Report saved to: screening_report.md
```