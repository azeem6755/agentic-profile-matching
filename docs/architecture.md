# Agentic Profile Matching — Architecture Design

## Technical Constraints

| Component | Configuration |
|-----------|---------------|
| LLM | `google/gemma-4-e4b` via LM Studio at `http://localhost:1234/v1` |
| Embedding Model | `embedding-gemma-300m` via LM Studio at `http://localhost:1234/v1` |
| Vector DB | ChromaDB (persistent, local) |
| Agent Framework | LangGraph |
| File System Tools | Sandboxed `fs_tools.py` (Milestone 1) |

---

## 1. System Overview

The system is an **agentic recruiter assistant** that takes a job description (JD) as input and autonomously reasons through multi-round candidate screening via a LangGraph state machine. It surfaces results through a conversational interface that supports natural language queries and iterative refinement.

```
┌─────────────────────────────────────────────────────────────────┐
│                     User / CLI Interface                        │
│           (Natural Language Queries + Feedback Loop)            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LangGraph Agent (matching_agent.py)            │
│                                                                 │
│  START → Parse JD → Extract Requirements → Search Resumes →    │
│  Rank Candidates → Generate Report → Human Feedback → END      │
│                                                                 │
│  Agent State:                                                   │
│    - conversation_history: list[Message]                        │
│    - job_requirements: { must_have, nice_to_have }              │
│    - candidate_shortlist: list[CandidateResult]                 │
│    - reasoning_log: list[str]                                   │
│    - screening_round: int                                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │  Tool Calls
          ┌────────────────────┼─────────────────────┐
          │                    │                     │
          ▼                    ▼                     ▼
┌──────────────────┐  ┌────────────────┐  ┌──────────────────────┐
│  FS Tools        │  │  RAG Search    │  │  Agent-Specific Tools│
│  (Milestone 1)   │  │  (Milestone 2) │  │                      │
│                  │  │                │  │ extract_requirements  │
│  - read_file     │  │  ChromaDB      │  │ compare_candidates   │
│  - list_files    │  │  hybrid search │  │ generate_interview_  │
│  - sandbox safe  │  │  + scoring     │  │   questions          │
└──────────────────┘  └───────┬────────┘  └──────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │     ChromaDB       │
                    │  resume_collection │
                    │                   │
                    │  chunks + vectors  │
                    │  + metadata:       │
                    │   - candidate_name │
                    │   - skills         │
                    │   - years_exp      │
                    │   - section_type   │
                    └────────────────────┘
```

---

## 2. LangGraph Agent Design

### 2.1 Agent State Schema

```python
class AgentState(TypedDict):
    conversation_history: list[dict]       # Full message history
    raw_jd: str                            # Original job description
    job_requirements: dict                 # { must_have: [], nice_to_have: [] }
    candidate_shortlist: list[dict]        # Ranked candidate results
    reasoning_log: list[str]              # Step-by-step agent reasoning
    screening_round: int                   # 1 = initial, 2 = deep, 3 = final
    human_feedback: str                    # Latest user refinement input
    final_report: str                      # Generated match report
```

### 2.2 Graph Nodes

| Node | Responsibility |
|------|---------------|
| `parse_jd` | Receive the raw JD string, normalize it |
| `extract_requirements` | Call LLM to separate must-have vs. nice-to-have skills/experience |
| `search_resumes` | Query ChromaDB via RAG search tool; retrieve top-N candidates |
| `rank_candidates` | Apply hybrid scoring (semantic + skill overlap + experience); produce ranked shortlist |
| `generate_report` | LLM produces a structured match report for each candidate |
| `human_feedback_loop` | Pause for user input; update requirements or refine shortlist |
| `END` | Return final ranked list + reports |

### 2.3 Conditional Edges

```
extract_requirements
    ├─→ search_resumes (always)

rank_candidates
    ├─→ generate_report (if screening_round == 3)
    └─→ search_resumes  (if screening_round < 3, narrow pool)

human_feedback_loop
    ├─→ extract_requirements (if user changes JD requirements)
    ├─→ rank_candidates      (if user adjusts weights/criteria)
    └─→ END                  (if user is satisfied)
```

---

## 3. Tool Specifications

### 3.1 Inherited Tools (Milestone 1 + 2)

| Tool | Source | Description |
|------|--------|-------------|
| `read_file(path)` | fs_tools.py | Sandboxed resume file reader |
| `list_files(dir)` | fs_tools.py | Lists resumes in a directory |
| `rag_search(query, filters)` | resume_rag.py | Hybrid vector + metadata search over ChromaDB |

### 3.2 New Agent Tools

#### `extract_requirements(jd: str) → dict`
- **Input:** Raw job description string
- **Action:** LLM call (`google/gemma-4-e4b`) with structured JSON output
- **Output:**
  ```json
  {
    "must_have": ["Python", "5+ years", "AWS"],
    "nice_to_have": ["Kubernetes", "FastAPI"],
    "role_type": "Backend Engineer"
  }
  ```

#### `compare_candidates(candidate_ids: list[str]) → str`
- **Input:** List of candidate identifiers (from shortlist)
- **Action:** Fetches candidate metadata from ChromaDB; LLM generates side-by-side comparison table
- **Output:** Markdown comparison table (skills, experience, match score, gaps)

#### `generate_interview_questions(candidate_id: str) → list[str]`
- **Input:** Single candidate ID
- **Action:** LLM generates role-specific questions based on candidate's profile vs. JD requirements
- **Output:** List of 5–10 targeted screening questions

---

## 4. Multi-Round Screening Pipeline

```
Round 1 — Initial Screen
  Input:  100 resumes in ChromaDB
  Action: RAG search + keyword match → score all candidates
  Output: Top 10 shortlist

Round 2 — Deep Analysis
  Input:  Top 10 candidates
  Action: LLM reads full resume sections; detailed skill gap analysis
  Output: Refined ranking of top 5 with reasoning

Round 3 — Final Recommendation
  Input:  Top 5 candidates
  Action: LLM generates hire/no-hire verdict + match report + interview questions
  Output: Structured final report per candidate
```

---

## 5. Vector Database Schema (ChromaDB)

- **Collection:** `resume_collection`
- **Embedding Model:** `embedding-gemma-300m` (via LM Studio)
- **Chunk Strategy:** Section-aware (EXPERIENCE, SKILLS, EDUCATION, SUMMARY)

| Metadata Field | Type | Description |
|----------------|------|-------------|
| `candidate_name` | string | Full name |
| `resume_path` | string | Absolute sandboxed path |
| `section_type` | string | EXPERIENCE / SKILLS / EDUCATION / SUMMARY |
| `skills` | string | Comma-separated skill tags |
| `years_experience` | int | Parsed total years of experience |
| `education_level` | string | Highest degree |

---

## 6. LLM Integration

All LLM calls use the OpenAI-compatible client pointed at LM Studio:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"  # placeholder, not validated
)

# Reasoning / generation
response = client.chat.completions.create(
    model="google/gemma-4-e4b",
    messages=[...],
    response_format={"type": "json_object"}  # JSON mode supported
)

# Embeddings
embedding = client.embeddings.create(
    model="embedding-gemma-300m",
    input="candidate text chunk"
)
```

---

## 7. Source File Structure

```
agentic-profile-matching/
├── docs/
│   ├── problem_statement.md
│   ├── problem_statement.txt
│   ├── contstraints.txt
│   └── architecture.md          ← this file
│
├── matching_agent.py             # LangGraph agent (main entry point)
├── resume_rag.py                 # ChromaDB ingestion pipeline (Milestone 2)
├── job_matcher.py                # Hybrid search + scoring engine (Milestone 2)
├── fs_tools.py                   # Sandboxed file system tools (Milestone 1)
├── tools/
│   ├── extract_requirements.py   # Must-have / nice-to-have parser
│   ├── compare_candidates.py     # Side-by-side candidate comparison
│   └── interview_questions.py    # Interview question generator
│
├── resumes/                      # Input resume files (txt/pdf/docx)
├── data/                         # ChromaDB persistent storage (gitignored)
├── requirements.txt
└── .gitignore
```

---

## 8. Conversational Interface Flow

```
User:  "Find me candidates with React and 3+ years experience"
         │
         ▼
Agent: extract_requirements → rag_search(React, min_exp=3) → rank → respond

User:  "Compare the top 3 matches side by side"
         │
         ▼
Agent: compare_candidates([id1, id2, id3]) → Markdown table → respond

User:  "Why did John rank higher than Jane?"
         │
         ▼
Agent: fetch reasoning_log for John & Jane → LLM explanation → respond

User:  "Actually, also require TypeScript"
         │
         ▼
Agent: update job_requirements → re-run search → new ranking → respond
```
