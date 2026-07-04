# Agentic Profile Matching — Implementation Steps

## Prerequisites
- LM Studio running at `http://localhost:1234/v1`
- Models loaded: `google/gemma-4-e4b` (LLM) and `embedding-gemma-300m` (embeddings)
- Python 3.10+
- Milestone 1 (`fs_tools.py`) and Milestone 2 (`resume_rag.py`, `job_matcher.py`) code available

---

## Phase 0: Project Setup

### Step 0.1 — Initialize project structure
Create the following layout in the project root:
```
agentic-profile-matching/
├── docs/
├── tools/
├── resumes/
├── data/              # ChromaDB storage (gitignored)
├── matching_agent.py
├── resume_rag.py
├── job_matcher.py
├── fs_tools.py
└── requirements.txt
```

### Step 0.2 — Create `requirements.txt`
```
langgraph>=0.2.0
langchain-core>=0.2.0
openai>=1.0.0
chromadb>=0.4.0
pypdf>=4.0.0
docx2txt>=0.8
python-dotenv>=1.0.0
```

### Step 0.3 — Set up virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 0.4 — Copy Milestone 1 & 2 files
Copy the following from prior milestones into the project root:
- `fs_tools.py` (Milestone 1 — sandboxed file system tools)
- `resume_rag.py` (Milestone 2 — ChromaDB ingestion pipeline)
- `job_matcher.py` (Milestone 2 — hybrid search + scoring engine)
- `resumes/` directory (all existing resume `.txt`/`.pdf`/`.docx` files)

### Step 0.5 — Connectivity diagnostic
Create and run `diagnose.py` to verify LM Studio is reachable:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Test LLM
r = client.chat.completions.create(model="google/gemma-4-e4b",
    messages=[{"role": "user", "content": "ping"}])
print("LLM OK:", r.choices[0].message.content)

# Test Embedding
e = client.embeddings.create(model="embedding-gemma-300m", input="test")
print("Embedding OK, dim:", len(e.data[0].embedding))
```
**Exit criteria:** Both calls succeed without errors.

---

## Phase 1: Ingest Resumes into ChromaDB

> This re-uses `resume_rag.py` from Milestone 2, updated for the new embedding model.

### Step 1.1 — Update embedding model in `resume_rag.py`
Change the embedding model constant:
```python
EMBEDDING_MODEL = "embedding-gemma-300m"
```

### Step 1.2 — Run ingestion pipeline
```bash
.venv/bin/python resume_rag.py
```
This will:
- Read all files in `resumes/`
- Chunk them by section (SUMMARY, SKILLS, EXPERIENCE, EDUCATION)
- Extract metadata via LLM (name, skills, years_experience, education_level)
- Store chunks + embeddings in ChromaDB `resume_collection`

### Step 1.3 — Verify ChromaDB contents
```bash
.venv/bin/python -c "
import chromadb
client = chromadb.PersistentClient(path='./data')
col = client.get_collection('resume_collection')
print('Total chunks:', col.count())
print('Sample:', col.peek(1))
"
```
**Exit criteria:** Collection has > 0 documents; metadata fields are populated.

---

## Phase 2: Build the Three New Agent Tools

Each tool lives in the `tools/` directory as a standalone module.

### Step 2.1 — `tools/extract_requirements.py`

**Purpose:** Parse a raw JD string into structured must-have / nice-to-have requirements.

```python
def extract_requirements(jd: str) -> dict:
    """
    Returns:
        {
          "must_have": ["Python", "5+ years"],
          "nice_to_have": ["Kubernetes"],
          "role_type": "Backend Engineer"
        }
    """
```
- Make a chat completion call to `google/gemma-4-e4b` with `response_format={"type": "json_object"}`
- Prompt instructs the LLM to classify each requirement as mandatory or optional
- Parse and return the JSON response

**Test:** Call with a sample JD string; verify the output JSON has all three keys.

---

### Step 2.2 — `tools/compare_candidates.py`

**Purpose:** Fetch profiles for a list of candidate IDs and produce a side-by-side Markdown comparison.

```python
def compare_candidates(candidate_ids: list[str]) -> str:
    """
    Returns a Markdown table comparing skills, experience,
    match score, and identified gaps across candidates.
    """
```
- Query ChromaDB for each candidate's chunks by `candidate_name` metadata filter
- Aggregate metadata (skills, years_experience, education_level)
- Pass aggregated profiles to `google/gemma-4-e4b` with a comparison prompt
- Return the Markdown table string

**Test:** Pass 2–3 known candidate names; verify a readable table is returned.

---

### Step 2.3 — `tools/interview_questions.py`

**Purpose:** Generate role-specific interview questions for a single candidate.

```python
def generate_interview_questions(candidate_id: str, jd_requirements: dict) -> list[str]:
    """
    Returns a list of 5–10 targeted screening questions.
    """
```
- Fetch candidate's SKILLS and EXPERIENCE chunks from ChromaDB
- Combine with `jd_requirements` (must_have / nice_to_have)
- Prompt LLM to generate questions that probe gaps and verify claimed skills
- Return parsed list of question strings

**Test:** Pass a known candidate ID + sample requirements; verify 5+ questions are returned.

---

## Phase 3: Implement the LangGraph Agent

### Step 3.1 — Define `AgentState` in `matching_agent.py`

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

class AgentState(TypedDict):
    conversation_history: list[dict]
    raw_jd: str
    job_requirements: dict          # { must_have, nice_to_have, role_type }
    candidate_shortlist: list[dict] # Ranked candidates with scores
    reasoning_log: Annotated[list[str], operator.add]
    screening_round: int            # 1, 2, or 3
    human_feedback: str
    final_report: str
```

### Step 3.2 — Implement graph node functions

Create one function per node. Each receives and returns `AgentState`.

| Node Function | Logic |
|---------------|-------|
| `node_parse_jd(state)` | Normalize `raw_jd`; set `screening_round = 1` |
| `node_extract_requirements(state)` | Call `extract_requirements(raw_jd)`; store in `job_requirements` |
| `node_search_resumes(state)` | Call `job_matcher` RAG search with `job_requirements`; limit based on round (100→10→5) |
| `node_rank_candidates(state)` | Sort candidates by composite score; update `candidate_shortlist` |
| `node_generate_report(state)` | LLM generates per-candidate match report; store in `final_report` |
| `node_human_feedback(state)` | Print current shortlist; read user input; update `human_feedback` |

### Step 3.3 — Implement conditional edge router

```python
def route_after_rank(state: AgentState) -> str:
    if state["screening_round"] >= 3:
        return "generate_report"
    return "search_resumes"   # loop for next round

def route_after_feedback(state: AgentState) -> str:
    feedback = state["human_feedback"].lower()
    if "done" in feedback or "exit" in feedback:
        return END
    if any(kw in feedback for kw in ["require", "add", "must", "change"]):
        return "extract_requirements"
    return "rank_candidates"
```

### Step 3.4 — Wire the graph

```python
workflow = StateGraph(AgentState)

workflow.add_node("parse_jd",             node_parse_jd)
workflow.add_node("extract_requirements", node_extract_requirements)
workflow.add_node("search_resumes",       node_search_resumes)
workflow.add_node("rank_candidates",      node_rank_candidates)
workflow.add_node("generate_report",      node_generate_report)
workflow.add_node("human_feedback",       node_human_feedback)

workflow.set_entry_point("parse_jd")

workflow.add_edge("parse_jd",             "extract_requirements")
workflow.add_edge("extract_requirements", "search_resumes")
workflow.add_edge("search_resumes",       "rank_candidates")
workflow.add_conditional_edges("rank_candidates", route_after_rank)
workflow.add_edge("generate_report",      "human_feedback")
workflow.add_conditional_edges("human_feedback", route_after_feedback)

agent = workflow.compile()
```

### Step 3.5 — Implement the CLI entry point

```python
if __name__ == "__main__":
    jd = input("Enter job description: ")
    result = agent.invoke({
        "raw_jd": jd,
        "conversation_history": [],
        "job_requirements": {},
        "candidate_shortlist": [],
        "reasoning_log": [],
        "screening_round": 1,
        "human_feedback": "",
        "final_report": ""
    })
    print(result["final_report"])
```

---

## Phase 4: Implement Multi-Round Screening Logic

### Step 4.1 — Round-aware search in `node_search_resumes`

```python
ROUND_LIMITS = {1: 10, 2: 5, 3: 5}

def node_search_resumes(state):
    n = ROUND_LIMITS[state["screening_round"]]
    # If round > 1, restrict search to current shortlist candidate names
    name_filter = None
    if state["screening_round"] > 1:
        name_filter = [c["candidate_name"] for c in state["candidate_shortlist"]]
    results = rag_search(state["job_requirements"], limit=n, name_filter=name_filter)
    return {"candidate_shortlist": results}
```

### Step 4.2 — Round increment in `node_rank_candidates`

After ranking, increment `screening_round`:
```python
def node_rank_candidates(state):
    ranked = sort_by_score(state["candidate_shortlist"])
    return {
        "candidate_shortlist": ranked,
        "screening_round": state["screening_round"] + 1
    }
```

### Step 4.3 — Final report generation in `node_generate_report`

For each candidate in the shortlist, call LLM to produce:
- Match score + rationale
- Strengths vs. job requirements
- Gaps and missing skills
- Hire / No-hire recommendation
- 3 suggested interview questions (via `generate_interview_questions`)

---

## Phase 5: Conversational Interface

### Step 5.1 — Natural language query parser

Add a pre-processing step that maps user queries to agent actions:

| User Intent | Detected By | Agent Action |
|-------------|-------------|--------------|
| Initial JD input | First message | Run full graph |
| "Compare top N" | keyword: `compare` | Call `compare_candidates` |
| "Why did X rank..." | keyword: `why` | Lookup `reasoning_log` for candidate |
| "Add requirement X" | keyword: `add`/`require` | Update `job_requirements`, re-rank |
| "Show top N" | keyword: `show`/`top` | Print current `candidate_shortlist[:N]` |
| "Done" / "Exit" | keyword: `done`/`exit` | Route to END |

### Step 5.2 — Integrate into `node_human_feedback`

Parse `human_feedback` string using the intent table above. For compare/why queries, call the appropriate tool and print results before returning to the graph router.

### Step 5.3 — Conversation history tracking

Each user input and agent response should be appended to `conversation_history` as:
```python
{"role": "user", "content": "..."}
{"role": "assistant", "content": "..."}
```
Pass `conversation_history` in every LLM call for context continuity.

---

## Phase 6: Testing & Verification

### Step 6.1 — Unit test each tool
Test `extract_requirements`, `compare_candidates`, and `generate_interview_questions` in isolation with mock inputs.

### Step 6.2 — End-to-end smoke test
```bash
.venv/bin/python matching_agent.py
```
Input a sample JD and walk through:
1. Agent extracts requirements ✓
2. Round 1 returns top 10 ✓
3. Round 2 narrows to top 5 ✓
4. Round 3 generates final report ✓
5. User asks "Compare top 3" — comparison table printed ✓
6. User adds a requirement — agent re-ranks ✓
7. User types "done" — agent exits cleanly ✓

### Step 6.3 — Experience filter verification
Test `--min-experience` constraint:
- Query with `min_experience=6`; confirm no candidates with < 6 years appear in results

### Step 6.4 — Explainability check
For each final candidate, verify the report contains:
- [ ] Match score (numeric)
- [ ] Listed strengths
- [ ] Listed gaps
- [ ] Hire/no-hire verdict
- [ ] At least 3 interview questions

---

## Phase 7: Documentation & Cleanup

### Step 7.1 — Update `README.md`
Document:
- Prerequisites and LM Studio setup
- Installation steps
- How to run the agent
- Sample interaction transcript

### Step 7.2 — Verify `.gitignore`
Ensure `data/`, `.venv/`, `__pycache__/`, and `.env` are excluded.

### Step 7.3 — Final project review
- [ ] All source files present per architecture file structure
- [ ] ChromaDB populated with resumes
- [ ] All three new tools implemented and tested
- [ ] LangGraph graph compiles and runs without errors
- [ ] Conversational interface handles all documented query types
- [ ] Multi-round screening completes all 3 rounds correctly
- [ ] Final match reports include explainability details

---

## Implementation Order Summary

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7
  Setup     Ingest   3 Tools   LangGraph  Multi-    Conv.     Testing   Docs
                               Agent      Round     Interface
```
