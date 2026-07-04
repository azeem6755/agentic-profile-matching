# Agentic Profile Matching — Problem Statement

## Assignment Overview

This is **Milestone 3** of the Airtribe AI Engineering series, building upon the sandboxed file system tools (Milestone 1) and the RAG-based resume matching engine (Milestone 2).

---

## Part A: Agent Architecture (40%)

Create `matching_agent.py` using **LangGraph**.

### Agent State Design
- Track conversation history
- Maintain job requirements understanding
- Store candidate shortlist and reasoning

### Agent Workflow (Graph Structure)

```
START → Parse JD → Extract Requirements → Search Resumes →
Rank Candidates → Generate Report → Human Feedback Loop → END
```

### Tools Available to Agent
- All file system tools (from Milestone 1)
- RAG search tool (from Milestone 2)
- Additional tools:
  - `extract_requirements(jd: str)` — Parse must-have vs. nice-to-have requirements
  - `compare_candidates(candidate_ids: list)` — Head-to-head comparison
  - `generate_interview_questions(candidate_id: str)` — Create screening questions

---

## Part B: Interactive Features (30%)

### Conversational Interface
- Accept natural language queries:
  - *"Find me candidates with React and 3+ years experience"*
  - *"Compare the top 3 matches side by side"*
  - *"Why did John rank higher than Jane?"*

### Iterative Refinement
- Allow users to adjust requirements mid-conversation
- Agent re-ranks based on new criteria
- Explains changes in rankings

---

## Part C: Advanced Capabilities (30%)

### Multi-Round Screening
- **Initial screen:** Top 10 from 100 resumes
- **Second round:** Deep analysis of top 10
- **Final round:** Generate hire/no-hire recommendation

### Explainability
- Generate detailed match reports
- Highlight strengths and gaps for each candidate
- Provide improvement suggestions for borderline candidates
