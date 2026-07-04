"""
matching_agent.py
Phase 3/4/5 — LangGraph Agentic Profile Matching System

Implements the full agent: state machine, multi-round screening,
and conversational interface via LangGraph.
"""
import os
import re
import sys
import json
import operator
from typing import TypedDict, Annotated
from openai import OpenAI
from langgraph.graph import StateGraph, END

from tools.extract_requirements import extract_requirements as _extract_req
from tools.compare_candidates import compare_candidates as _compare
from tools.interview_questions import generate_interview_questions as _questions
from job_matcher import match_job

# ── Config ────────────────────────────────────────────────────────────────────
LM_STUDIO_URL = "http://localhost:1234/v1"
LLM_MODEL     = "google/gemma-4-e4b"
ROUND_LIMITS  = {1: 10, 2: 5, 3: 5}

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "strengths":         {"type": "array", "items": {"type": "string"}},
        "gaps":              {"type": "array", "items": {"type": "string"}},
        "verdict":           {"type": "string", "enum": ["HIRE", "INTERVIEW", "REJECT"]},
        "verdict_reasoning": {"type": "string"},
    },
    "required": ["strengths", "gaps", "verdict", "verdict_reasoning"]
}

# ── Shared client (singleton) ─────────────────────────────────────────────────
_client: OpenAI | None = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    return _client


# ══════════════════════════════════════════════════════════════════════════════
# Step 3.1 — Agent State
# ══════════════════════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    conversation_history: list[dict]
    raw_jd:               str
    job_requirements:     dict           # {role_type, must_have[], nice_to_have[]}
    candidate_shortlist:  list[dict]     # ranked candidates with scores
    reasoning_log:        Annotated[list[str], operator.add]  # append-only
    screening_round:      int            # 1 → 2 → 3
    human_feedback:       str
    final_report:         str


# ══════════════════════════════════════════════════════════════════════════════
# Step 3.2 — Graph Node Functions
# ══════════════════════════════════════════════════════════════════════════════

def node_parse_jd(state: AgentState) -> dict:
    jd = state["raw_jd"].strip()
    print(f"\n{'='*60}")
    print(f"  🚀 Agentic Profile Matching — Starting")
    print(f"  JD: {jd[:120]}...")
    print(f"{'='*60}\n")
    return {
        "raw_jd":             jd,
        "screening_round":    1,
        "candidate_shortlist": [],
        "final_report":       "",
        "reasoning_log":      ["[parse_jd] JD received. Round 1 starting."],
    }


def node_extract_requirements(state: AgentState) -> dict:
    """Call Phase 2 tool; merge any mid-conversation refinements into the JD."""
    print("📋 Extracting requirements...")
    feedback = state.get("human_feedback", "")
    jd_text  = state["raw_jd"]
    if feedback:
        jd_text = f"{jd_text}\n\nAdditional constraints from user: {feedback}"

    req = _extract_req(jd_text, client=get_client())
    print(f"  Role: {req.get('role_type')} | Must-have: {len(req.get('must_have',[]))} items")

    history = list(state.get("conversation_history", []))
    history.append({"role": "assistant", "content": f"Requirements extracted: {json.dumps(req)}"})
    return {
        "job_requirements":     req,
        "conversation_history": history,
        "reasoning_log":        [f"[extract_req] {req}"],
    }


def node_search_resumes(state: AgentState) -> dict:
    """
    Phase 4, Step 4.1 — Round-aware search.
    Round 1: full RAG search via job_matcher.
    Round 2+: narrow existing shortlist (no re-query).
    """
    round_num = state["screening_round"]
    limit     = ROUND_LIMITS.get(round_num, 5)
    req       = state.get("job_requirements", {})
    must_have = req.get("must_have", [])

    print(f"\n🔍 [Round {round_num}] Searching resumes (limit={limit})...")

    if round_num > 1:
        current  = state.get("candidate_shortlist", [])
        narrowed = current[:limit]
        log = f"[Round {round_num}] Narrowed shortlist {len(current)} → {len(narrowed)}."
        print(f"  {log}")
        return {"candidate_shortlist": narrowed, "reasoning_log": [log]}

    # Round 1: full RAG search — pass required_skills=None so match_job uses its own
    # lean LLM extraction (short token-level skills like "Kubernetes" that substring-match
    # candidate text, rather than the verbose must_have phrases from node_extract_requirements)
    try:
        results   = match_job(jd_text=state["raw_jd"], required_skills=None, limit=limit)
        shortlist = results.get("top_matches", [])
    except Exception as e:
        print(f"  ⚠️  Search error: {e}")
        shortlist = []

    log = f"[Round 1] RAG search returned {len(shortlist)} candidates."
    print(f"  {log}")
    return {"candidate_shortlist": shortlist, "reasoning_log": [log]}


def node_rank_candidates(state: AgentState) -> dict:
    """Phase 4, Step 4.2 — Sort by match_score and increment screening round."""
    round_num = state["screening_round"]
    limit     = ROUND_LIMITS.get(round_num, 5)
    ranked    = sorted(
        state.get("candidate_shortlist", []),
        key=lambda x: x.get("match_score", 0),
        reverse=True
    )[:limit]

    next_round = round_num + 1
    log = f"[Round {round_num}] Ranked {len(ranked)} candidates → advancing to round {next_round}."
    print(f"\n  📊 {log}")
    for i, c in enumerate(ranked, 1):
        print(f"    {i}. {c.get('candidate_name','?')}  score={c.get('match_score',0)}")

    return {
        "candidate_shortlist": ranked,
        "screening_round":     next_round,
        "reasoning_log":       [log],
    }


def node_generate_report(state: AgentState) -> dict:
    """Phase 4, Step 4.3 — Full hire/no-hire report with LLM analysis + interview questions."""
    print(f"\n📄 Generating final report...")
    client    = get_client()
    req       = state.get("job_requirements", {})
    shortlist = state.get("candidate_shortlist", [])

    if not shortlist:
        return {
            "final_report":  "⚠️  No candidates to report on.",
            "reasoning_log": ["[report] Empty shortlist."],
        }

    sections = [
        "# Final Screening Report",
        f"**Role:** {req.get('role_type', 'Unknown')}",
        f"**Must-Have:** {', '.join(req.get('must_have', []))}",
        "---\n",
    ]

    for rank, cand in enumerate(shortlist, 1):
        name   = cand.get("candidate_name", "Unknown")
        score  = cand.get("match_score", 0)
        skills = cand.get("matched_skills", [])
        init_r = cand.get("reasoning", "")
        print(f"  Analysing {name} (score={score})...")

        prompt = (
            f"Candidate: {name}\nMatch Score: {score}/100\n"
            f"Matched Skills: {', '.join(skills)}\nInitial Reasoning: {init_r}\n\n"
            f"Must-Have: {', '.join(req.get('must_have',[]))}\n"
            f"Nice-to-Have: {', '.join(req.get('nice_to_have',[]))}\n\n"
            "Give a concise structured evaluation with strengths, gaps, and a verdict."
        )
        try:
            resp     = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a senior recruiter. Be concise and evidence-based."},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_schema", "json_schema": {"name": "report", "schema": REPORT_SCHEMA}},
                temperature=0.1,
            )
            analysis = json.loads(resp.choices[0].message.content)
        except Exception as e:
            analysis = {"strengths": skills, "gaps": [], "verdict": "INTERVIEW", "verdict_reasoning": str(e)}

        questions = _questions(name, req, client=client)
        verdict   = analysis.get("verdict", "INTERVIEW")
        icon      = {"HIRE": "✅", "INTERVIEW": "🟡", "REJECT": "❌"}.get(verdict, "🟡")

        block = [
            f"## {rank}. {name}  {icon} {verdict}",
            f"**Match Score:** {score}/100",
            "\n**Strengths:**",
            *[f"- {s}" for s in analysis.get("strengths", [])],
            "\n**Gaps:**",
            *(([f"- {g}" for g in analysis.get("gaps", [])]) or ["- None identified"]),
            f"\n**Verdict:** {verdict} — {analysis.get('verdict_reasoning','')}",
        ]
        if questions:
            block.append("\n**Suggested Interview Questions:**")
            block += [f"{i}. {q}" for i, q in enumerate(questions[:5], 1)]
        block.append("\n---\n")
        sections.extend(block)

    full_report = "\n".join(sections)
    print("✅ Report complete.")
    return {"final_report": full_report, "reasoning_log": ["[report] Done."]}


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Conversational Interface
# ══════════════════════════════════════════════════════════════════════════════

def _parse_intent(text: str) -> str:
    """Map natural-language input to an intent token."""
    t = text.lower()
    if any(k in t for k in ["done", "exit", "quit", "finish", "that's all"]):
        return "done"
    if any(k in t for k in ["compare", "vs", "versus", "side by side"]):
        return "compare"
    if any(k in t for k in ["why", "explain", "reason", "how come"]):
        return "why"
    if any(k in t for k in ["add", "require", "must", "change", "also need", "include"]):
        return "add_requirement"
    if any(k in t for k in ["show", "list", "top", "display"]):
        return "show"
    return "unknown"


def _extract_n(text: str, default: int = 3) -> int:
    nums = re.findall(r'\d+', text)
    return int(nums[0]) if nums else default


def node_human_feedback(state: AgentState) -> dict:
    """Phase 5 — Print report/shortlist, read NL input, execute inline tool calls."""
    shortlist = state.get("candidate_shortlist", [])
    req       = state.get("job_requirements", {})
    report    = state.get("final_report", "")
    client    = get_client()
    history   = list(state.get("conversation_history", []))

    if report:
        print("\n" + "=" * 60)
        print(report)
        print("=" * 60)

    print(f"\n📋 Top {len(shortlist)} candidates:")
    for i, c in enumerate(shortlist, 1):
        print(f"  {i}. {c.get('candidate_name','?')}  (score: {c.get('match_score',0)})")

    print("\n💬 Options:")
    print("  'compare top N'          — side-by-side comparison table")
    print("  'why did <Name> rank...' — explain ranking decision")
    print("  'add requirement: ...'   — refine and re-rank")
    print("  'show top N'             — adjust displayed count")
    print("  'done' / 'exit'          — finish\n")

    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        user_input = "done"

    if not user_input:
        user_input = "done"

    history.append({"role": "user", "content": user_input})
    intent = _parse_intent(user_input)
    print(f"  [intent: {intent}]")

    # ── Inline tool calls ──────────────────────────────────────
    if intent == "compare":
        n     = _extract_n(user_input, 3)
        names = [c.get("candidate_name") for c in shortlist[:n] if c.get("candidate_name")]
        print(f"\n⚖️  Comparing {names}...\n")
        result = _compare(names, jd_requirements=req, client=client)
        print(result)
        history.append({"role": "assistant", "content": result})
        # Keep same state, re-show feedback prompt next iteration
        user_input = "continue"

    elif intent == "why":
        m = re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', user_input)
        if m:
            name = m.group(0)
            matched = [c for c in shortlist if name.lower() in c.get("candidate_name", "").lower()]
            if matched:
                c   = matched[0]
                exp = (
                    f"**{c['candidate_name']}** scored {c['match_score']}/100.\n"
                    f"- Matched skills: {', '.join(c.get('matched_skills', []))}\n"
                    f"- Reasoning: {c.get('reasoning', 'N/A')}"
                )
                print(f"\n{exp}\n")
                history.append({"role": "assistant", "content": exp})
        user_input = "continue"

    elif intent == "show":
        n = _extract_n(user_input, len(shortlist))
        print(f"\n  Showing top {n}:")
        for i, c in enumerate(shortlist[:n], 1):
            print(f"  {i}. {c.get('candidate_name','?')}  score={c.get('match_score',0)}")
        user_input = "continue"

    return {
        "human_feedback":       user_input,
        "conversation_history": history,
        "reasoning_log":        [f"[feedback] intent={intent}, input='{user_input[:60]}'"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Step 3.3 — Conditional Edge Routers
# ══════════════════════════════════════════════════════════════════════════════

def route_after_rank(state: AgentState) -> str:
    """After ranking, go to generate_report once screening_round >= 3."""
    if state["screening_round"] >= 3:
        return "generate_report"
    return "search_resumes"


def route_after_feedback(state: AgentState) -> str:
    """Route based on user intent after viewing the report."""
    feedback = state.get("human_feedback", "").lower()
    if any(k in feedback for k in ["done", "exit", "quit", "finish"]):
        return END
    if any(k in feedback for k in ["require", "add", "must", "change", "also need", "include"]):
        return "extract_requirements"
    # For compare/why/show/continue, loop back to feedback without re-ranking
    return "human_feedback"


# ══════════════════════════════════════════════════════════════════════════════
# Step 3.4 — Wire the Graph
# ══════════════════════════════════════════════════════════════════════════════

def build_agent():
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
    workflow.add_conditional_edges("rank_candidates",  route_after_rank)
    workflow.add_edge("generate_report",      "human_feedback")
    workflow.add_conditional_edges("human_feedback",   route_after_feedback)

    return workflow.compile()


agent = build_agent()


# ══════════════════════════════════════════════════════════════════════════════
# Step 3.5 — CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Agentic Profile Matching System")
    print("  Powered by LangGraph + LM Studio (gemma-4-e4b)")
    print("=" * 60)

    if len(sys.argv) > 1:
        # Accept JD as a CLI argument for non-interactive use
        jd = " ".join(sys.argv[1:])
    else:
        print("\nPaste the job description below (press Enter twice to submit):\n")
        lines = []
        try:
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
        except EOFError:
            pass
        jd = "\n".join(lines).strip()

    if not jd:
        print("No job description provided. Exiting.")
        sys.exit(1)

    initial_state: AgentState = {
        "raw_jd":               jd,
        "conversation_history": [],
        "job_requirements":     {},
        "candidate_shortlist":  [],
        "reasoning_log":        [],
        "screening_round":      1,
        "human_feedback":       "",
        "final_report":         "",
    }

    final_state = agent.invoke(initial_state)

    # Save final report to file
    report_path = "screening_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_state.get("final_report", "No report generated."))
    print(f"\n💾 Report saved to: {report_path}")
