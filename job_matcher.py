import os
import sys
import json
import argparse
from openai import OpenAI
import chromadb

# Import file system tools (sandbox safeguards)
from fs_tools import read_file

# Configuration
LM_STUDIO_URL = "http://localhost:1234/v1"
EMBEDDING_MODEL = "text-embedding-embeddinggemma-300m-qat"
LLM_MODEL = "google/gemma-4-e4b"
CHROMA_DB_PATH = "./data/chroma_db"
COLLECTION_NAME = "resume_collection"

# Schema for generating match reasoning
REASONING_SCHEMA = {
    "type": "object",
    "properties": {
        "matched_skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of key skills from the job description that the candidate possesses."
        },
        "relevant_excerpts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-3 short, exact sentence quotes from the candidate's resume that demonstrate their alignment."
        },
        "reasoning": {
            "type": "string",
            "description": "A 2-3 sentence paragraph explaining why this candidate is a match based on experience, skills, and education."
        }
    },
    "required": ["matched_skills", "relevant_excerpts", "reasoning"]
}

# Schema to extract skills from JD automatically if not provided
JD_SKILLS_SCHEMA = {
    "type": "object",
    "properties": {
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of key skills, tools, and languages required in the job description."
        }
    },
    "required": ["skills"]
}

def get_jd_embedding(client: OpenAI, text: str) -> list:
    """
    Generates embedding for the Job Description.
    """
    clean_text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=clean_text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def extract_skills_from_jd(client: OpenAI, jd_text: str) -> list:
    """
    Calls LLM to automatically extract required skills from the Job Description.
    """
    prompt = f"Identify all required skills, tools, frameworks, and programming languages in this job description:\n\n{jd_text}"
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a recruiting assistant. Extract the skills list as a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "jd_skills",
                    "schema": JD_SKILLS_SCHEMA
                }
            },
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("skills", [])
    except Exception as e:
        print(f"Error auto-extracting skills: {str(e)}", file=sys.stderr)
        return []

def generate_match_reasoning(client: OpenAI, jd_text: str, candidate_name: str, resume_chunks: list) -> dict:
    """
    Uses LLM to evaluate the candidate's resume parts against the JD.
    Returns matched skills, exact excerpts, and reasoning.
    """
    combined_resume = "\n\n".join(resume_chunks)
    prompt = (
        f"Evaluate candidate '{candidate_name}' for the following job description:\n\n"
        f"JOB DESCRIPTION:\n{jd_text}\n\n"
        f"CANDIDATE RESUME EXCERPTS:\n{combined_resume}"
    )
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert technical recruiter. Analyze the candidate and output a JSON reasoning object matching the schema."},
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "match_reasoning",
                    "schema": REASONING_SCHEMA
                }
            },
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "matched_skills": [],
            "relevant_excerpts": ["Error generating excerpts."],
            "reasoning": f"Could not generate reasoning due to LLM error: {str(e)}"
        }

def match_job(jd_text: str, min_experience: float = 0.0, required_skills: list = None, limit: int = 10):
    """
    Queries ChromaDB, applies filters, calculates composite scores,
    and runs LLM-based verification for the top candidate matches.
    """
    # 1. Connect to LM Studio and ChromaDB
    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Error: Vector database not found at '{CHROMA_DB_PATH}'. Please run 'resume_rag.py' first.", file=sys.stderr)
        sys.exit(1)
        
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Check if collection exists
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print(f"Error: Collection '{COLLECTION_NAME}' not found in ChromaDB. Please ingest resumes first.", file=sys.stderr)
        sys.exit(1)
        
    if collection.count() == 0:
        print(f"Error: Collection '{COLLECTION_NAME}' is empty. Please ingest resumes first.", file=sys.stderr)
        sys.exit(1)

    # 2. Get Job Description embedding
    jd_embedding = get_jd_embedding(client, jd_text)
    
    # 3. Retrieve chunks from ChromaDB
    # We query more than K chunks to allow post-filtering and candidate aggregation
    query_results = collection.query(
        query_embeddings=[jd_embedding],
        n_results=min(100, collection.count())
    )
    
    if not query_results["documents"] or not query_results["documents"][0]:
        return {"job_description": jd_text, "top_matches": []}
        
    documents = query_results["documents"][0]
    metadatas = query_results["metadatas"][0]
    distances = query_results["distances"][0]
    
    # 4. Group and aggregate chunks by candidate
    candidates = {}
    for idx, doc in enumerate(documents):
        meta = metadatas[idx]
        dist = distances[idx]
        
        path = meta["resume_path"]
        name = meta["candidate_name"]
        
        if path not in candidates:
            # Parse candidate skills list
            skills_str = meta.get("skills", "")
            skills_list = [s.strip() for s in skills_str.split(",") if s.strip()]
            
            candidates[path] = {
                "name": name,
                "resume_path": path,
                "experience_years": float(meta.get("experience_years", 0.0)),
                "skills": skills_list,
                "chunks": [],
                "best_distance": dist
            }
        else:
            if dist < candidates[path]["best_distance"]:
                candidates[path]["best_distance"] = dist
                
        candidates[path]["chunks"].append(doc)

    # 5. Extract target skills from JD if not explicitly provided
    if not required_skills:
        required_skills = extract_skills_from_jd(client, jd_text)
    
    # Normalize required skills for keyword matching (lowercase)
    required_skills_norm = [s.lower() for s in required_skills]
    
    # 6. Filter and Score candidates
    scored_candidates = []
    
    for path, cand in candidates.items():
        # Hard Filter A: Experience Years
        if cand["experience_years"] < min_experience:
            continue
            
        # Hard Filter B: Must-have Skills
        # A candidate must have ALL specified required_skills.
        # For multi-word skills (e.g. "linux administration"), we match on the first
        # significant word (length > 3) so "Linux Administration" matches "Linux".
        has_all_skills = True
        cand_skills_lower = [s.lower() for s in cand["skills"]]
        combined_chunks_lower = " ".join(cand["chunks"]).lower()

        def _skill_token(skill_phrase: str) -> str:
            """Extract the first significant word from a skill phrase."""
            words = [w for w in skill_phrase.split() if len(w) > 3]
            return words[0] if words else skill_phrase

        for req_skill in required_skills_norm:
            token = _skill_token(req_skill)
            # Check if token appears in the candidate's skill list or any resume chunk
            in_skills = any(token in s for s in cand_skills_lower)
            in_chunks  = token in combined_chunks_lower
            if not in_skills and not in_chunks:
                has_all_skills = False
                break

        if not has_all_skills:
            continue

        # Calculate Scores
        
        # A. Semantic Score (60%)
        # ChromaDB by default uses L2 squared distance.
        # Let's map it to similarity: similarity = 1.0 / (1.0 + distance)
        similarity = 1.0 / (1.0 + cand["best_distance"])
        semantic_score = similarity * 100.0
        
        # B. Skill Match Score (30%)
        if required_skills:
            matched_count = 0
            for req_skill in required_skills_norm:
                token = _skill_token(req_skill)
                if any(token in s for s in cand_skills_lower) or token in combined_chunks_lower:
                    matched_count += 1
            skill_score = (matched_count / len(required_skills)) * 100.0
        else:
            skill_score = 100.0  # Default if no skills required
            
        # C. Experience Score (10%)
        if min_experience > 0:
            # Proximity to required experience, cap at 100
            exp_score = min(100.0, (cand["experience_years"] / min_experience) * 100.0)
        else:
            # If no min experience, score is based on a sigmoid/cap of years (e.g. 5+ years gets 100)
            exp_score = min(100.0, (cand["experience_years"] / 5.0) * 100.0)
            
        # Composite Match Score
        final_score = (semantic_score * 0.6) + (skill_score * 0.3) + (exp_score * 0.1)
        final_score = round(final_score, 1)
        
        scored_candidates.append({
            "candidate_name": cand["name"],
            "resume_path": cand["resume_path"],
            "match_score": final_score,
            "experience_years": cand["experience_years"],
            "chunks": cand["chunks"],
            "skills": cand["skills"]
        })

    # Sort candidates by final score descending
    scored_candidates.sort(key=lambda x: x["match_score"], reverse=True)
    
    # Limit to top-K matches
    top_candidates = scored_candidates[:limit]
    
    # 7. Generate LLM Match Reasoning for Top Candidates
    final_matches = []
    for tc in top_candidates:
        # Ask LLM to review and generate reasoning/excerpts
        reasoning_data = generate_match_reasoning(
            client=client,
            jd_text=jd_text,
            candidate_name=tc["candidate_name"],
            resume_chunks=tc["chunks"]
        )
        
        final_matches.append({
            "candidate_name": tc["candidate_name"],
            "resume_path": tc["resume_path"],
            "match_score": tc["match_score"],
            "matched_skills": reasoning_data.get("matched_skills", []),
            "relevant_excerpts": reasoning_data.get("relevant_excerpts", []),
            "reasoning": reasoning_data.get("reasoning", "")
        })

    # Output structured result
    output_data = {
        "job_description": jd_text,
        "top_matches": final_matches
    }
    
    return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Job Matching Engine")
    parser.add_argument("--jd", type=str, required=True, help="Job description text")
    parser.add_argument("--min-experience", type=float, default=0.0, help="Minimum required years of experience")
    parser.add_argument("--skills", type=str, default="", help="Comma-separated list of required must-have skills")
    parser.add_argument("--limit", type=int, default=10, help="Max matches to return")
    
    args = parser.parse_args()
    
    # Parse required skills list
    req_skills = [s.strip() for s in args.skills.split(",") if s.strip()] if args.skills else None
    
    results = match_job(
        jd_text=args.jd,
        min_experience=args.min_experience,
        required_skills=req_skills,
        limit=args.limit
    )
    
    print(json.dumps(results, indent=2))
