import os
import re
import json
from openai import OpenAI
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

# Import file system tools from Milestone 1 (copied to current root)
from fs_tools import read_file, list_files

# Configuration
LM_STUDIO_URL = "http://localhost:1234/v1"
EMBEDDING_MODEL = "text-embedding-embeddinggemma-300m-qat"
LLM_MODEL = "google/gemma-4-e4b"
CHROMA_DB_PATH = "./data/chroma_db"
COLLECTION_NAME = "resume_collection"

# Schema for metadata extraction
METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "The full name of the candidate. If not found, use 'Unknown'."
        },
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of key technical, tools, programming languages, and professional skills mentioned in the resume."
        },
        "experience_years": {
            "type": "number",
            "description": "Total number of years of professional work experience. Calculate this as a decimal float value from the experience dates."
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string", "description": "Degree title, e.g., Bachelor of Science, Master, PhD, etc."},
                    "major": {"type": "string", "description": "Major field of study, e.g., Computer Science, Electrical Engineering, etc."},
                    "institution": {"type": "string", "description": "Name of the university, college, or school."}
                },
                "required": ["degree", "major", "institution"]
            },
            "description": "List of educational degrees and institutions."
        }
    },
    "required": ["name", "skills", "experience_years", "education"]
}

class LMStudioEmbeddingFunction(EmbeddingFunction):
    """
    Custom ChromaDB embedding function that calls the local LM Studio embeddings endpoint.
    """
    def __init__(self, client, model_name=EMBEDDING_MODEL):
        self.client = client
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            try:
                # Replace newline characters with spaces as recommended for some embedding models
                clean_text = text.replace("\n", " ")
                response = self.client.embeddings.create(
                    input=clean_text,
                    model=self.model_name
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"Error generating embedding for chunk: {str(e)}")
                # Return dummy/zero vector of size 768 on failure to prevent entire batch crash
                embeddings.append([0.0] * 768)
        return embeddings

def chunk_resume(text: str) -> list:
    """
    Splits the resume text into section-aware chunks.
    Detects section headers and divides the text accordingly.
    """
    # Define section regex headers to look for
    headers_regex = re.compile(
        r'(?m)^(?:[#\-\*\s]*)\b(EXPERIENCE|WORK\s+EXPERIENCE|EMPLOYMENT\s+HISTORY|EDUCATION|ACADEMIC\s+BACKGROUND|SKILLS|TECHNICAL\s+SKILLS|PROJECTS|SUMMARY|PROFESSIONAL\s+SUMMARY|OBJECTIVE|CERTIFICATIONS|AWARDS)\b[:\s]*$',
        re.IGNORECASE
    )
    
    # Find all header matches and positions
    matches = list(headers_regex.finditer(text))
    
    chunks = []
    
    if not matches:
        # Default fallback: split by paragraphs/newlines if no headers found
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for idx, p in enumerate(paragraphs):
            chunks.append({
                "text": p,
                "section_type": "general"
            })
        return chunks

    # Process first section before any matched header (e.g. Header contact details)
    first_match_start = matches[0].start()
    intro_text = text[:first_match_start].strip()
    if intro_text:
        chunks.append({
            "text": intro_text,
            "section_type": "contact_info"
        })
        
    # Process sections defined by matches
    for i in range(len(matches)):
        start_idx = matches[i].end()
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
        
        section_name = matches[i].group(1).lower()
        section_text = text[start_idx:end_idx].strip()
        
        # Standardize section types
        if "exp" in section_name or "employ" in section_name:
            section_type = "experience"
        elif "edu" in section_name or "acad" in section_name:
            section_type = "education"
        elif "skill" in section_name or "techn" in section_name:
            section_type = "skills"
        elif "proj" in section_name:
            section_type = "projects"
        elif "sum" in section_name or "obj" in section_name:
            section_type = "summary"
        else:
            section_type = "general"
            
        if not section_text:
            continue
            
        # If the section content is extremely long, sub-chunk it to preserve local context
        max_chunk_size = 1000
        overlap = 150
        if len(section_text) > max_chunk_size:
            # Sub-chunking
            cursor = 0
            sub_idx = 1
            while cursor < len(section_text):
                sub_text = section_text[cursor : cursor + max_chunk_size]
                chunks.append({
                    "text": f"[{matches[i].group(0).strip()} Part {sub_idx}]\n{sub_text}",
                    "section_type": section_type
                })
                cursor += (max_chunk_size - overlap)
                sub_idx += 1
        else:
            chunks.append({
                "text": f"[{matches[i].group(0).strip()}]\n{section_text}",
                "section_type": section_type
            })
            
    return chunks

def extract_metadata(client: OpenAI, text: str) -> dict:
    """
    Uses LM Studio (google/gemma-4-e4b) in json_schema mode to extract structured metadata.
    """
    # Truncate text if it is extremely long to fit model context window
    truncated_text = text[:8000]
    prompt = f"Analyze the following resume text and extract the candidate profile metadata details:\n\n{truncated_text}"
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a recruiting intelligence assistant. Extract metadata details from the resume and output a valid JSON object matching the requested schema."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "candidate_metadata",
                    "schema": METADATA_SCHEMA
                }
            },
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error calling LLM metadata extraction: {str(e)}")
        # Return fallback empty structure
        return {
            "name": "Unknown",
            "skills": [],
            "experience_years": 0.0,
            "education": []
        }

def ingest_resumes(directory: str):
    """
    Ingests all resumes in the specified directory: parses, extracts metadata,
    chunks, embeds, and stores them in ChromaDB.
    """
    print(f"Initializing connection to LM Studio...")
    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    
    print(f"Initializing persistent ChromaDB at '{CHROMA_DB_PATH}'...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Custom embedding function targeting LM Studio
    embedding_fn = LMStudioEmbeddingFunction(client)
    
    # Create or retrieve collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    
    # List files using Milestone 1 tools
    files = list_files(directory)
    valid_files = [f for f in files if "status" not in f and f["name"].lower().endswith(('.pdf', '.docx', '.pptx', '.txt'))]
    
    print(f"Found {len(valid_files)} resumes to process.")
    
    for idx, file_meta in enumerate(valid_files):
        rel_path = file_meta["relative_path"]
        filename = file_meta["name"]
        print(f"\n[{idx+1}/{len(valid_files)}] Processing: {filename}...")
        
        # 1. Read full text
        parsed = read_file(rel_path)
        if parsed.get("status") == "error":
            print(f"  Failed to read file: {parsed.get('message')}")
            continue
            
        full_text = parsed["content"]
        if not full_text.strip():
            print(f"  Empty content in resume: {filename}")
            continue
            
        # 2. Extract Metadata
        print("  Extracting metadata via LLM...")
        metadata = extract_metadata(client, full_text)
        candidate_name = metadata.get("name", "Unknown") or "Unknown"
        experience_years = float(metadata.get("experience_years", 0.0) or 0.0)
        skills_list = metadata.get("skills", []) or []
        education_list = metadata.get("education", []) or []
        
        print(f"  Parsed Name: {candidate_name}")
        print(f"  Parsed Experience: {experience_years} years")
        print(f"  Parsed Skills Count: {len(skills_list)}")
        
        # 3. Chunk Document
        print("  Chunking resume...")
        chunks = chunk_resume(full_text)
        print(f"  Generated {len(chunks)} chunks.")
        
        # 4. Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        # Flatten skills and education for ChromaDB metadata (must be simple types)
        skills_str = ", ".join(skills_list)
        education_str = json.dumps(education_list)
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{chunk_idx}"
            documents.append(chunk["text"])
            ids.append(chunk_id)
            metadatas.append({
                "candidate_name": candidate_name,
                "resume_path": rel_path,
                "skills": skills_str,
                "experience_years": experience_years,
                "education": education_str,
                "section_type": chunk["section_type"],
                "chunk_index": chunk_idx
            })
            
        # 5. Add to ChromaDB
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"  Successfully stored chunks in ChromaDB.")
        except Exception as e:
            print(f"  Failed to store in ChromaDB: {str(e)}")
            
    print(f"\nIngestion pipeline complete. Total collection count: {collection.count()}")

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("./data", exist_ok=True)
    # Run ingestion on the local resumes folder
    ingest_resumes("resumes")
