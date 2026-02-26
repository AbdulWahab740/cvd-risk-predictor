"""
Clinical Guidelines Parser and Indexer for Elasticsearch
Extracts structured recommendations from cardiovascular guideline PDFs
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
# Configuration
GUIDELINES_DIR = "./RAG/utils/guidelines"  # Put your PDF files here
OUTPUT_JSON = "./RAG/guidelines_parsed.json"  # Intermediate storage
ES_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
ES_API_KEY = os.getenv("ELASTIC_API_KEY")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GUIDELINES_INDEX = "clinical_guidelines"

# Guideline metadata
GUIDELINE_METADATA = {
    "2017-acc-aha": {
        "guideline_name": "2017 ACC/AHA Guideline for High Blood Pressure in Adults",
        "organization": "ACC/AHA",
        "year": 2017,
        "primary_topic": "Hypertension",
        "url": "https://www.ahajournals.org/doi/10.1161/HYP.0000000000000065"
    },
    "2018-aha-acc": {
        "guideline_name": "2018 AHA/ACC Guideline on Management of Blood Cholesterol",
        "organization": "ACC/AHA",
        "year": 2018,
        "primary_topic": "Coronary_Artery_Disease",
        "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000000625"
    },
    "2019-acc-aha": {
        "guideline_name": "2019 ACC/AHA Guideline on Primary Prevention of CVD",
        "organization": "ACC/AHA",
        "year": 2019,
        "primary_topic": "Coronary_Artery_Disease",
        "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000000678"
    }
}

# Feature-to-Topic mapping
FEATURE_TOPIC_MAPPING = {
    "ap_hi": ["Hypertension", "Coronary_Artery_Disease"],
    "ap_lo": ["Hypertension", "Coronary_Artery_Disease"],
    "cholesterol": ["Coronary_Artery_Disease", "Hyperlipidemia"],
    "gluc": ["Diabetes", "Hypertension", "Coronary_Artery_Disease"],
    "smoke": ["Coronary_Artery_Disease", "Smoking_Cessation"],
    "alco": ["Coronary_Artery_Disease", "Lifestyle_Modification"],
    "active": ["Coronary_Artery_Disease", "Exercise", "Lifestyle_Modification"],
    "bmi": ["Obesity", "Coronary_Artery_Disease", "Lifestyle_Modification"]
}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file"""
    print(f"Extracting text from {pdf_path}...")
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text


def extract_recommendation_class(text: str) -> str:
    """
    Extract recommendation class from text
    Looks for: Class I, Class IIa, Class IIb, Class III
    """
    # Common patterns in ACC/AHA guidelines
    class_patterns = [
        r"Class\s+(I{1,3}[ab]?)",
        r"\(Class\s+(I{1,3}[ab]?)\)",
        r"COR\s+(I{1,3}[ab]?)",
        r"Recommendation\s+Class\s+(I{1,3}[ab]?)"
    ]
    
    for pattern in class_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            rec_class = match.group(1).upper()
            # Normalize
            if rec_class == "I":
                return "Class I"
            elif rec_class in ["IIA", "IIA"]:
                return "Class IIa"
            elif rec_class in ["IIB", "IIB"]:
                return "Class IIb"
            elif rec_class == "III":
                return "Class III"
    
    return None


def extract_evidence_level(text: str) -> str:
    """
    Extract evidence level from text
    Looks for: Level A, B-R, B-NR, C-LD, C-EO
    """
    evidence_patterns = [
        r"Level\s+([A-C](?:-[A-Z]{1,2})?)",
        r"\(Level\s+([A-C](?:-[A-Z]{1,2})?)\)",
        r"LOE\s+([A-C](?:-[A-Z]{1,2})?)",
        r"Evidence\s+Level\s+([A-C](?:-[A-Z]{1,2})?)"
    ]
    
    for pattern in evidence_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            level = match.group(1).upper()
            return f"Level {level}"
    
    return None


def intelligent_chunk_guideline(text: str, guideline_meta: Dict, chunk_size: int = 800) -> List[Dict]:
    """
    Intelligently chunk guideline text into semantically meaningful sections
    """
    chunks = []
    
    # Split by recommendation sections (common in ACC/AHA guidelines)
    # Look for patterns like "Recommendation 1.1" or numbered sections
    section_pattern = r"(?:Recommendation|Section)\s+\d+(?:\.\d+)?[:\.]?\s*(.+?)(?=(?:Recommendation|Section)\s+\d+|\Z)"
    
    sections = re.findall(section_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if not sections:
        # Fallback: simple chunking by paragraph
        paragraphs = text.split('\n\n')
        sections = []
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    sections.append(current_chunk)
                current_chunk = para + "\n\n"
        if current_chunk:
            sections.append(current_chunk)
    
    # Process each section
    for i, section_text in enumerate(sections):
        # Clean text
        section_text = section_text.strip()
        if len(section_text) < 100:  # Skip very short sections
            continue
        
        # Extract section title (first sentence or line)
        lines = section_text.split('\n')
        section_title = lines[0].strip() if lines else f"Section {i+1}"
        
        # Truncate long titles
        if len(section_title) > 200:
            section_title = section_title[:200] + "..."
        
        # Extract recommendation class and evidence level
        rec_class = extract_recommendation_class(section_text)
        evidence_level = extract_evidence_level(section_text)
        
        # Infer topics based on keywords
        topics = infer_topics_from_text(section_text, guideline_meta['primary_topic'])
        
        chunk = {
            "guideline_name": guideline_meta['guideline_name'],
            "guideline_organization": guideline_meta['organization'],
            "publication_year": guideline_meta['year'],
            "guideline_url": guideline_meta.get('url', ''),
            "section_title": section_title,
            "content": section_text,
            "recommendation_class": rec_class,
            "evidence_level": evidence_level,
            "topic": guideline_meta['primary_topic'],  # Primary topic
            "related_topics": topics,  # Additional related topics
            "chunk_id": f"{guideline_meta['guideline_name']}_chunk_{i}"
        }
        
        chunks.append(chunk)
    
    return chunks


def infer_topics_from_text(text: str, primary_topic: str) -> List[str]:
    """
    Infer related topics from text based on keywords
    """
    topics = [primary_topic]  # Always include primary
    
    # Keyword-based topic detection
    topic_keywords = {
        "Hypertension": ["blood pressure", "hypertension", "hypertensive", "systolic", "diastolic", "BP"],
        "Coronary_Artery_Disease": ["coronary", "CAD", "atherosclerosis", "plaque", "angina"],
        "Heart_Failure": ["heart failure", "HF", "ejection fraction", "HFrEF", "HFpEF"],
        "Hyperlipidemia": ["cholesterol", "LDL", "statin", "lipid", "hyperlipidemia"],
        "Diabetes": ["diabetes", "glucose", "glycemic", "HbA1c", "insulin"],
        "Smoking_Cessation": ["smoking", "tobacco", "nicotine", "cessation", "quit smoking"],
        "Exercise": ["physical activity", "exercise", "sedentary", "fitness"],
        "Obesity": ["obesity", "weight loss", "BMI", "overweight"],
        "Lifestyle_Modification": ["lifestyle", "diet", "DASH", "Mediterranean", "nutrition"]
    }
    
    text_lower = text.lower()
    for topic, keywords in topic_keywords.items():
        if topic != primary_topic:  # Don't duplicate primary
            if any(keyword.lower() in text_lower for keyword in keywords):
                topics.append(topic)
    
    return list(set(topics))  # Remove duplicates


def parse_all_guidelines(guidelines_dir: str) -> List[Dict]:
    """
    Parse all guideline PDFs in directory
    """
    all_chunks = []
    
    pdf_files = list(Path(guidelines_dir).glob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        filename = pdf_file.name
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
        
        # Get metadata using partial match
        meta = None
        for key, m in GUIDELINE_METADATA.items():
            if key in filename.lower():
                meta = m
                break
        
        if not meta:
            print(f"Warning: No metadata match for {filename}, skipping")
            continue
        
        # Extract text
        text = extract_text_from_pdf(str(pdf_file))
        if not text:
            print(f"Failed to extract text from {filename}")
            continue
        
        print(f"Extracted {len(text)} characters")
        
        # Chunk intelligently
        chunks = intelligent_chunk_guideline(text, meta)
        print(f"Created {len(chunks)} chunks")
        
        # Show sample
        if chunks:
            print(f"\nSample chunk:")
            print(f"  Title: {chunks[0]['section_title'][:100]}...")
            print(f"  Class: {chunks[0]['recommendation_class']}")
            print(f"  Evidence: {chunks[0]['evidence_level']}")
            print(f"  Topics: {chunks[0]['related_topics']}")
        
        all_chunks.extend(chunks)
    
    print(f"\n{'='*60}")
    print(f"Total chunks extracted: {len(all_chunks)}")
    print(f"{'='*60}")
    
    return all_chunks


def save_parsed_guidelines(chunks: List[Dict], output_path: str):
    """Save parsed chunks to JSON file"""
    print(f"\nSaving parsed guidelines to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks")


def create_guidelines_index(es: Elasticsearch):
    """Create Elasticsearch index for guidelines"""
    mapping = {
        "mappings": {
            "properties": {
                "guideline_name": {"type": "keyword"},
                "guideline_organization": {"type": "keyword"},
                "publication_year": {"type": "integer"},
                "guideline_url": {"type": "keyword"},
                "section_title": {"type": "text"},
                "content": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "recommendation_class": {"type": "keyword"},
                "evidence_level": {"type": "keyword"},
                "topic": {"type": "keyword"},
                "related_topics": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    # Delete if exists
    if es.indices.exists(index=GUIDELINES_INDEX):
        print(f"Deleting existing index: {GUIDELINES_INDEX}")
        es.indices.delete(index=GUIDELINES_INDEX)
    
    # Create new
    print(f"Creating index: {GUIDELINES_INDEX}")
    es.indices.create(index=GUIDELINES_INDEX, body=mapping)


def index_guidelines_to_elasticsearch(chunks: List[Dict]):
    """Index guideline chunks into Elasticsearch with embeddings"""
    print("\n" + "="*60)
    print("Indexing Guidelines to Elasticsearch")
    print("="*60)
    
    # Connect to ES
    es = Elasticsearch(
        cloud_id=ES_CLOUD_ID,
        api_key=ES_API_KEY
    )
    
    # Create index
    create_guidelines_index(es)
    
    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, cache_folder='./models')
    
    # Generate embeddings
    print(f"Generating embeddings for {len(chunks)} chunks...")
    texts = [chunk['content'] for chunk in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)
    
    # Add embeddings to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk['embedding'] = embedding.tolist()
    
    # Bulk index
    print(f"\nBulk indexing to Elasticsearch...")
    
    def generate_actions():
        for chunk in chunks:
            yield {
                "_index": GUIDELINES_INDEX,
                "_id": chunk['chunk_id'],
                "_source": chunk
            }
    
    success, failed = helpers.bulk(es, generate_actions(), chunk_size=50)
    
    print(f"Successfully indexed: {success}")
    print(f"Failed: {len(failed) if failed else 0}")
    
    # Verify
    count = es.count(index=GUIDELINES_INDEX)['count']
    print(f"\nTotal documents in index: {count}")
    
    return es


def test_guideline_search(es: Elasticsearch, embedder: SentenceTransformer):
    """Test searching the guidelines index"""
    print("\n" + "="*60)
    print("Testing Guideline Search")
    print("="*60)
    
    test_queries = [
        {
            "query": "blood pressure management targets high risk",
            "topics": ["Hypertension"],
            "description": "Hypertension treatment"
        },
        {
            "query": "statin therapy cholesterol LDL targets",
            "topics": ["Coronary_Artery_Disease"],
            "description": "Cholesterol management"
        },
        {
            "query": "smoking cessation pharmacotherapy",
            "topics": ["Smoking_Cessation", "Coronary_Artery_Disease"],
            "description": "Smoking cessation"
        }
    ]
    
    for test in test_queries:
        print(f"\n{'='*60}")
        print(f"Test: {test['description']}")
        print(f"Query: {test['query']}")
        print(f"Topics: {test['topics']}")
        print(f"{'='*60}")
        
        # Generate query embedding
        query_embedding = embedder.encode(test['query']).tolist()
        
        # Hybrid search
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": test['query'],
                                "fields": ["section_title^3", "content^2"],
                                "type": "best_fields"
                            }
                        }
                    ],
                    "filter": [
                        {"terms": {"related_topics": test['topics']}}
                    ]
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": 10,
                "num_candidates": 50
            },
            "size": 3,
            "_source": ["guideline_name", "section_title", "content", 
                       "recommendation_class", "evidence_level", "topic"]
        }
        
        results = es.search(index=GUIDELINES_INDEX, body=search_body)
        
        print(f"\nFound {results['hits']['total']['value']} results")
        print(f"Showing top {len(results['hits']['hits'])}:\n")
        
        for i, hit in enumerate(results['hits']['hits'], 1):
            doc = hit['_source']
            print(f"Result {i}:")
            print(f"  Guideline: {doc['guideline_name']}")
            print(f"  Section: {doc['section_title'][:100]}...")
            print(f"  Class: {doc.get('recommendation_class', 'N/A')}")
            print(f"  Evidence: {doc.get('evidence_level', 'N/A')}")
            print(f"  Content: {doc['content'][:200]}...")
            print(f"  Score: {hit['_score']:.4f}")
            print()


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("Clinical Guidelines Parser and Indexer")
    print("="*80)
    
    # Check guidelines directory
    if not os.path.exists(GUIDELINES_DIR):
        print(f"\nError: Guidelines directory not found: {GUIDELINES_DIR}")
        print("Please create the directory and add your PDF files:")
        for filename in GUIDELINE_METADATA.keys():
            print(f"  - {filename}")
        return
    
    # Parse guidelines
    chunks = parse_all_guidelines(GUIDELINES_DIR)
    
    if not chunks:
        print("\nNo chunks extracted. Check your PDF files.")
        return
    
    # Save intermediate JSON
    save_parsed_guidelines(chunks, OUTPUT_JSON)
    
    # Index to Elasticsearch
    if ES_CLOUD_ID and ES_API_KEY:
        es = index_guidelines_to_elasticsearch(chunks)
        
        # Test search
        embedder = SentenceTransformer(EMBEDDING_MODEL, cache_folder='./models')
        test_guideline_search(es, embedder)
    else:
        print("\nSkipping Elasticsearch indexing (no credentials)")
        print("Set ELASTIC_CLOUD_ID and ELASTIC_API_KEY to enable")
    
    print("\n" + "="*80)
    print("Processing Complete!")
    print("="*80)
    print(f"\nParsed chunks saved to: {OUTPUT_JSON}")
    if ES_CLOUD_ID and ES_API_KEY:
        print(f"Indexed to Elasticsearch: {GUIDELINES_INDEX}")


if __name__ == "__main__":
    main()