import json
import os
import logging
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
# --- IMPORT TOPICS FROM data_fetch.py ---
try:
    from utils.data_fetch import core_topics
    core_topic = core_topics()
    logging.info("Successfully imported core_topics from data_fetch.py")
except ImportError:
    logging.error("Error: Could not import core_topics from data_fetch.py. "
                  "Please ensure data_fetch.py exists and contains 'core_topics'.")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DATA_DIR = "./utils/data"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Elasticsearch Configuration
ES_CLOUD_ID = st.secrets["ELASTIC_CLOUD_ID", None]  # Set via environment variable
ES_API_KEY = st.secrets["ELASTIC_API_KEY", None]  # Set via environment variable
ES_USERNAME = st.secrets["ELASTIC_USERNAME", "elastic"]
ES_PASSWORD = st.secrets["ELASTIC_PASSWORD", None]
ES_HOST = st.secrets["ELASTICSEARCH_HOST", "http://localhost:9200"]  # For local ES

# Index names
PUBMED_INDEX = "pubmed_cardio_abstracts"
GUIDELINES_INDEX = "clinical_guidelines"

# Batch size for bulk indexing
BATCH_SIZE = 100

# --- Elasticsearch Index Mappings ---
PUBMED_MAPPING = {
    "mappings": {
        "properties": {
            "pmid": {"type": "keyword"},
            "pmcid": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "standard"
            },
            "abstract": {
                "type": "text",
                "analyzer": "standard"
            },
            "content_for_embedding": {
                "type": "text",
                "analyzer": "standard"
            },
            "journal": {"type": "keyword"},
            "publication_year": {"type": "text"},
            "topic": {
                "type": "keyword"
            },
            "embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMENSION,
                "index": True,
                "similarity": "cosine"
            }
        }
    },

}


def get_elasticsearch_client() -> Elasticsearch:
    """
    Create and return Elasticsearch client based on available configuration.
    Priority: Cloud ID > Local connection
    """
    logging.info("Connecting to Elasticsearch...")
    
    try:
        # Option 1: Elastic Cloud with Cloud ID and API Key
        if ES_CLOUD_ID and ES_API_KEY:
            logging.info("Using Elastic Cloud connection (Cloud ID + API Key)")
            es = Elasticsearch(
                cloud_id=ES_CLOUD_ID,
                api_key=ES_API_KEY,
                request_timeout=60
            )
        
        # Option 2: Elastic Cloud with Cloud ID and Username/Password
        elif ES_CLOUD_ID and ES_PASSWORD:
            logging.info("Using Elastic Cloud connection (Cloud ID + Username/Password)")
            es = Elasticsearch(
                cloud_id=ES_CLOUD_ID,
                basic_auth=(ES_USERNAME, ES_PASSWORD),
                request_timeout=60
            )
        
        # Option 3: Local Elasticsearch
        else:
            logging.info(f"Using local Elasticsearch connection: {ES_HOST}")
            es = Elasticsearch(
                [ES_HOST],
                basic_auth=(ES_USERNAME, ES_PASSWORD) if ES_PASSWORD else None,
                request_timeout=60
            )
        
        # Test connection
        if es.ping():
            info = es.info()
            logging.info(f"Successfully connected to Elasticsearch: {info['version']['number']}")
            return es
        else:
            raise ConnectionError("Failed to ping Elasticsearch")
            
    except Exception as e:
        logging.error(f"Failed to connect to Elasticsearch: {e}")
        logging.error("Please check your connection settings:")
        logging.error(f"  - ES_CLOUD_ID: {'Set' if ES_CLOUD_ID else 'Not set'}")
        logging.error(f"  - ES_API_KEY: {'Set' if ES_API_KEY else 'Not set'}")
        logging.error(f"  - ES_PASSWORD: {'Set' if ES_PASSWORD else 'Not set'}")
        logging.error(f"  - ES_HOST: {ES_HOST}")
        exit(1)


def create_index_if_not_exists(es: Elasticsearch, index_name: str, mapping: Dict[str, Any]):
    """
    Create Elasticsearch index with given mapping if it doesn't exist.
    If it exists, delete and recreate to ensure clean state.
    """
    try:
        if es.indices.exists(index=index_name):
            logging.warning(f"Index '{index_name}' already exists. Deleting and recreating...")
            es.indices.delete(index=index_name)
            logging.info(f"Deleted existing index '{index_name}'")
        
        es.indices.create(index=index_name, body=mapping)
        logging.info(f"Created index '{index_name}' successfully")
        
    except Exception as e:
        logging.error(f"Failed to create index '{index_name}': {e}")
        raise


def generate_embedding(text: str, model: SentenceTransformer) -> List[float]:
    """Generate embedding vector for given text"""
    try:
        embedding = model.encode(text, show_progress_bar=False)
        return embedding.tolist()
    except Exception as e:
        logging.error(f"Failed to generate embedding: {e}")
        return None


def prepare_pubmed_documents(topic_name: str, topic_info: Dict[str, Any], 
                             embedding_model: SentenceTransformer) -> List[Dict[str, Any]]:
    """
    Load and prepare documents from JSONL file for a specific topic.
    Returns list of documents ready for Elasticsearch indexing.
    """
    documents = []
    
    topic_jsonl_filename = f"{topic_name.lower()}_articles_data.jsonl"
    topic_jsonl_path = os.path.join(DATA_DIR, topic_name, topic_jsonl_filename)
    
    logging.info(f"Processing topic: {topic_name}")
    logging.info(f"Looking for file: {topic_jsonl_path}")
    
    if not os.path.exists(topic_jsonl_path):
        logging.warning(f"JSONL file not found for topic '{topic_name}'. Skipping.")
        return documents
    
    processed_count = 0
    
    with open(topic_jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        for line_num, line in enumerate(tqdm(lines, desc=f"Processing {topic_name}")):
            try:
                article_meta = json.loads(line)
                pmid = article_meta.get("pmid")
                
                if not pmid:
                    logging.warning(f"Line {line_num+1}: Missing PMID. Skipping.")
                    continue
                
                title = article_meta.get("title", "").strip()
                abstract = article_meta.get("abstract", "").strip()
                content_for_embedding = article_meta.get("content_for_embedding", "").strip()
                
                if not content_for_embedding:
                    logging.debug(f"No 'content_for_embedding' for PMID {pmid}. Skipping.")
                    continue
                
                # Create combined text for embedding (title + content)
                text_for_embedding = ""
                if title:
                    text_for_embedding += title + ". "
                text_for_embedding += content_for_embedding
                
                if not text_for_embedding.strip():
                    logging.debug(f"No text for embedding for PMID {pmid}. Skipping.")
                    continue
                
                # Generate embedding
                embedding = generate_embedding(text_for_embedding, embedding_model)
                if embedding is None:
                    logging.warning(f"Failed to generate embedding for PMID {pmid}. Skipping.")
                    continue
                
                # Prepare document for Elasticsearch
                doc = {
                    "pmid": pmid,
                    "pmcid": article_meta.get("pmcid", "N/A"),
                    "title": title,
                    "abstract": abstract,
                    "content_for_embedding": content_for_embedding,
                    "journal": article_meta.get("journal", "N/A"),
                    "publication_year": article_meta.get("publication_year", "N/A"),
                    "topic": topic_name,
                    "embedding": embedding
                }
                
                documents.append(doc)
                processed_count += 1
                
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error at line {line_num+1}: {e}")
            except Exception as e:
                logging.error(f"Error processing line {line_num+1} for {topic_name}: {e}")
    
    logging.info(f"Prepared {processed_count} documents from topic '{topic_name}'")
    return documents


def bulk_index_documents(es: Elasticsearch, index_name: str, documents: List[Dict[str, Any]]):
    """
    Bulk index documents into Elasticsearch in batches.
    """
    if not documents:
        logging.warning("No documents to index")
        return
    
    logging.info(f"Starting bulk indexing of {len(documents)} documents to '{index_name}'...")
    
    def generate_actions():
        """Generator for bulk API"""
        for doc in documents:
            yield {
                "_index": index_name,
                "_id": doc.get("pmid", None),  # Use PMID as document ID
                "_source": doc
            }
    
    try:
        # Use helpers.bulk for efficient batch indexing
        success, failed = helpers.bulk(
            es,
            generate_actions(),
            chunk_size=BATCH_SIZE,
            raise_on_error=False,
            stats_only=False
        )
        
        logging.info(f"Successfully indexed {success} documents")
        if failed:
            logging.warning(f"Failed to index {len(failed)} documents")
            for item in failed[:5]:  # Show first 5 failures
                logging.error(f"Failed item: {item}")
        
    except Exception as e:
        logging.error(f"Bulk indexing failed: {e}")
        raise


def verify_index(es: Elasticsearch, index_name: str):
    """
    Verify index was created successfully and show document count.
    """
    try:
        if es.indices.exists(index=index_name):
            doc_count = es.count(index=index_name)['count']
            logging.info(f"Index '{index_name}' contains {doc_count} documents")
            
            # Get sample document
            sample = es.search(index=index_name, body={"query": {"match_all": {}}, "size": 1})
            if sample['hits']['hits']:
                logging.info(f"Sample document from '{index_name}':")
                doc = sample['hits']['hits'][0]['_source']
                logging.info(f"  PMID: {doc.get('pmid', 'N/A')}")
                logging.info(f"  Title: {doc.get('title', 'N/A')[:100]}...")
                logging.info(f"  Topic: {doc.get('topic', 'N/A')}")
                logging.info(f"  Embedding dimension: {len(doc.get('embedding', []))}")
        else:
            logging.error(f"Index '{index_name}' does not exist!")
            
    except Exception as e:
        logging.error(f"Error verifying index '{index_name}': {e}")


def test_hybrid_search(es: Elasticsearch, index_name: str, embedding_model: SentenceTransformer):
    """
    Test hybrid search (keyword + vector search) on the created index.
    """
    logging.info("\n" + "="*80)
    logging.info("Testing Hybrid Search")
    logging.info("="*80)
    
    test_query = "hypertension blood pressure management treatment"
    logging.info(f"Test query: '{test_query}'")
    
    # Generate query embedding
    query_embedding = generate_embedding(test_query, embedding_model)
    
    # Hybrid search query
    search_body = {
        "query": {
            "bool": {
                "should": [
                    # Keyword search (BM25)
                    {
                        "multi_match": {
                            "query": test_query,
                            "fields": ["title^3", "abstract^2", "content_for_embedding"],
                            "type": "best_fields"
                        }
                    }
                ],
                "filter": [
                    {"term": {"topic": "Hypertension"}}
                ]
            }
        },
        "knn": {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": 10,
            "num_candidates": 100
        },
        "size": 5,
        "_source": ["pmid", "title", "abstract", "topic", "publication_year"]
    }
    
    try:
        results = es.search(index=index_name, body=search_body)
        
        logging.info(f"\nFound {results['hits']['total']['value']} total results")
        logging.info(f"Showing top {len(results['hits']['hits'])} results:\n")
        
        for i, hit in enumerate(results['hits']['hits'], 1):
            doc = hit['_source']
            logging.info(f"Result {i}:")
            logging.info(f"  PMID: {doc.get('pmid', 'N/A')}")
            logging.info(f"  Title: {doc.get('title', 'N/A')}")
            logging.info(f"  Topic: {doc.get('topic', 'N/A')}")
            logging.info(f"  Year: {doc.get('publication_year', 'N/A')}")
            logging.info(f"  Score: {hit['_score']:.4f}")
            logging.info(f"  Abstract: {doc.get('abstract', 'N/A')[:200]}...")
            logging.info("")
        
        logging.info("Hybrid search test completed successfully!")
        
    except Exception as e:
        logging.error(f"Hybrid search test failed: {e}")


# --- Main function to build the vector store ---
def build_elasticsearch_vectorstore():
    """
    Main function to build Elasticsearch vector store from medical research articles.
    """
    logging.info("="*80)
    logging.info("Starting Elasticsearch Vector Store Building Process")
    logging.info("="*80)
    
    # 1. Connect to Elasticsearch
    es = get_elasticsearch_client()
    
    # 2. Load embedding model
    logging.info(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder='./models')
        logging.info("Embedding model loaded successfully")
        logging.info(f"Embedding dimension: {embedding_model.get_sentence_embedding_dimension()}")
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}")
        return
    
    # 3. Create index
    create_index_if_not_exists(es, PUBMED_INDEX, PUBMED_MAPPING)
    
    # 4. Process all topics and prepare documents
    all_documents = []
    
    for topic_name, topic_info in core_topic.items():
        topic_documents = prepare_pubmed_documents(topic_name, topic_info, embedding_model)
        all_documents.extend(topic_documents)
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Total documents prepared: {len(all_documents)}")
    logging.info(f"{'='*80}\n")
    
    if not all_documents:
        logging.error("No documents to index. Exiting.")
        return
    
    # 5. Bulk index documents
    bulk_index_documents(es, PUBMED_INDEX, all_documents)
    
    # 6. Verify indexing
    logging.info("\nVerifying index...")
    verify_index(es, PUBMED_INDEX)
    
    # 7. Test hybrid search
    test_hybrid_search(es, PUBMED_INDEX, embedding_model)
    
    logging.info("\n" + "="*80)
    logging.info("Elasticsearch Vector Store Creation Complete!")
    logging.info("="*80)
    logging.info(f"\nIndex name: {PUBMED_INDEX}")
    logging.info(f"Total documents indexed: {len(all_documents)}")
    logging.info("\nYou can now use this index with Elastic Agent Builder!")
    logging.info("\nNext steps:")
    logging.info("  1. Go to Kibana → Agent Builder")
    logging.info("  2. Create a new agent")
    logging.info(f"  3. Add a search tool pointing to index: {PUBMED_INDEX}")
    logging.info("  4. Configure hybrid search with vector + keyword search")


if __name__ == "__main__":
    # Check for required environment variables
    if not any([ES_CLOUD_ID, ES_HOST]):
        logging.warning("\n" + "="*80)
        logging.warning("No Elasticsearch connection configured!")
        logging.warning("="*80)
        logging.warning("\nPlease set one of the following:")
        logging.warning("\nFor Elastic Cloud:")
        logging.warning("  export ELASTIC_CLOUD_ID='your-cloud-id'")
        logging.warning("  export ELASTIC_API_KEY='your-api-key'")
        logging.warning("  # OR")
        logging.warning("  export ELASTIC_PASSWORD='your-password'")
        logging.warning("\nFor Local Elasticsearch:")
        logging.warning("  export ELASTICSEARCH_HOST='http://localhost:9200'")
        logging.warning("  export ELASTIC_PASSWORD='your-password'  # if auth enabled")
        logging.warning("\n" + "="*80 + "\n")
    
    build_elasticsearch_vectorstore()