"""
MindBridge 2.0 - Governed Knowledge Retrieval System
=====================================================

Package exports for dense vector retrieval, governance enforcement,
document ingestion, and vector store persistence.
"""

from backend.retrieval.governance import (
    DOCUMENT_GOVERNANCE_REGISTRY,
    DocumentGovernance,
    RetrievalTier,
    ClinicalSafetyLevel,
    get_document_governance,
    is_document_eligible_for_ingestion,
    get_all_governed_documents,
    get_counts_by_tier,
)
from backend.retrieval.ingestion import (
    MarkdownSectionChunker,
    KnowledgeBaseIngestor,
    parse_frontmatter_safe,
    load_compatible_tokenizer,
)
from backend.retrieval.embeddings import (
    BGEEmbeddingEngine,
    MockEmbeddingEngine,
    EmbeddingError,
    EMBEDDING_DIMENSION,
    DEFAULT_BGE_MODEL,
    BGE_QUERY_PREFIX,
    get_default_embedding_engine,
)
from backend.retrieval.vector_store import (
    FAISSVectorStore,
    IndexIntegrityError,
    PROPOSED_RETRIEVAL_THRESHOLD,
    INDEX_FILE_NAME,
    METADATA_FILE_NAME,
)
from backend.retrieval.retriever import (
    RAGRetriever,
    RetrievalAuthorization,
    format_citations,
    filter_used_citations,
    build_retriever_from_knowledge_base,
)
from backend.retrieval.counselchat_retriever import (
    CounselChatRetriever,
    MultiCollectionRetriever,
    build_counselchat_faiss_index,
    build_counselchat_experimental_dev_subset,
    build_consolidated_counselchat_faiss_index,
    chunk_consolidated_counselchat_dataset,
)

__all__ = [
    # Governance
    "DOCUMENT_GOVERNANCE_REGISTRY",
    "DocumentGovernance",
    "RetrievalTier",
    "ClinicalSafetyLevel",
    "get_document_governance",
    "is_document_eligible_for_ingestion",
    "get_all_governed_documents",
    "get_counts_by_tier",
    # Ingestion
    "MarkdownSectionChunker",
    "KnowledgeBaseIngestor",
    "parse_frontmatter_safe",
    "load_compatible_tokenizer",
    # Embeddings
    "BGEEmbeddingEngine",
    "MockEmbeddingEngine",
    "EmbeddingError",
    "EMBEDDING_DIMENSION",
    "DEFAULT_BGE_MODEL",
    "BGE_QUERY_PREFIX",
    "get_default_embedding_engine",
    # Vector Store
    "FAISSVectorStore",
    "IndexIntegrityError",
    "PROPOSED_RETRIEVAL_THRESHOLD",
    "INDEX_FILE_NAME",
    "METADATA_FILE_NAME",
    # Retriever
    "RAGRetriever",
    "RetrievalAuthorization",
    "format_citations",
    "filter_used_citations",
    "build_retriever_from_knowledge_base",
    # CounselChat RAG
    "CounselChatRetriever",
    "MultiCollectionRetriever",
    "build_counselchat_faiss_index",
    "build_counselchat_experimental_dev_subset",
    "build_consolidated_counselchat_faiss_index",
    "chunk_consolidated_counselchat_dataset",
]
