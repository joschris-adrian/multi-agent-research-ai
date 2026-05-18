import time
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days

class VectorStore:

    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="research_memory",
            embedding_function=self.embedding_function
        )
        try:
            self.reranker = CrossEncoder(RERANKER_MODEL)
            print("[vector_store] reranker loaded")
        except Exception as e:
            print(f"[vector_store] reranker unavailable: {e}")
            self.reranker = None

    def add_documents(self, documents):
        now = int(time.time())
        for i, doc in enumerate(documents):
            text = f"""
            Title: {doc['title']}
            Content: {doc['content']}
            Source: {doc['source']}
            """
            self.collection.add(
                documents=[text],
                ids=[f"doc_{i}_{hash(text)}"],
                metadatas=[{"timestamp": now}]
            )

    def evict_expired(self):
        now = int(time.time())
        try:
            all_items = self.collection.get(include=["metadatas"])
            expired_ids = [
                id_ for id_, meta in zip(all_items["ids"], all_items["metadatas"])
                if now - meta.get("timestamp", now) > TTL_SECONDS
            ]
            if expired_ids:
                self.collection.delete(ids=expired_ids)
                print(f"[vector_store] evicted {len(expired_ids)} expired chunks")
        except Exception as e:
            print(f"[vector_store] eviction failed: {e}")
                        
    def search(self, query, top_k=5):
        self.evict_expired()
        try:
            # fetch more candidates than needed for reranker to work with
            candidate_k = top_k * 3 if self.reranker else top_k
            results = self.collection.query(
                query_texts=[query],
                n_results=candidate_k,
                include=["documents", "distances"]
            )
            docs = results["documents"][0]
            distances = results["distances"][0]

            if not docs:
                return []

            # initial cosine scoring
            scored = [
                {"content": doc, "score": 1 - dist}
                for doc, dist in zip(docs, distances)
            ]

            # rerank if available
            if self.reranker:
                pairs = [[query, item["content"]] for item in scored]
                rerank_scores = self.reranker.predict(pairs)
                for item, rerank_score in zip(scored, rerank_scores):
                    item["rerank_score"] = float(rerank_score)
                scored = sorted(scored, key=lambda x: x["rerank_score"], reverse=True)
            else:
                scored = sorted(scored, key=lambda x: x["score"], reverse=True)

            # return top_k after reranking, filtered by initial cosine threshold
            return [
                item for item in scored[:top_k]
                if item["score"] > 0.3
            ]
        except Exception as e:
            print(f"[vector_store] search failed: {e}")
            return []