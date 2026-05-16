import time
import chromadb
from chromadb.utils import embedding_functions

TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days

class VectorStore:

    def __init__(self):

        # Create persistent client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Use default embedding model
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        # Create collection
        self.collection = self.client.get_or_create_collection(
            name="research_memory",
            embedding_function=self.embedding_function
        )

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
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "distances"]
            )
            docs = results["documents"][0]
            distances = results["distances"][0]
            scored = sorted(
                [{"content": doc, "score": 1 - dist} for doc, dist in zip(docs, distances)],
                key=lambda x: x["score"],
                reverse=True
            )
            return [item for item in scored if item["score"] > 0.3]
        except Exception:
            return []