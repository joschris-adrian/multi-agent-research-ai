import chromadb
from chromadb.utils import embedding_functions


class VectorStore:

    def __init__(self):

        # Create persistent client
        self.client = chromadb.Client()

        # Use default embedding model
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        # Create collection
        self.collection = self.client.get_or_create_collection(
            name="research_memory",
            embedding_function=self.embedding_function
        )

    def add_documents(self, documents):

        for i, doc in enumerate(documents):

            text = f"""
            Title: {doc['title']}
            Content: {doc['content']}
            Source: {doc['source']}
            """

            self.collection.add(
                documents=[text],
                ids=[f"doc_{i}_{hash(text)}"]
            )

    def search(self, query, n_results=5):

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        return results["documents"][0]