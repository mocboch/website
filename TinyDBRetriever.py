import numpy as np
import pandas as pd
from GoogleEmbeddings import Embeddings
from tinydb import TinyDB, Query
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


class TinyDBRetriever():
    def __init__(self, tinydb_filepath: str, google_api_key: str, k: int):
        self.tinydb_filepath = tinydb_filepath
        self.google_api_key = google_api_key
        self.k = k

    def embedQuery(self, query: str):
        embeddings = Embeddings(api_key=self.google_api_key)
        embedded_query = embeddings.embed_query(query)
        return embedded_query

    def getVecSearch(self) -> list[tuple[int, list[float]]]:
        db = TinyDB(self.tinydb_filepath)
        table = db.table('_default')
        Q = Query()
        vec_search = [tuple((t.doc_id, t['question-embedded'])) for t in table.all()]
        db.close()
        return vec_search

    def getSimilarityScores(self, query: list[float], keys: list[tuple[int, list[float]]]) -> pd.DataFrame():
        scores = []
        for tup in keys:
            num = np.dot(query, tup[1])
            denom = np.sqrt(np.dot(query, query) * np.dot(tup[1], tup[1]))
            scores.append(tuple((tup[0], num/denom)))
        return pd.DataFrame(scores).set_index(0).sort_values(ascending=False, by=1).rename(columns={0:'doc_id', 1:'score'})

    def _get_relevant_documents(self, query: str) -> list[Document]:

        embedded_query = self.embedQuery(query)
        vecsearch = self.getVecSearch()
        scores = self.getSimilarityScores(embedded_query, vecsearch)[:self.k]

        db = TinyDB(self.tinydb_filepath)
        table = db.table('_default')
        Q = Query()

        docs = [Document(
            page_content=table.get(doc_id=doc[0])['answer'],
            metadata={"question": table.get(doc_id=doc[0])['question']}) for doc in scores.iterrows()]

        return docs

    def as_retriever(self):
        return RunnableLambda(self._get_relevant_documents)





