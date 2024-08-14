import google.generativeai as genai
class Embeddings():
    def __init__(self,  api_key, model='models/text-embedding-004', dim=64):
        self.model, self.dim = model, dim
        genai.configure(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = [genai.embed_content(model=self.model, content=text,
                                          task_type='RETRIEVAL_DOCUMENT',
                                          output_dimensionality=self.dim)['embedding']
                      for text in texts]
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return genai.embed_content(model=self.model, content=text, task_type='RETRIEVAL_DOCUMENT',
                                   output_dimensionality=self.dim)['embedding']
