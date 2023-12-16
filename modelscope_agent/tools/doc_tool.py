from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from modelscope_agent.tools.tool import Tool


# if __name__ == '__main__':
#     # load the pdf file from directory
#     loaders = [PyPDFLoader('data/doc/指数.txt'), TextLoader('data/doc/巴菲特的投资理念.txt')]
#     docs = []
#     for file in loaders:
#         docs.extend(file.load())

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
#     split_docs = text_splitter.split_documents(docs)

#     embeddings = HuggingFaceEmbeddings(
#         model_name='BAAI/bge-base-zh-v1.5',
#         cache_folder='/Users/fanque/.cache/huggingface/hub/',
#         model_kwargs={'device': 'cpu'})

#     vector_db = Chroma.from_documents(split_docs, embeddings, persist_directory='chroma')
#     vector_db.persist()

#     docs = vector_db.as_retriever(search_kwargs={"k": 5}).invoke(question)
#     context_doc_str = '\n\n'.join([f'{doc.metadata["source"]}\n{doc.page_content}' for doc in docs])

import os
from typing import Dict, Iterable, List, Union

import json
from langchain.document_loaders import (PyPDFLoader, TextLoader,
                                        UnstructuredFileLoader)
from langchain.embeddings import ModelScopeEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, VectorStore

SUPPORTED_KNOWLEDGE_TYPE = ['.txt', '.md', '.pdf']


class Retrieval:

    def __init__(self,
                 embedding: Embeddings = None,
                 vs_cls: VectorStore = None,
                 top_k: int = 5,
                 vs_params: Dict = {}):
        self.embedding = embedding or ModelScopeEmbeddings(
            model_id='damo/nlp_gte_sentence-embedding_chinese-base')
        self.top_k = top_k
        self.vs_cls = vs_cls or FAISS
        self.vs_params = vs_params
        self.vs: VectorStore = None

    def construct(self, docs):
        assert len(docs) > 0
        if isinstance(docs[0], str):
            self.vs = self.vs_cls.from_texts(docs, self.embedding,
                                             **self.vs_params)
        elif isinstance(docs[0], Document):
            self.vs = self.vs_cls.from_documents(docs, self.embedding,
                                                 **self.vs_params)

    def retrieve(self, query: str) -> List[str]:
        res = self.vs.similarity_search(query, k=self.top_k)
        if 'page' in res[0].metadata:
            res.sort(key=lambda doc: doc.metadata['page'])
        return [r.page_content for r in res]


class ToolRetrieval(Retrieval):

    def __init__(self,
                 embedding: Embeddings = None,
                 vs_cls: VectorStore = None,
                 top_k: int = 5,
                 vs_params: Dict = {}):
        super().__init__(embedding, vs_cls, top_k, vs_params)

    def retrieve(self, query: str) -> Dict[str, str]:
        res = self.vs.similarity_search(query, k=self.top_k)

        final_res = {}

        for r in res:
            content = r.page_content
            name = json.loads(content)['name']
            final_res[name] = content

        return final_res


class KnowledgeRetrieval(Retrieval):

    def __init__(self,
                 docs,
                 embedding: Embeddings = None,
                 vs_cls: VectorStore = None,
                 top_k: int = 5,
                 vs_params: Dict = {}):
        super().__init__(embedding, vs_cls, top_k, vs_params)
        self.docs = docs
        self.construct(docs)

    @staticmethod
    def file_preprocess(file_path):
        textsplitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        all_files = []
        if isinstance(file_path, str) and os.path.isfile(file_path):
            all_files.append(file_path)
        elif isinstance(file_path, list):
            all_files = file_path
        elif os.path.isdir(file_path):
            for root, dirs, files in os.walk(file_path):
                for f in files:
                    all_files.append(os.path.join(root, f))
        else:
            raise ValueError('file_path must be a file or a directory')

        docs = []
        for f in all_files:
            if f.lower().endswith('.txt'):
                loader = TextLoader(f, autodetect_encoding=True)
                docs += (loader.load_and_split(textsplitter))
            elif f.lower().endswith('.md'):
                loader = UnstructuredFileLoader(f, mode='elements')
                docs += loader.load()
            elif f.lower().endswith('.pdf'):
                loader = PyPDFLoader(f)
                docs += (loader.load_and_split(textsplitter))
            else:
                raise ValueError(
                    f'not support file type: {f}, will be support soon')
        return docs

    @classmethod
    def from_file(cls,
                  file_path: Union[str, list],
                  embedding: Embeddings = None,
                  vs_cls: VectorStore = None,
                  top_k: int = 5,
                  vs_params: Dict = {}):
        # default embedding and vs class
        if embedding is None:
            model_id = 'damo/nlp_gte_sentence-embedding_chinese-base'
            embedding = ModelScopeEmbeddings(model_id=model_id)
        if vs_cls is None:
            vs_cls = FAISS
        docs = KnowledgeRetrieval.file_preprocess(file_path)

        if len(docs) == 0:
            return None
        else:
            return cls(docs, embedding, vs_cls, top_k, vs_params)

    def add_file(self, file_path: Union[str, list]):
        docs = KnowledgeRetrieval.file_preprocess(file_path)
        self.add_docs(docs)

    def add_docs(self, docs):
        assert len(docs) > 0
        if isinstance(docs[0], str):
            self.vs.add_texts(docs, **self.vs_params)
        elif isinstance(docs[0], Document):
            self.vs.add_documents(docs, **self.vs_params)


class DocTool(Tool):
    '''
    '''

    description = '能返回pdf、txt或markdown中的相关信息，pdf文档读取请用这个工具'
    name = 'doc_search'
    parameters: list = [
        {
            'name': 'file_path',
            'description': '文件路径',
            'required': True
        },
        {
            'name': 'query',
            'description': '查询关键词',
            'required': True
        },
    ]

    def __init__(self, cfg={}):
        super().__init__(cfg)

    def __call__(self, *args, **kwargs):
        parsed_input : Dict[str ,str] = kwargs
        print('DocTool parsed_input', parsed_input)

        query = parsed_input['query']
        file = parsed_input['file_path']

        knowledge_retrieval = KnowledgeRetrieval.from_file([file])

        result = []
        if query.strip() != '':
            result = knowledge_retrieval.retrieve(query) if knowledge_retrieval else []
        else:
            for doc in knowledge_retrieval.docs[:5]:
                result.append(doc.page_content)

        result = '\n\n'.join([f'>>>part{i} start>>>\n\n{s}\n\n>>>part{i} end>>>' for i, s in enumerate(result)])
        
        result = self._parse_output(result)
        print('DocTool result', result)
        return result


if __name__ == '__main__':
    query = 'qwen'
    query = '  '
    l = DocTool()(**{'query': query, 'file_path': '/Users/fanque/workspace/project/ai/language-model/paper/识图/Qwen-VL- A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond.pdf'})
    print(l)
