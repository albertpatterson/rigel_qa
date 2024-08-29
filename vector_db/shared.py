from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import gc
import torch
from llm.llm import LLM_Manager

PERSISTED_DATA_PATH = "data/chroma.db"
ALL_DATA_COLLECTION_NAME = "all_text"

# emb_model_name = "BAAI/bge-m3"
# emb_model = None


def get_emb_model():
    return LLM_Manager.get_model("embedding")

    # global emb_model
    # if emb_model is not None:
    #     return emb_model
    # emb_model = SentenceTransformer(emb_model_name, device="cuda")
    # emb_model.half()
    # return emb_model


# def clear_emb_model():
#     global emb_model

#     if emb_model is not None:
#         del emb_model
#         emb_model = None

#     gc.collect()
#     torch.cuda.empty_cache()


def get_splitter():
    chunk_size = 500
    chunk_overlap = int(0.2 * chunk_size)

    emb_model = get_emb_model()

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=emb_model.tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter


class EmbeddingFcn(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        emb_model = get_emb_model()
        # return emb_model.encode(input).tolist()
        return emb_model.encode(input)


def get_embeddings(texts):
    return EmbeddingFcn()(texts)


def get_tokens(texts):
    emb_model = get_emb_model()
    token_ids = emb_model.tokenize(texts)["input_ids"]
    return [len(t) - 2 for t in token_ids]


def get_client():
    return chromadb.PersistentClient(path=PERSISTED_DATA_PATH)


def get_collection(name, clear=False):
    client = get_client()

    collection_names = [c.name for c in client.list_collections()]

    already_exists = name in collection_names

    if clear and already_exists:
        client.delete_collection(name)
        return client.create_collection(name=name, embedding_function=EmbeddingFcn())

    if already_exists:
        return client.get_collection(name=name, embedding_function=EmbeddingFcn())

    return client.create_collection(name=name, embedding_function=EmbeddingFcn())


def get_blog_collection(clear=False):
    return get_collection("blog", clear=clear)
