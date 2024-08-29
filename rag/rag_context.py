from vector_db.shared import get_blog_collection as get_blog_collection_raw
from dataclasses import dataclass, asdict
from typing import Optional


blog_collection = None


def get_blog_collection():
    global blog_collection

    if blog_collection is None:
        blog_collection = get_blog_collection_raw()

    return blog_collection


def map_results(results, fields=["ids", "documents"]):
    out = []
    for field in fields:
        data = results[field]
        for i in range(len(data)):
            if len(out) <= i:
                out.append([])

            row = out[i]
            for j in range(len(data[i])):
                if len(out[i]) <= j:
                    row.append({})

                el = row[j]
                el[field] = data[i][j]

    return out


@dataclass
class RAGContextConfig:
    n_main: Optional[int] = None
    n_long: Optional[int] = None
    min_len_long: Optional[int] = None

    def merge(self, other):
        return RAGContextConfig(**{**asdict(self), **asdict(other)})


def query(
    collection,
    query_text,
    n_results,
    include=["documents", "distances", "metadatas"],
    **kwargs
):
    return map_results(
        collection.query(
            query_texts=[query_text], n_results=n_results, include=include, **kwargs
        ),
        ["ids"] + include,
    )[0]


def query_blog(
    query_text, n_results, include=["documents", "distances", "metadatas"], **kwargs
):
    collection = get_blog_collection()
    return query(collection, query_text, n_results, include, **kwargs)


def query_with_some_long(collection, query_text, config: RAGContextConfig):
    main = map_results(
        collection.query(
            query_texts=[query_text],
            n_results=config.n_main,
            include=["documents", "distances", "metadatas"],
        ),
        ["documents", "distances", "ids", "metadatas"],
    )[0]
    longs = map_results(
        collection.query(
            query_texts=[query_text],
            n_results=config.n_long,
            include=["documents", "distances", "metadatas"],
            where={"len": {"$gt": config.min_len_long}},
        ),
        ["documents", "distances", "ids", "metadatas"],
    )[0]

    main_ids = [el["ids"] for el in main]
    long_ids = [el["ids"] for el in longs]
    main_ids_set = set(main_ids)

    for i in range(len(long_ids)):
        long_el = longs[i]
        if long_el["ids"] not in main_ids_set:
            main.append(long_el)
            main_ids_set.add(long_el["ids"])

    return main


DEFAULT_BLOG_CONFIG = RAGContextConfig(n_main=100, n_long=20, min_len_long=100)


def get_rag_context_content(
    question,
    blog_config: Optional[RAGContextConfig] = DEFAULT_BLOG_CONFIG,
):

    blog_config = (
        DEFAULT_BLOG_CONFIG
        if blog_config is None
        else DEFAULT_BLOG_CONFIG.merge(blog_config)
    )

    all_results = []
    blog_collection = get_blog_collection()
    blog_results = query_with_some_long(blog_collection, question, blog_config)
    all_results += blog_results

    all_results.sort(key=lambda x: x["distances"])

    return all_results
