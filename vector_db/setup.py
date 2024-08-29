from vector_db.shared import (
    get_splitter,
    get_embeddings,
    get_blog_collection,
    get_tokens,
)
from util.util import load_data


def setup_vector_db():

    blog_data = load_data("blog_data.pkl")

    kept_data = []
    post_index = 0
    for category, data in blog_data.items():
        for d in data:
            kept_data.append(
                {
                    "text": d["content"],
                    "link": d["link"],
                    "tags": d["classes"],
                    "title": d["title"],
                    "created": d["date"],
                    "category": category,
                    "post_index": post_index,
                }
            )
            post_index += 1

    split_text_data = []
    splitter = get_splitter()
    for i in range(len(kept_data)):
        text = kept_data[i]["text"]
        split_text = splitter.split_text(text)
        for st_i in range(len(split_text)):
            split_text_data.append(
                {**kept_data[i], "text": split_text[st_i], "split_index": st_i}
            )

    def fomate_date(date):
        if date is None:
            return ""
        return date.strftime("%Y-%m-%d %H:%M:%S")

    def get_metadata(split_text_datum):
        out = {
            **split_text_datum,
            "tags": ",,, ".join(split_text_datum["tags"]),
            "len": len(split_text_datum["text"]),
            "tokens": get_tokens([split_text_datum["text"]])[0],
            "created": fomate_date(split_text_datum["created"]),
        }

        del out["text"]
        return out

    documents = [d["text"] for d in split_text_data]
    metadatas = [get_metadata(d) for d in split_text_data]
    embeddings = get_embeddings(documents)
    ids = [f"blog-{i}" for i in range(len(split_text_data))]

    blog_collection = get_blog_collection(True)
    blog_collection.add(
        ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
    )

    # print(f"num items: {blog_collection.count()}")
    # print()
    # print('items similar to "Work":')
    # print(
    #     blog_collection.query(
    #         query_texts=["Work"], n_results=5, include=["documents", "metadatas"]
    #     )
    # )
