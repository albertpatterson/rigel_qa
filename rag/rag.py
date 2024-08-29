from rag.rag_prompt import get_blog_rag_prompt
from vector_db.shared import get_blog_collection
from rag.rag_context import query_blog
import time
from llm.llm import LLM_Manager

# print('start')
# nvidia_smi()

# # client = get_client_llama_cpp_q8()
# # client = get_client_llama_cpp_q5_k_l()
# nvidia_smi()

# out = client.generate("Where can I buy a birthday cake?")
# print(out)

# out = client.generate("Which of the Beatles was the oldest?")
# print(out)

# out = client.generate("What is the capital of the richest country?")
# print(out)

# prompt = get_blog_rag_prompt('Where did he meet his wife? Her name is Britta.')
# prompt = get_blog_rag_prompt('What does he do for work?')
# prompt = get_blog_rag_prompt('What company does he work for?')
# prompt = get_blog_rag_prompt('Where does he live?')
# prompt = get_blog_rag_prompt('What does he say about Cebu?')

# prompt = get_blog_rag_prompt('Who is his wife?')
# prompt = get_blog_rag_prompt('Who is his Britta?')

# prompt = get_blog_rag_prompt('What does he say about the army?')


# def answer(questions, max_est_tokens=4000):
#     t0 = time.time()
#     prompts = [get_blog_rag_prompt(q, max_est_tokens)[0] for q in questions]
#     clear_emb_model()
#     client = get_client_llama_cpp_q8()
#     outs = [client.generate(prompt) for prompt in prompts]
#     client.cleanup()
#     dt = time.time() - t0

#     return outs, dt


def get_best_sources(question_context_els, answer_context_els):
    # print(answer_context_els)

    question_map = {el["ids"]: el for el in question_context_els}

    found_context = []
    for answer_context_el in answer_context_els:
        answer_context_el_id = answer_context_el["ids"]
        if answer_context_el_id in question_map:
            # print(answer_context_el_id)
            answer_context_el_distance = answer_context_el["distances"]
            question_context_el_distance = question_map[answer_context_el_id][
                "distances"
            ]
            mean_distance = (
                answer_context_el_distance + question_context_el_distance
            ) / 2
            found_context.append(
                {
                    **answer_context_el,
                    "distances": mean_distance,
                }
            )

    found_context.sort(key=lambda x: x["distances"])
    sources_set = set()
    sources = []
    for el in found_context:
        post_index = el["metadatas"]["post_index"]
        link = el["metadatas"]["link"]
        if link in sources_set:
            continue
        if post_index in sources_set:
            continue
        if link is None or link == "":
            sources.append({"source": post_index, "distance": el["distances"]})
            sources_set.add(post_index)
        else:
            sources.append({"source": link, "distance": el["distances"]})
            sources_set.add(link)

    return sources

    # post_index_set = set()
    # post_indices = []
    # for el in found_context:
    #     post_index = el['metadatas']['post_index']
    #     distance = el['distances']
    #     if post_index not in post_index_set:
    #         post_index_set.add(post_index)
    #         post_indices.append({'post_index': post_index, 'distance': distance})
    # return post_indices


def answer_with_sources(question, max_est_tokens=4000, model_name="text_generation"):
    prompt, prompt_sources = get_blog_rag_prompt(question, max_est_tokens)

    # print("prompt")
    # print(prompt)

    client = LLM_Manager.get_model(model_name)
    out = client.generate(prompt)
    out_sources = query_blog(out, n_results=100)
    best_sources = get_best_sources(prompt_sources, out_sources)

    return out, best_sources


# prompt = get_blog_rag_prompt('What did he do while in the army?', 4000)
# clear_emb_model()

# nvidia_smi()

# # print(prompt)
# client = get_client_llama_cpp_q8()
# out = client.generate(prompt)
# print(out)


# qs=[
#     'What company does he work for?',
#     'What does he say about Cebu?',
#     'Who is his Britta?',
#     'What did he do while in the army?',
#     'What does he do in his free time?',
#     'What does he say about retirement?',
#     'What does he say about investing?',
#     'What does he say about God and church?',
#     "What are the names of his children?",
#     "When did he come to the US?",
#     "what does he say about Britta?",
#     "what does he say about Brent?",
#     "what does he say about Brittney?",
#     "what does he say about his father?",
#     "what does he say about his mother?",
# ]
# qs = [
#     # "Why is he concerned about Brent's health?",
#     # "What sort of health event did Rigal have?"
#     "When did Rigal have a heart attack?",
#     "Did Rigal have more than one heart attack?",
# ]
# ans, dt = answer(qs, 4000)

# print(f'answered in {dt}:')
# for i in range(len(qs)):
#     # print(f'{qs[i]}: {ans[i]}')
#     print(qs[i])
#     print(ans[i])
#     print()
#     print()


# out, sources = answer_with_sources('What did he do while in the army?', 4000)
# out, sources = answer_with_sources('What did he say about Cebu?', 4000)

# out, sources = answer_with_sources('What did he say about Psychology?', 4000)
# out, sources = answer_with_sources('Has he always been religious?', 4000)
# out, sources = answer_with_sources('What is the religious turning point he talks about?', 4000)
# out, sources = answer_with_sources('What did he say about Aikido?', 4000)
# out, sources = answer_with_sources('What did he say about moving to the US?', 4000)
# out, sources = answer_with_sources('What did he say about moving away from the Philippines?', 4000)
# out, sources = answer_with_sources('What did he say about the town where he grew up?', 4000)
# out, sources = answer_with_sources('What did he say about leaving the army?', 4000)
# out, sources = answer_with_sources('What kind of work did he do at Samsung?', 4000)
# out, sources = answer_with_sources('Why did he move back to the Philippines?', 4000)
# out, sources = answer_with_sources('Why does he want to move back to the Philippines?', 4000)
# out, sources = answer_with_sources('What did he say about ai?', 4000)
# out, sources = answer_with_sources('What did he say about Austin?', 4000)
# out, sources = answer_with_sources('What did he say about Round Rock?', 4000)
# out, sources = answer_with_sources('Where did he grow up in the Philippines?', 4000)
# out, sources = answer_with_sources('What is Rigal\'s full name?', 4000)
# out, sources = answer_with_sources("What is Rigal's middle name?", 4000)


# print(out)
# print()
# print()
# print(sources)
# print()
# print()


# best_post_index = sources[0]['post_index']

# print('get best post')
# blog_collection = get_blog_collection()

# blog_parts = blog_collection.get(where={"post_index": {"$eq": best_post_index}})
# ids = blog_parts['ids']
# mapped = []
# for i in range(len(ids)):
#     document = blog_parts['documents'][i]
#     split_index = blog_parts['metadatas'][i]['split_index']
#     mapped.append({
#         'document': document,
#         'split_index': split_index,
#     })
# mapped.sort(key=lambda x: x['split_index'])

# print('\n\n'.join([m['document'] for m in mapped]))

# blog_parts = blog_collection.get(where={"post_index": {"$eq": best_post_index}})
# print(blog_parts)
