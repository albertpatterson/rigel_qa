from rag.rag_context import get_rag_context_content, RAGContextConfig


def get_blog_rag_prompt(question, max_est_tokens=4000):
    config = RAGContextConfig(n_main=100, n_long=20, min_len_long=100)
    context_content = get_rag_context_content(question, blog_config=config)

    preamble = (
        "I need to answer some questions about my friend Rigal based on his blog posts."
    )

    context_introduction = (
        "Here are some blog posts that might help you answer the question: "
    )

    context_parts = []
    est_tokens = 0

    for i, context in enumerate(context_content):
        tokens = context["metadatas"]["tokens"]
        next_tokens_total = est_tokens + tokens + 1
        if next_tokens_total > max_est_tokens:
            break

        context_parts.append(context["documents"].strip())
        est_tokens = next_tokens_total

    context_text = "\n\n==========\n\n".join(context_parts)

    suffix = f"""Using the information from the blog posts, answer the following question.
QUESTION: {question}
ANSWER: """

    prompt = f"""{preamble}
    
{context_introduction}

{context_text}

{suffix}"""

    return prompt, context_content[:i]


# print(get_blog_rag_prompt('Where did he meet his wife? Her name is Britta.'))
