from src.lost_in_the_middle.gpt import get_openai_chat_completion


def interleaved_prompt(question, documents):
    system_message = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). The question will be repeated before each listed search result for context."

    formatted_documents = []
    for document_index, document in enumerate(documents):
        formatted_documents.append(f"Document [{document_index+1}](Title: {document.title}) {document.text}")

    if not question.endswith("?"):
        question += "?"
    q = f"\nQuestion: {question}\n"
    search_results = q.join(formatted_documents)
    user_message = f"{q}{search_results}{q}Answer:"

    prompt = {
        "system_message": system_message,
        "user_message": user_message,
    }

    return prompt


def know_your_weakness(question, documents):
    system_message = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). Remember that your performance on retrieval tasks is worse on text in the middle of your context window, so please attend carefully to the documents in the middle when looking for the answer."

    formatted_documents = []
    for document_index, document in enumerate(documents):
        formatted_documents.append(f"Document [{document_index+1}](Title: {document.title}) {document.text}")

    if not question.endswith("?"):
        question += "?"
    space = f"\n\n"
    search_results = space.join(formatted_documents)
    user_message = f"{search_results}\n\n{question}\nAnswer:"

    prompt = {
        "system_message": system_message,
        "user_message": user_message,
    }

    return prompt


def summarize_first(question, documents):
    system_message = "Please succinctly summarize the content of this list of documents. Be sure to also include information related to the given question in your summary."

    formatted_documents = []
    for document_index, document in enumerate(documents):
        formatted_documents.append(f"Document [{document_index+1}](Title: {document.title}) {document.text}")
    if not question.endswith("?"):
        question += "?"
    space = f"\n\n"
    search_results = space.join(formatted_documents)
    user_message = f"{question}\n\n{search_results}"
    prompt = {
        "system_message": new_system_message,
        "user_message": new_user_message,
    }

    return prompt
