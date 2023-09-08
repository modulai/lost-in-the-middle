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
