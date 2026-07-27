from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough


def _serialize_retrieved_documents(agent_input: dict) -> dict:
    """Convierte los documentos recuperados en contexto y metadatos evaluables."""
    context_parts = []
    doc_sources = []

    for document in agent_input["documents"]:
        if hasattr(document, "metadata") and isinstance(document.metadata, dict):
            source = document.metadata.get("source")
            if source:
                doc_sources.append(source)

        if hasattr(document, "page_content"):
            context_parts.append(str(document.page_content))
        elif isinstance(document, dict):
            context_parts.append(str(document.get("page_content", document)))
        else:
            context_parts.append(str(document))

    return {
        "question": agent_input["question"],
        "destination": agent_input["destination"],
        "context": "\n\n".join(context_parts),
        "doc_sources": doc_sources,
    }


def build_domain_rag_chain(
    answer_chain: Runnable,
    retriever: Runnable,
    domain: str,
) -> Runnable:
    """
    Construye un agente RAG LCEL completo.

    La cadena recibe la pregunta y el destino, ejecuta una única recuperación,
    serializa el contexto y genera la respuesta dentro del mismo Runnable.
    """
    question_from_input = RunnableLambda(
        lambda agent_input: agent_input["question"]
    ).with_config(run_name=f"extract_{domain}_question")

    retrieval = (question_from_input | retriever).with_config(
        run_name=f"retrieve_{domain}_context"
    )

    return (
        RunnablePassthrough.assign(documents=retrieval)
        | RunnableLambda(_serialize_retrieved_documents).with_config(
            run_name=f"prepare_{domain}_context"
        )
        | RunnablePassthrough.assign(
            response=answer_chain.with_config(
                run_name=f"generate_{domain}_response"
            )
        )
    ).with_config(run_name=f"{domain}_rag_agent")
