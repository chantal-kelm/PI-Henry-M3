from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.agents.document_loader import load_domain_documents
from src.agents.rag_chain import build_domain_rag_chain


@lru_cache(maxsize=1)
def get_hr_chain():
    """
    Carga la base de conocimiento de Recursos Humanos, aplica fragmentación (chunking),
    genera los embeddings y retorna la cadena de respuesta junto con su retriever.

    El pipeline ejecuta el retriever una sola vez y entrega a la cadena un diccionario
    con ``question`` y ``context``. La cadena y el índice se reutilizan durante
    toda la vida del proceso.
    """
    raw_docs = load_domain_documents("hr_docs")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # --- FRAGMENTACIÓN / CHUNKING EXPLÍCITA ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
    )
    docs = text_splitter.split_documents(raw_docs)

    if not docs:
        raise ValueError("La colección de HR no generó ningún chunk utilizable.")

    vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un agente experto especializado en Recursos Humanos (HR). Responde la consulta utilizando ÚNICAMENTE el contexto provisto.\n\nContexto:\n{context}"),
        ("human", "{question}")
    ])

    answer_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return answer_chain, retriever


@lru_cache(maxsize=1)
def get_hr_rag_chain():
    """Retorna el agente RAG completo de HR como un único Runnable LCEL."""
    answer_chain, retriever = get_hr_chain()
    return build_domain_rag_chain(answer_chain, retriever, domain="hr")
