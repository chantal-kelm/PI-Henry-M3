from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.agents.document_loader import load_domain_documents
from src.agents.rag_chain import build_domain_rag_chain


@lru_cache(maxsize=1)
def get_finance_chain():
    """
    Carga la base de conocimiento de Finanzas, aplica fragmentación (chunking),
    genera los embeddings y retorna la cadena de respuesta junto con su retriever.

    El pipeline ejecuta el retriever una sola vez y entrega a la cadena un diccionario
    con ``question`` y ``context``. La cadena y el índice se reutilizan durante
    toda la vida del proceso.
    """
    raw_docs = load_domain_documents("finance_docs")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # --- FRAGMENTACIÓN / CHUNKING EXPLÍCITA ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
    )
    docs = text_splitter.split_documents(raw_docs)

    if not docs:
        raise ValueError("La colección de Finance no generó ningún chunk utilizable.")

    vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    system_instruction = (
        "Eres un agente experto especializado en Finanzas y Contabilidad corporativa. "
        "Responde de manera clara, formal y detallada utilizando ÚNICAMENTE el contexto provisto.\n\n"
        "REGLAS IMPORTANTES:\n"
        "1. Detalla minuciosamente las políticas generales de rendición, plazos, topes de gastos y normativas de viáticos según los documentos.\n"
        "2. Si el usuario te consulta sobre fechas límite específicas, estado de sus reembolsos particulares o datos personales que no figuran explícitamente en el contexto, "
        "resume las reglas generales del documento provisto y aclara amablemente que, por cuestiones de privacidad y actualización en tiempo real, "
        "no posees acceso a su cuenta o transacciones individuales, derivándolo al área de administración o finanzas para validar su caso particular.\n\n"
        "Contexto:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "{question}")
    ])

    answer_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return answer_chain, retriever


@lru_cache(maxsize=1)
def get_finance_rag_chain():
    """Retorna el agente RAG completo de Finance como un único Runnable LCEL."""
    answer_chain, retriever = get_finance_chain()
    return build_domain_rag_chain(answer_chain, retriever, domain="finance")
