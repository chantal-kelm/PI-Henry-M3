import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_hr_chain():
    """
    Carga la base de conocimiento de Recursos Humanos, aplica fragmentación (chunking),
    genera los embeddings y retorna la cadena RAG junto con su retriever.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    raw_docs = []
    dir_path = "data/hr_docs"
    if os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_docs.append(Document(page_content=f.read(), metadata={"source": file}))

    # --- FRAGMENTACIÓN / CHUNKING EXPLÍCITA ---
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
    )
    docs = text_splitter.split_documents(raw_docs)
                
    vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un agente experto especializado en Recursos Humanos (HR). Responde la consulta utilizando ÚNICAMENTE el contexto provisto.\n\nContexto:\n{context}"),
        ("human", "{question}")
    ])
    
    format_docs = lambda documents: "\n\n".join(d.page_content for d in documents)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Retornamos la tupla (cadena, retriever)
    return rag_chain, retriever