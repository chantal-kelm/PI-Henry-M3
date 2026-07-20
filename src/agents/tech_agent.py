import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

def get_tech_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    docs = []
    dir_path = "data/tech_docs"
    if os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            with open(os.path.join(dir_path, file), "r", encoding="utf-8") as f:
                docs.append(Document(page_content=f.read(), metadata={"source": file}))
                
    vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un agente experto especializado en Soporte Tecnico (IT Support). Responde utilizando ÚNICAMENTE el contexto provisto.\n\nContexto:\n{context}"),
        ("human", "{question}")
    ])
    
    return (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )