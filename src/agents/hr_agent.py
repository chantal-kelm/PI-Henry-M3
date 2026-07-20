import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

def get_hr_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    docs = []
    dir_path = "data/hr_docs"
    if os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            with open(os.path.join(dir_path, file), "r", encoding="utf-8") as f:
                docs.append(Document(page_content=f.read(), metadata={"source": file}))
                
    vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # NUEVO SYSTEM PROMPT: Más completo, formal y con instrucciones de control
    system_instruction = (
        "Eres un agente experto especializado en Recursos Humanos (HR). "
        "Responde de manera clara, profesional y detallada utilizando ÚNICAMENTE el contexto provisto.\n\n"
        "REGLAS IMPORTANTES:\n"
        "1. Explica minuciosamente lo que estipula la política general según la documentación.\n"
        "2. Si el usuario te pregunta cuántos días de vacaciones tiene de forma personal o individual, "
        "indica detalladamente lo que dice la norma del documento y aclara cordialmente que, como asistente general, "
        "no posees acceso directo a su saldo histórico particular o legajo en tiempo real, por lo que deberá verificarlo "
        "con el área de liquidaciones.\n\n"
        "Contexto:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "{question}")
    ])
    
    return (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    