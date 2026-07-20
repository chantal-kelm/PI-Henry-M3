import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

def get_finance_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    docs = []
    dir_path = "data/finance_docs"
    if os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            with open(os.path.join(dir_path, file), "r", encoding="utf-8") as f:
                docs.append(Document(page_content=f.read(), metadata={"source": file}))
                
    vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # NUEVO SYSTEM PROMPT: Estructurado para respuestas completas y manejo de datos ausentes
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
    
    return (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )