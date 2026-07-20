from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def get_orchestrator_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres el Agente Orquestador. Clasifica la consulta en: 'hr', 'tech', o 'finance'.\n"
                   "Responde estrictamente en JSON: {\"destination\": \"hr\" | \"tech\" | \"finance\"}"),
        ("human", "{question}")
    ])
    return prompt | llm | JsonOutputParser()