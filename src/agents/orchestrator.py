from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_orchestrator_chain():
    """
    Cadena de clasificación de intenciones basada en LLM.
    Analiza la pregunta del usuario y determina la categoría correspondiente.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un enrutador inteligente para un sistema corporativo.
Analiza la pregunta del usuario y clasifícala estrictamente en UNA de las siguientes categorías:

- hr: Consultas sobre vacaciones, licencias, beneficios, sueldos o Recursos Humanos.
- finance: Consultas sobre viáticos, gastos, reembolsos, facturas o Finanzas.
- tech: Consultas sobre VPN, credenciales, soporte técnico, hardware o software.
- out_of_scope: Cualquier consulta ajena a los temas corporativos anteriores.

Responde ÚNICAMENTE con la etiqueta en minúsculas (hr, finance, tech, o out_of_scope) sin texto adicional ni puntuación."""),
        ("human", "{question}")
    ])

    return prompt | llm | StrOutputParser()