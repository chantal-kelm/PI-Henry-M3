import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ajustá las rutas según tus archivos reales
docs_info = [
    ("HR", "data/hr_docs/doc_hr.txt"),
    ("Tech", "data/tech_docs/doc_tech.txt"),
    ("Finance", "data/finance_docs/doc_finance.txt"),
]

# Usamos los mismos parámetros que tenés en tu proyecto
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)

print("🔍 CONTANDO CHUNKS POR DOMINIO:\n" + "="*35)

for domain, path in docs_info:
    if os.path.exists(path):
        loader = TextLoader(path, encoding="utf-8")
        raw_docs = loader.load()
        chunks = text_splitter.split_documents(raw_docs)
        count = len(chunks)
        status = "✅ CUMPLE (>= 50)" if count >= 50 else "⚠️ FALTA (Menor a 50)"
        print(f"📌 {domain} ({path}): {count} chunks -> {status}")
    else:
        print(f"❌ No se encontró el archivo: {path}")

print("="*35)