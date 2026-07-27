from pathlib import Path

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
SUPPORTED_TEXT_EXTENSIONS = {".csv", ".md", ".txt"}


def load_domain_documents(
    collection_name: str,
    data_root: Path = DATA_ROOT,
) -> list[Document]:
    """Carga una colección textual y falla si no puede formar una base válida."""
    collection_path = data_root / collection_name

    if not collection_path.exists():
        raise FileNotFoundError(
            f"No existe la colección documental: {collection_path}"
        )

    if not collection_path.is_dir():
        raise NotADirectoryError(
            f"La colección documental no es un directorio: {collection_path}"
        )

    document_paths = sorted(
        path
        for path in collection_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS
    )

    if not document_paths:
        supported = ", ".join(sorted(SUPPORTED_TEXT_EXTENSIONS))
        raise ValueError(
            f"La colección {collection_path} no contiene documentos compatibles "
            f"({supported})."
        )

    documents = []

    for document_path in document_paths:
        content = document_path.read_text(encoding="utf-8")
        if not content.strip():
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": document_path.name,
                    "source_path": str(document_path.relative_to(data_root)),
                },
            )
        )

    if not documents:
        raise ValueError(
            f"La colección {collection_path} solo contiene documentos vacíos."
        )

    return documents
