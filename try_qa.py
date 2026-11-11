from app.document_loader import load_document
from app.qa_pipeline import answer_question

# Load your sample CSV file.
document = load_document("data/sample_sales.csv")

# Ask a question using your RAG pipeline.
result = answer_question(
    document,
    "how many rows are there",
)

print("\nAnswer:")
print(result.answer)

print("\nSources:")
for source in result.sources:
    print(
        f"File: {source['filename']} | "
        f"Chunk: {source['chunk_index']} | "
        f"Score: {source['score']:.3f}"
    )
    print(source["text"])
    print("-" * 50)
