from app.document_loader import load_document

result = load_document("data/sample_sales.csv")

print("File:", result.filename)
print("Metadata:", result.metadata)
print("\nDocument preview:")
print(result.text)

print("\nDataFrame:")
print(result.tables[0])

print("\nTotal revenue:", result.tables[0]["revenue"].sum())