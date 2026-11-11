import pandas as pd
from pathlib import Path

from app.document_loader import load_pdf
from app.pdf_data_extractor import extract_pdf_dataframe


# PDF location
pdf_path = Path(r"C:\Users\satya\Downloads\Sales_Orders_Annual_Report_1000_Orders.pdf")

# Load PDF using your existing document loader
document = load_pdf(pdf_path)

# Select the detailed table
question = (
    "What is the total revenue for orders "
    "in the North region and Electronics category?"
)

df = extract_pdf_dataframe(document, question)

# Apply both filters independently
result = df[
    (df["Region"].astype(str).str.strip().str.casefold() == "north")
    &
    (df["Category"].astype(str).str.strip().str.casefold() == "electronics")
].copy()

# Convert revenue to numeric
result["Revenue"] = pd.to_numeric(
    result["Revenue"],
    errors="coerce",
)

# Calculate verified result
total_revenue = result["Revenue"].sum()

print("\n========== VERIFICATION RESULTS ==========")
print("Total matching orders:", len(result))
print("Verified North + Electronics revenue:", f"{total_revenue:,.2f}")
print("API revenue:", "20,310,495.00")
print("==========================================")
