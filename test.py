import pandas as pd

# Ganti path sesuai lokasi dataset kamu
df = pd.read_csv("C:/Users/helwyza/Desktop/smart-waste/python/fashion-dataset/styles.csv", encoding="utf-8", on_bad_lines="skip")


# Kolom yang ingin dicek
columns_to_check = [
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "year",
    "usage"
]

# Ambil nilai unik
for col in columns_to_check:
    print(f"\nUnique values for '{col}':")
    print(sorted(df[col].dropna().unique().tolist()))
