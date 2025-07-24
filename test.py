import pandas as pd

# Load datasets
df = pd.read_csv("fashion-dataset/styles.csv", quotechar='"', encoding='utf-8', on_bad_lines='skip')
df_images = pd.read_csv("fashion-dataset/images.csv")

# Bersihkan kolom filename di df_images
df_images["filename"] = df_images["filename"].str.replace(".jpg", "", regex=False)
df_images["filename"] = pd.to_numeric(df_images["filename"], errors="coerce")
df_images = df_images.dropna(subset=["filename"])
df_images["filename"] = df_images["filename"].astype(int)

# Gabungkan berdasarkan id dan filename
df_combined = pd.merge(df, df_images, left_on="id", right_on="filename", how="left")

# Isi kosong dengan string kosong dan buat kolom text gabungan
df_combined["productDisplayName"] = df_combined["productDisplayName"].fillna("").astype(str)
df_combined["articleType"] = df_combined["articleType"].fillna("")
df_combined["baseColour"] = df_combined["baseColour"].fillna("")

# Gabungkan teks untuk TF-IDF
df_combined["text"] = (
    df_combined["productDisplayName"] + " " +
    df_combined["articleType"] + " " +
    df_combined["baseColour"]
)

# Tampilkan hasil kolom text
print(df_combined[["id", "productDisplayName", "articleType", "baseColour", "text"]].head(10))
