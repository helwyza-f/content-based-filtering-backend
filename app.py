from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from supabase import create_client
from collections import OrderedDict

# ====== CONFIG ======
SUPABASE_URL = "https://jkcfkwigybhsxjitbopz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImprY2Zrd2lneWJoc3hqaXRib3B6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MTcwNjUzMywiZXhwIjoyMDY3MjgyNTMzfQ._07D8Gi2jIUlSbGd72gYpsr62XU8JebwAF0lAWLdOac"

app = Flask(__name__)
CORS(app)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ====== DataFrame Cache ======
df_combined = None
id_index_map = {}
tfidf = None
tfidf_matrix = None


def fetch_products_from_supabase():
    global df_combined, tfidf, tfidf_matrix, id_index_map
    print("Fetching data from Supabase...")

    batch_size = 1000
    offset = 0
    all_data = []

    while True:
        response = supabase.table("products") \
            .select("*") \
            .range(offset, offset + batch_size - 1) \
            .execute()

        data = response.data
        if not data:
            break

        all_data.extend(data)
        offset += batch_size
        print(f"Loaded {len(all_data)} records so far...")

    df_combined = pd.DataFrame(all_data)
    if df_combined.empty:
        print("⚠️ No products found in Supabase.")
        return

    # Pastikan kolom text aman
    for col in ["product_display_name", "article_type", "base_colour", "usage", "gender", "sub_category"]:
        if col in df_combined.columns:
            df_combined[col] = df_combined[col].fillna("").astype(str)

    # Kolom gabungan untuk TF-IDF
    df_combined["text"] = (
        df_combined["product_display_name"] + " " +
        df_combined["article_type"] + " " +
        df_combined["base_colour"] + " " +
        df_combined["usage"] + " " +
        df_combined["gender"]
    )

    # Build TF-IDF Matrix
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df_combined["text"])
    id_index_map = {row["id"]: idx for idx, row in df_combined.iterrows()}

    print(f"✅ Cache updated: {len(df_combined)} products loaded.")


# Panggil pertama kali saat startup
fetch_products_from_supabase()

# ====== Skin Tone Color Map ======
SKIN_TONE_COLOR_MAP = {
    "warm": ["Red", "Orange", "Yellow", "Brown", "Coffee Brown", "Beige", "Gold",
             "Copper", "Mustard", "Tan", "Peach", "Skin", "Rust", "Mushroom Brown"],
    "cool": ["Blue", "Navy Blue", "Green", "Teal", "Turquoise Blue", "Purple",
             "Lavender", "Magenta", "Maroon", "Black", "Silver", "Steel", "Mauve"],
    "neutral": ["White", "Off White", "Grey", "Charcoal", "Cream", "Olive", "Burgundy",
                "Rose", "Khaki", "Grey Melange", "Bronze", "Taupe", "Nude"],
    "unknown": ["Multi", "Fluorescent Green", "Metallic", ""]
}

# ====== Article Type → Kategori Map ======
ARTICLE_TYPE_MAP = {
    # Topwear
    "Shirts": "Topwear", "Tshirts": "Topwear", "Tops": "Topwear",
    "Sweatshirts": "Topwear", "Sweaters": "Topwear", "Blazers": "Topwear",
    "Jackets": "Topwear", "Kurta Sets": "Topwear", "Kurtas": "Topwear",
    "Tunics": "Topwear", "Shrug": "Topwear", "Dresses": "Topwear",

    # Bottomwear
    "Jeans": "Bottomwear", "Track Pants": "Bottomwear", "Trousers": "Bottomwear",
    "Shorts": "Bottomwear", "Skirts": "Bottomwear", "Boxers": "Bottomwear",
     "Capris": "Bottomwear", "Lounge Pants": "Bottomwear",
    "Night suits": "Bottomwear", 

    # Footwear
    "Casual Shoes": "Footwear", "Formal Shoes": "Footwear", "Sports Shoes": "Footwear",
    "Sandals": "Footwear", "Flip Flops": "Footwear", "Heels": "Footwear",
    "Flats": "Footwear", "Sports Sandals": "Footwear",

    # Accessories
    "Watches": "Accessories", "Belts": "Accessories", "Handbags": "Accessories",
    "Wallets": "Accessories", "Clutches": "Accessories", "Backpacks": "Accessories",
    "Duffel Bag": "Accessories", "Laptop Bag": "Accessories", "Caps": "Accessories",
    "Sunglasses": "Accessories", "Scarves": "Accessories", "Mufflers": "Accessories",
    "Bracelet": "Accessories", "Earrings": "Accessories", "Ring": "Accessories",
   
}

# ====== API ROUTES ======
@app.route("/unique-categories", methods=["GET"])
def unique_categories():
    global df_combined
    return jsonify({
        "masterCategory": df_combined['master_category'].dropna().unique().tolist(),
        "subCategory": df_combined['sub_category'].dropna().unique().tolist(),
        "baseColour": df_combined['base_colour'].dropna().unique().tolist(),
        "gender": df_combined['gender'].dropna().unique().tolist(),
        "usage": df_combined['usage'].dropna().unique().tolist(),
        "articleType": df_combined['article_type'].dropna().unique().tolist()
    })


@app.route("/recommend", methods=["POST"])
def recommend_by_user():
    try:
        data = request.get_json()
        gender = data.get("gender")
        skin_tone = data.get("skin_tone")

        if not gender or not skin_tone:
            return jsonify({"error": "Missing gender or skin_tone"}), 400

        category_order = ["Topwear", "Bottomwear", "Footwear", "Accessories"]
        ordered_styles = ["formal", "casual", "sports"]

        recommendations = OrderedDict()

        for style in ordered_styles:
            filtered_df = df_combined.copy()

            # Filter gender
            filtered_df = filtered_df[filtered_df["gender"].str.lower() == gender.lower()]

            # Filter usage
            filtered_df = filtered_df[filtered_df["usage"].str.lower() == style.lower()]

            # Filter warna sesuai skin tone
            valid_colours = [c.lower() for c in SKIN_TONE_COLOR_MAP.get(skin_tone.lower(), [])]
            filtered_df = filtered_df[filtered_df["base_colour"].str.lower().isin(valid_colours)]

            # Mapping article_type ke kategori utama
            filtered_df["mapped_category"] = filtered_df["article_type"].map(ARTICLE_TYPE_MAP).fillna(filtered_df["sub_category"])

            # Hapus duplikat (berdasarkan nama + type)
            filtered_df = filtered_df.drop_duplicates(subset=["product_display_name", "article_type"])

            # Pilih kolom penting
            filtered_df = filtered_df[
                ["id", "product_display_name", "base_colour", "article_type", "mapped_category", "image_link"]
            ]

            category_dict = OrderedDict()
            for cat in category_order:
                subset = filtered_df[filtered_df["mapped_category"] == cat].head(8)
                category_dict[cat] = subset.to_dict(orient="records")

            recommendations[style] = category_dict

        return jsonify({"recommendations": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
