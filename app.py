from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from math import ceil

app = Flask(__name__)
CORS(app)

# === Load dan Persiapan Data ===
df = pd.read_csv("fashion-dataset/styles.csv", quotechar='"', encoding='utf-8', on_bad_lines='skip')
df_images = pd.read_csv("fashion-dataset/images.csv")

df_images["filename"] = pd.to_numeric(df_images["filename"].str.replace(".jpg", "", regex=False), errors="coerce")
df_images.dropna(subset=["filename"], inplace=True)
df_images["filename"] = df_images["filename"].astype(int)

df_combined = pd.merge(df, df_images, left_on="id", right_on="filename", how="left")

# Isi nilai kosong & buat kolom teks gabungan
for col in ["productDisplayName", "articleType", "baseColour", "usage", "gender"]:
    df_combined[col] = df_combined[col].fillna("").astype(str)

df_combined["text"] = (
    df_combined["productDisplayName"] + " " +
    df_combined["articleType"] + " " +
    df_combined["baseColour"] + " " +
    df_combined["usage"] + " " +
    df_combined["gender"]
)

# === TF-IDF Preparation ===
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df_combined["text"])
id_index_map = {row["id"]: idx for idx, row in df_combined.iterrows()}

# === Skin Tone Color Mapping berdasarkan baseColour ===
SKIN_TONE_COLOR_MAP = {
    "warm": [
        "Red", "Orange", "Yellow", "Brown", "Coffee Brown", "Beige", "Gold",
        "Copper", "Mustard", "Tan", "Peach", "Skin", "Rust", "Mushroom Brown"
    ],
    "cool": [
        "Blue", "Navy Blue", "Green", "Teal", "Turquoise Blue", "Purple", "Lavender",
        "Magenta", "Maroon", "Black", "Silver", "Steel", "Mauve"
    ],
    "neutral": [
        "White", "Off White", "Grey", "Charcoal", "Cream", "Olive", "Burgundy",
        "Rose", "Khaki", "Grey Melange", "Bronze", "Taupe", "Nude"
    ],
    # Opsional: warna tidak dikategorikan
    "unknown": [
        "Multi", "Fluorescent Green", "Metallic", ""
    ]
}



RELEVANT_CATEGORY_MAP = {
    ("Apparel", "Topwear"): [("Apparel", "Bottomwear"), ("Footwear", None), ("Accessories", None)],
    ("Apparel", "Bottomwear"): [("Apparel", "Topwear"), ("Footwear", None), ("Accessories", None)],
    ("Apparel", "Apparel Set"): [("Footwear", None), ("Accessories", None)],
    ("Footwear", "Shoes"): [("Apparel", "Topwear"), ("Apparel", "Bottomwear"), ("Accessories", None)],
    ("Footwear", "Flip Flops"): [("Apparel", "Topwear"), ("Apparel", "Bottomwear")],
    ("Accessories", None): [("Apparel", "Topwear"), ("Apparel", "Bottomwear"), ("Footwear", None)],
}

def get_relevant_categories(master, sub):
    """
    Mendapatkan kategori relevan berdasarkan masterCategory dan subCategory.
    
    Jika tidak ada mapping yang sesuai, maka akan mengembalikan kategori default.
    
    Parameters
    ----------
    master : str
        Master category
    sub : str
        Sub category
    
    Returns
    -------
    List of tuples
        [(master_category1, sub_category1), (master_category2, sub_category2), ...]
    """
    return RELEVANT_CATEGORY_MAP.get((master, sub)) or \
           RELEVANT_CATEGORY_MAP.get((master, None)) or \
           [("Apparel", "Topwear"), ("Apparel", "Bottomwear"), ("Footwear", None)]

def build_reference(ref):
    """
    Membuat object referensi dari baris DataFrame.

    Parameters
    ----------
    ref : pd.Series
        Baris DataFrame yang berisi informasi produk

    Returns
    -------
    dict
        Objek dengan kunci id, productDisplayName, baseColour, articleType, link, usage, gender, season
    """
    return {
        "id": int(ref["id"]),
        "productDisplayName": ref["productDisplayName"],
        "baseColour": ref["baseColour"],
        "articleType": ref["articleType"],
        "link": ref["link"] if pd.notna(ref["link"]) else None,
        "usage": ref["usage"],
        "gender": ref["gender"],
        "season": ref["season"]
    }

def filter_by_user_preferences(df, style=None, skin_tone=None, gender=None, season=None):
    """
    Filter DataFrame berdasarkan preferensi pengguna.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame yang berisi informasi produk
    style : str, optional
        Gaya produk yang diinginkan. Jika None, maka tidak akan difilter.
    skin_tone : str, optional
        Warna kulit yang diinginkan. Jika None, maka tidak akan difilter.
    gender : str, optional
        Jenis kelamin yang diinginkan. Jika None, maka tidak akan difilter.
    season : str, optional
        Musim yang diinginkan. Jika None, maka tidak akan difilter.

    Returns
    -------
    pd.DataFrame
        DataFrame yang sudah difilter berdasarkan preferensi pengguna
    """
    result = df.copy()
    if style:
        result = result[result["usage"].str.lower() == style.lower()]
    if gender:
        result = result[result["gender"].str.lower() == gender.lower()]
    if season:
        result = result[result["season"].str.lower() == season.lower()]
    if skin_tone:
        colors = SKIN_TONE_COLOR_MAP.get(skin_tone.lower(), [])
        result = result[result["baseColour"].str.lower().isin([c.lower() for c in colors])]
    return result

def generate_recommendations(item_id, skin_tone=None):
    item_id = int(item_id)
    if item_id not in id_index_map:
        return {"message": "Item ID not found."}

    ref_item = df_combined[df_combined["id"] == item_id].iloc[0]
    index = id_index_map[item_id]

    filtered_df = filter_by_user_preferences(
        df_combined,
        style=ref_item["usage"],
        gender=ref_item["gender"],
        # season=ref_item["season"],
        skin_tone=skin_tone
    )

    # Filter kategori relevan
    relevant_cats = get_relevant_categories(ref_item["masterCategory"], ref_item["subCategory"])
    filtered_df = filtered_df[
        filtered_df.apply(
            lambda row: any(row["masterCategory"] == mc and (sc is None or row["subCategory"] == sc)
                            for mc, sc in relevant_cats),
            axis=1
        )
    ]

    if filtered_df.empty:
        return {
            "reference": build_reference(ref_item),
            "recommendations": [],
            "message": "No relevant items found."
        }

    tfidf_filtered = tfidf.transform(filtered_df["text"])
    similarity_scores = cosine_similarity(tfidf_matrix[index], tfidf_filtered).flatten()

    penalties = [0.9 if art == ref_item["articleType"] else 0.5 for art in filtered_df["articleType"]]
    similarity_scores *= penalties

    top_indices = similarity_scores.argsort()[::-1]

    seen_ids, seen_names = set(), set()
    recommendations = []

    for i in top_indices:
        rec = filtered_df.iloc[i]
        if rec["id"] == item_id or rec["id"] in seen_ids or rec["productDisplayName"] in seen_names:
            continue
        recommendations.append({
            "id": int(rec["id"]),
            "productDisplayName": rec["productDisplayName"],
            "baseColour": rec["baseColour"],
            "articleType": rec["articleType"],
            "link": rec["link"] if pd.notna(rec["link"]) else None,
            "gender": rec["gender"],
            "season": rec["season"],
            "usage": rec["usage"]
        })

        seen_ids.add(rec["id"])
        seen_names.add(rec["productDisplayName"])
        if len(recommendations) >= 6:
            break

    return {
        "reference": build_reference(ref_item),
        "recommendations": recommendations
    }

# === API Routes ===
@app.route("/recommend", methods=["POST"])
def recommend_route():
    try:
        data = request.get_json()
        item_id = data.get("id")
        skin_tone = data.get("skin_tone")

        if not item_id or not skin_tone:
            return jsonify({"error": "Missing id or skin_tone"}), 400

        result = generate_recommendations(item_id, skin_tone)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/catalog", methods=["GET"])
def catalog_route():
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 12))
        skin_tone = request.args.get("skin_tone")  # ✅ Tambahkan ini

        filters = {
            "gender": request.args.get("gender"),
            "season": request.args.get("season"),
            "usage": request.args.get("usage"),
            "masterCategory": request.args.get("masterCategory")
        }

        catalog = df_combined.copy()

        # Filter hanya Apparel: Topwear & Bottomwear
        catalog = catalog[
            (catalog["masterCategory"] == "Apparel") &
            (catalog["subCategory"].isin(["Topwear", "Bottomwear"]))
        ]

        for key, val in filters.items():
            if val and key in catalog.columns:
                catalog = catalog[catalog[key].str.lower() == val.lower()]

        # ✅ Tambahkan filter skin tone (warna) di sini
        # if skin_tone and skin_tone.lower() in SKIN_TONE_COLOR_MAP:
        #     valid_colours = [c.lower() for c in SKIN_TONE_COLOR_MAP[skin_tone.lower()]]
        #     catalog = catalog[catalog["baseColour"].str.lower().isin(valid_colours)]

        catalog = catalog[["id", "productDisplayName", "baseColour", "articleType", "link"]].dropna()

        total_items = len(catalog)
        total_pages = ceil(total_items / limit)
        start = (page - 1) * limit
        end = start + limit
        paginated = catalog.iloc[start:end]

        return jsonify({
            "catalog": paginated.to_dict(orient="records"),
            "totalItems": total_items,
            "total_pages": total_pages,
            "current_page": page
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/catalog/<int:item_id>", methods=["GET"])
def item_by_id_route(item_id):
    try:
        item = df_combined[df_combined["id"] == item_id]
        if not item.empty:
            result = item[["id", "productDisplayName", "baseColour", "articleType", "link"]].to_dict(orient="records")[0]
            return jsonify(result)
        return jsonify({"error": "Item not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Run App ===
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
