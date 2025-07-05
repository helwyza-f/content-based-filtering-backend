from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load datasets
df = pd.read_csv("fashion-dataset/styles.csv", quotechar='"', encoding='utf-8', on_bad_lines='skip')
df_images = pd.read_csv("fashion-dataset/images.csv")

df_images["filename"] = df_images["filename"].str.replace(".jpg", "", regex=False)
df_images["filename"] = pd.to_numeric(df_images["filename"], errors="coerce")
df_images = df_images.dropna(subset=["filename"])
df_images["filename"] = df_images["filename"].astype(int)

df_combined = pd.merge(df, df_images, left_on="id", right_on="filename", how="left")

# Text for TF-IDF
df_combined["productDisplayName"] = df_combined["productDisplayName"].fillna("").astype(str)
df_combined["articleType"] = df_combined["articleType"].fillna("")
df_combined["baseColour"] = df_combined["baseColour"].fillna("")
df_combined["text"] = (
    df_combined["productDisplayName"] + " " +
    df_combined["articleType"] + " " +
    df_combined["baseColour"]
)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df_combined["text"])

# === SKIN TONE MAP ===
SKIN_TONE_COLOR_MAP = {
    "warm": ["red", "orange", "yellow", "brown", "olive", "beige"],
    "cool": ["blue", "green", "purple", "gray", "black"],
    "neutral": ["white", "black", "gray", "navy", "denim"]
}

# === CATEGORY RELEVANCE ===
RELEVANT_CATEGORY_MAP = {
    ("Apparel", "Topwear"): [("Apparel", "Bottomwear"), ("Footwear", None), ("Accessories", None)],
    ("Apparel", "Bottomwear"): [("Apparel", "Topwear"), ("Footwear", None), ("Accessories", None)],
    ("Apparel", "Apparel Set"): [("Footwear", None), ("Accessories", None)],
    ("Footwear", "Shoes"): [("Apparel", "Topwear"), ("Apparel", "Bottomwear"), ("Accessories", None)],
    ("Footwear", "Flip Flops"): [("Apparel", "Topwear"), ("Apparel", "Bottomwear")],
    ("Accessories", None): [("Apparel", "Topwear"), ("Apparel", "Bottomwear"), ("Footwear", None)],
}

def get_relevant_categories(master, sub):
    if (master, sub) in RELEVANT_CATEGORY_MAP:
        return RELEVANT_CATEGORY_MAP[(master, sub)]
    elif master == "Accessories":
        return RELEVANT_CATEGORY_MAP[("Accessories", None)]
    elif master == "Footwear":
        return RELEVANT_CATEGORY_MAP.get(("Footwear", "Shoes"), [])
    elif master == "Apparel":
        return [("Footwear", None), ("Accessories", None)]
    else:
        return [("Apparel", "Topwear"), ("Apparel", "Bottomwear"), ("Footwear", None)]

# Mapping ID ke index untuk cosine similarity
id_index_map = {row["id"]: idx for idx, row in df_combined.iterrows()}

# === FILTERING FUNGSIONAL ===
def filter_by_user_preferences(style=None, skin_tone=None, gender=None, season=None):
    filtered = df_combined.copy()
    if style:
        filtered = filtered[filtered["usage"].str.lower() == style.lower()]
    if gender:
        filtered = filtered[filtered["gender"].str.lower() == gender.lower()]
    if season:
        filtered = filtered[filtered["season"].str.lower() == season.lower()]
    if skin_tone:
        color_list = SKIN_TONE_COLOR_MAP.get(skin_tone.lower(), [])
        filtered = filtered[filtered["baseColour"].str.lower().isin([c.lower() for c in color_list])]
    return filtered

# === REKOMENDASI BERDASARKAN ITEM REFERENSI SAJA ===
def recommend_outfit(item_id, skin_tone=None):
    item_id = int(item_id)

    if item_id not in id_index_map:
        return {"message": "Item ID not found."}

    index = id_index_map[item_id]
    reference_item = df_combined[df_combined["id"] == item_id].iloc[0]

    ref_style = reference_item["usage"]
    ref_gender = reference_item["gender"]
    ref_season = reference_item["season"]

    filtered_df = filter_by_user_preferences(
        style=ref_style,
        gender=ref_gender,
        season=ref_season,
        skin_tone=skin_tone
    )

    ref_master = reference_item["masterCategory"]
    ref_sub = reference_item["subCategory"]
    kategori_relevan = get_relevant_categories(ref_master, ref_sub)

    filtered_df = filtered_df[
        filtered_df.apply(
            lambda row: any(
                row["masterCategory"] == km and (ks is None or row["subCategory"] == ks)
                for km, ks in kategori_relevan
            ),
            axis=1
        )
    ]

    if filtered_df.empty:
        return {
            "reference": build_reference(reference_item),
            "recommendations": [],
            "message": "No relevant items found."
        }

    tfidf_matrix_filtered = tfidf.transform(filtered_df["text"])
    similarity_scores = cosine_similarity(tfidf_matrix[index], tfidf_matrix_filtered).flatten()

    penalties = [0.9 if art == reference_item["articleType"] else 0.5 for art in filtered_df["articleType"]]
    similarity_scores *= penalties

    top_indices = similarity_scores.argsort()[::-1]

    seen_ids = set()
    seen_names = set()
    recommendations = []

    for i in top_indices:
        rec = filtered_df.iloc[i]
        rec_id = int(rec["id"])
        name = rec["productDisplayName"]

        if rec_id == item_id or rec_id in seen_ids or name in seen_names:
            continue

        seen_ids.add(rec_id)
        seen_names.add(name)

        recommendations.append({
            "id": rec_id,
            "productDisplayName": name,
            "baseColour": rec["baseColour"],
            "articleType": rec["articleType"],
            "link": rec["link"] if pd.notna(rec["link"]) else None
        })

        if len(recommendations) >= 6:
            break

    return {
        "reference": build_reference(reference_item),
        "recommendations": recommendations
    }

def build_reference(ref):
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

from math import ceil

@app.route("/catalog", methods=["GET"])
def get_catalog():
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 12))

        filters = {
            "gender": request.args.get("gender"),
            "season": request.args.get("season"),
            "usage": request.args.get("usage"),
            "masterCategory": request.args.get("masterCategory")
        }

        # filter data
        catalog_data = df_combined.copy()
        for key, value in filters.items():
            if value:
                catalog_data = catalog_data[catalog_data[key].str.lower() == value.lower()]

        catalog_data = catalog_data[["id", "productDisplayName", "baseColour", "articleType", "link"]]
        catalog_data = catalog_data.dropna(subset=["productDisplayName"])

        total_items = len(catalog_data)
        total_pages = ceil(total_items / limit) if limit > 0 else 1

        start = (page - 1) * limit
        end = start + limit
        paginated_data = catalog_data.iloc[start:end]

        return jsonify({
            "catalog": paginated_data.to_dict(orient="records"),
            "totalItems": total_items,
            "total_pages": total_pages,
            "current_page": page
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/recommend", methods=["POST"])
def recommend_outfit():
    try:
        data = request.get_json()
        item_id = int(data.get("id"))
        skin_tone = data.get("skin_tone")

        if not item_id or not skin_tone:
            return jsonify({"error": "Missing required parameters"}), 400

        # Ambil reference item
        reference = df_combined[df_combined["id"] == item_id].iloc[0]

        # Filter kandidat dengan usage, gender, season yang sama
        candidates = df_combined[
            (df_combined["usage"].str.lower() == reference["usage"].lower()) &
            (df_combined["gender"].str.lower() == reference["gender"].lower()) &
            (df_combined["season"].str.lower() == reference["season"].lower()) &
            (df_combined["id"] != item_id)
        ]

        # Filter lagi berdasarkan skin tone (misalnya dari warna dominan)
        if skin_tone.lower() == "warm":
            candidates = candidates[candidates["baseColour"].isin(["Red", "Orange", "Yellow", "Brown", "Beige"])]
        elif skin_tone.lower() == "cool":
            candidates = candidates[candidates["baseColour"].isin(["Blue", "Green", "Purple", "Grey", "Black"])]
        elif skin_tone.lower() == "neutral":
            candidates = candidates[candidates["baseColour"].isin(["White", "Grey", "Black", "Beige"])]

        recommendations = candidates[["id", "productDisplayName", "baseColour", "articleType", "link", "usage", "gender", "season"]].dropna().sample(n=6, random_state=42)

        return jsonify({
            "reference": reference[["id", "productDisplayName", "baseColour", "articleType", "link", "usage", "gender", "season"]].to_dict(),
            "recommendations": recommendations.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/catalog/<int:item_id>", methods=["GET"])
def get_item_by_id(item_id):
    try:
        item = df_combined[["id", "productDisplayName", "baseColour", "articleType", "link"]]
        item = item[item["id"] == item_id]

        if not item.empty:
            return jsonify(item.to_dict(orient="records")[0])
        else:
            return jsonify({"error": "Item not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
