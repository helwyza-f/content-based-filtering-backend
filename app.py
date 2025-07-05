from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Load datasets
df = pd.read_csv("fashion-dataset/styles.csv", quotechar='"', encoding='utf-8', on_bad_lines='skip')
df_images = pd.read_csv("fashion-dataset/images.csv")

df_images["filename"] = df_images["filename"].str.replace(".jpg", "", regex=False)
df_images["filename"] = pd.to_numeric(df_images["filename"], errors="coerce")
df_images = df_images.dropna(subset=["filename"])
df_images["filename"] = df_images["filename"].astype(int)

df_combined = pd.merge(df, df_images, left_on="id", right_on="filename", how="left")

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

SKIN_TONE_COLOR_MAP = {
    "warm": ["red", "orange", "yellow", "brown", "olive", "beige"],
    "cool": ["blue", "green", "purple", "gray", "black"],
    "neutral": ["white", "black", "gray", "navy", "denim"]
}

id_index_map = {row["id"]: idx for idx, row in df_combined.iterrows()}

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

# === FILTER FUNGSIONAL ===
def filter_by_user_preferences(style=None, skin_tone=None, gender=None):
    filtered = df_combined.copy()
    if style:
        filtered = filtered[filtered["usage"].str.lower() == style.lower()]
    if gender:
        filtered = filtered[filtered["gender"].str.lower() == gender.lower()]
    if skin_tone:
        color_list = SKIN_TONE_COLOR_MAP.get(skin_tone.lower(), [])
        filtered = filtered[filtered["baseColour"].str.lower().isin([c.lower() for c in color_list])]
    return filtered

# === REKOMENDASI ===
def recommend_outfit(item_id, style=None, skin_tone=None, gender=None):
    item_id = int(item_id)

    if item_id not in id_index_map:
        return {"message": "Item ID not found."}

    index = id_index_map[item_id]
    reference_item = df_combined[df_combined["id"] == item_id].iloc[0]
    reference_data = {
        "id": int(reference_item["id"]),
        "productDisplayName": reference_item["productDisplayName"],
        "baseColour": reference_item["baseColour"],
        "articleType": reference_item["articleType"],
        "link": reference_item["link"] if pd.notna(reference_item["link"]) else None
    }

    filtered_df = filter_by_user_preferences(style, skin_tone, gender)
    if filtered_df.empty:
        return {
            "reference": reference_data,
            "recommendations": [],
            "message": "No matching items found."
        }

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
            "reference": reference_data,
            "recommendations": [],
            "message": "No relevant category matches found."
        }

    tfidf_matrix_filtered = tfidf.transform(filtered_df["text"])
    similarity_scores = cosine_similarity(tfidf_matrix[index], tfidf_matrix_filtered).flatten()

    ref_type = reference_item["articleType"]
    penalties = [0.9 if art == ref_type else 0.5 for art in filtered_df["articleType"]]
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
        "reference": reference_data,
        "recommendations": recommendations
    }

# === ROUTES ===
@app.route("/catalog", methods=["GET"])
def get_catalog():
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 12))

        catalog_data = df_combined[["id", "productDisplayName", "baseColour", "articleType", "link"]]
        catalog_data = catalog_data.dropna(subset=["productDisplayName"])

        total_items = len(catalog_data)
        start = (page - 1) * limit
        end = start + limit
        paginated_data = catalog_data.iloc[start:end]

        return jsonify({
            "catalog": paginated_data.to_dict(orient="records"),
            "totalItems": total_items
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

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    item_id = data.get("id")
    style = data.get("style")
    skin_tone = data.get("skin_tone")
    gender = data.get("gender")

    try:
        result = recommend_outfit(item_id, style, skin_tone, gender)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
