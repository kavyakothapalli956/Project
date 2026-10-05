import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Recipe Recommendation System",
    page_icon="🍳",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# Load recipe data
recipes = pd.read_csv("recipes.csv")

# Title
st.markdown(
    '<div class="main-title">🍳 Recipe Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Find delicious recipes based on your preferences</div>',
    unsafe_allow_html=True
)

st.divider()

# User preferences
st.subheader("🔍 Choose Your Preferences")

# Cuisine and Diet
col1, col2 = st.columns(2)

with col1:
    cuisine = st.selectbox(
        "🌍 Cuisine",
        ["All", "Indian", "Italian", "Chinese"]
    )

with col2:
    diet = st.selectbox(
        "🥗 Diet",
        ["All", "Vegetarian", "Non-Vegetarian"]
    )

# Meal type and difficulty
col3, col4 = st.columns(2)

with col3:
    meal_type = st.selectbox(
        "🍽️ Meal Type",
        ["All", "Breakfast", "Lunch", "Dinner", "Snack"]
    )

with col4:
    difficulty = st.selectbox(
        "📊 Difficulty",
        ["All", "Easy", "Medium", "Hard"]
    )

# Cooking time and calories
col5, col6 = st.columns(2)

with col5:
    max_time = st.slider(
        "⏱️ Maximum Cooking Time",
        10, 60, 60
    )

with col6:
    max_calories = st.slider(
        "🔥 Maximum Calories",
        100, 700, 700,
        step=50
    )

# Main ingredient
ingredient = st.text_input(
    "🥕 Main Ingredient",
    placeholder="Example: paneer"
)
st.write("")

# Recipe images
image_urls = {
     "Paneer Tikka": "https://images.unsplash.com/photo-1567188040759-fb8a883dc6d8",
    "Veg Biryani": "https://images.unsplash.com/photo-1589302168068-964664d93dc0",
    "Chicken Biryani": "https://images.unsplash.com/photo-1563379091339-03246963d96c",
    "Pasta Alfredo": "https://images.unsplash.com/photo-1645112411341-6c4fd023714a",
    "Chicken Pasta": "https://images.unsplash.com/photo-1473093295043-cdd812d0e601",
    "Veg Fried Rice": "https://images.unsplash.com/photo-1603133872878-684f208fb84b",
    "Chicken Noodles": "https://images.unsplash.com/photo-1585032226651-759b368d7246",
    "Masala Dosa": "https://images.unsplash.com/photo-1668236543090-82eba5ee5976",
    "Palak Paneer": "https://images.unsplash.com/photo-1631452180519-c014fe946bc7",
    "Chicken Curry": "https://images.unsplash.com/photo-1603894584373-5ac82b2ae398"
}

# Recommendation button
if st.button(
    "🍽️ Recommend Recipes",
    use_container_width=True
):

    # Create recipe text
    recipes["recipe_text"] = (
        recipes["cuisine"].fillna("") + " " +
        recipes["diet"].fillna("") + " " +
        recipes["meal_type"].fillna("") + " " +
        recipes["difficulty"].fillna("") + " " +
        recipes["ingredients"].fillna("")
    )

    # Create user preference text
    preferences = []

    if cuisine != "All":
        preferences.append(cuisine)

    if diet != "All":
        preferences.append(diet)

    if meal_type != "All":
        preferences.append(meal_type)

    if difficulty != "All":
        preferences.append(difficulty)

    if ingredient.strip():
        preferences.append(ingredient)

    user_text = " ".join(preferences)

    # Calculate similarity
    if user_text:

        vectorizer = TfidfVectorizer()

        recipe_vectors = vectorizer.fit_transform(
            recipes["recipe_text"]
        )

        user_vector = vectorizer.transform(
            [user_text]
        )

        similarity = cosine_similarity(
            user_vector,
            recipe_vectors
        )[0]

        recipes["similarity"] = similarity

    else:
        recipes["similarity"] = 0

   # Filter by cooking time and calories
    result = recipes[
        (recipes["cooking_time"] <= max_time) &
        (recipes["calories"] <= max_calories)
    ].copy()

    # Filter by cuisine
    if cuisine != "All":
        result = result[
            result["cuisine"] == cuisine
        ]

    # Filter by diet
    if diet != "All":
        result = result[
            result["diet"] == diet
        ]

    # Filter by meal type
    if meal_type != "All":
        result = result[
            result["meal_type"] == meal_type
        ]

    # Filter by difficulty
    if difficulty != "All":
        result = result[
            result["difficulty"] == difficulty
        ]

    # Filter by ingredient
    if ingredient.strip():
        result = result[
            result["ingredients"].str.contains(
                ingredient,
                case=False,
                na=False
            )
        ]

    # Convert similarity into percentage
    result["match_score"] = (
        result["similarity"] * 100
    ).round(0)

    # Sort by similarity
    result = result.sort_values(
        by="similarity",
        ascending=False
    )

    st.divider()
    st.subheader("🍲 Recommended Recipes")

    if len(result) > 0:

        # Show top 5 recipes
        for _, row in result.head(5).iterrows():

            with st.container(border=True):

                # Recipe image
                if row["name"] in image_urls:
                    st.image(
                        image_urls[row["name"]],
                        width=300
                    )

                # Recipe name
                st.markdown(
                    f"### 🍲 {row['name']}"
                )

                # Recipe details
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.write(
                        f"🌍 **Cuisine:** {row['cuisine']}"
                    )

                with col2:
                    st.write(
                        f"🥗 **Diet:** {row['diet']}"
                    )

                with col3:
                    st.write(
                        f"🍽️ **Meal:** {row['meal_type']}"
                    )

                with col4:
                    st.write(
                        f"📊 **Difficulty:** {row['difficulty']}"
                    )

                st.write(
                    f"⏱️ **Cooking Time:** {row['cooking_time']} min"
                )

                st.write(
                    f"🥕 **Ingredients:** {row['ingredients']}"
                )

                st.write(
                    f"🔥 **Calories:** {row['calories']} kcal"
                )

                st.write(
                    f"⭐ **Match Score:** {row['match_score']}%"
                )

                # Explain recommendation
                reasons = []

                if cuisine != "All" and row["cuisine"] == cuisine:
                    reasons.append(f"{cuisine} cuisine")

                if diet != "All" and row["diet"] == diet:
                    reasons.append(f"{diet} diet")

                if meal_type != "All" and row["meal_type"] == meal_type:
                    reasons.append(f"{meal_type} meal")

                if difficulty != "All" and row["difficulty"] == difficulty:
                    reasons.append(f"{difficulty} difficulty")

                if ingredient.strip() and ingredient.lower() in row["ingredients"].lower():
                    reasons.append(f"contains {ingredient}")

                if row["cooking_time"] <= max_time:
                    reasons.append("fits your cooking time")

                if reasons:
                    explanation = ", ".join(reasons)
                else:
                    explanation = "matches your general preferences"

                st.info(
                    f"💡 **Why recommended:** {explanation.capitalize()}."
                )

                # Detailed instructions
                st.markdown(
                    "#### 👨‍🍳 Detailed Preparation Instructions"
                )

                st.write(
                    row["instructions"]
                )

    else:
        st.warning(
            "No recipes match your preferences. Try changing your filters."
        )