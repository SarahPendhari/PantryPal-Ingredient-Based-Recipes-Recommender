import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed data and model components
train_data = pd.read_csv('preprocessed_data/train_data.csv')
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('models/X_train_vec.pkl', 'rb') as f:
    X_train_vec = pickle.load(f)

# Function to recommend recipes based on ingredients
def recommend_recipes(ingredients):
    ingredients_vec = vectorizer.transform([' '.join(ingredients)])
    similarities = cosine_similarity(ingredients_vec, X_train_vec).flatten()
    indices = similarities.argsort()[-3:][::-1]  # Get top 3 recipes
    return train_data.iloc[indices][['recipe_name']]

# Streamlit app
st.title('PantryPal: Ingredient-Based Recipe Recommendation System')

st.header('Enter the ingredients you have:')
ingredients_input = st.text_area("Enter ingredients separated by commas", "")

if st.button('Get Recipes'):
    if ingredients_input:
        ingredients_list = [ingredient.strip().lower() for ingredient in ingredients_input.split(',')]
        recommended_recipes = recommend_recipes(ingredients_list)
        st.write("Recommended Recipes:")
        for index, row in recommended_recipes.iterrows():
            st.write(f"- {row['recipe_name']}")
    else:
        st.write("Please enter some ingredients.")
