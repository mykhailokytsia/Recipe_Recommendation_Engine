import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Download stopwords and wordnet
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing and Parsing of Ingredients
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def ingredient_parser(ingredient):
    ingredient = ingredient.lower()
    ingredient = re.sub(r'[^\w\s]', '', ingredient)
    ingredient = ' '.join([word for word in ingredient.split() if word not in stop_words])
    ingredient = ' '.join([lemmatizer.lemmatize(word) for word in ingredient.split()])
    return ingredient

# Example recipes dataset
recipes = [
    ['egg yolk', 'parmesan cheese', 'pancetta', 'spaghetti', 'garlic', 'olive oil'],
    ['tomato sauce', 'pasta', 'zucchini', 'onion', 'cheese'],
    # Add more recipes as needed...
]

# Train Word2Vec model
model = Word2Vec(
    sentences=recipes,
    vector_size=100,
    window=6,
    min_count=1,
    workers=8,
    sg=0  # CBOW model
)

# Save and load the model (optional)
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")

# Function to average word embeddings
def avg_word_embeddings(ingredients, model):
    embeddings = [model.wv[word] for word in ingredients if word in model.wv]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)

# Prepare TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit([' '.join(recipe) for recipe in recipes])

# Function to calculate TF-IDF weighted embeddings
def tfidf_weighted_embeddings(ingredients, model, vectorizer):
    tfidf = vectorizer.transform([' '.join(ingredients)]).toarray()[0]
    embeddings = [model.wv[word] * tfidf[idx] for idx, word in enumerate(vectorizer.get_feature_names_out()) if word in model.wv]
    return np.sum(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)

# Function to recommend recipes
def recommend_recipes(user_ingredients, recipes, model, vectorizer, method='avg'):
    if method == 'avg':
        user_embedding = avg_word_embeddings(user_ingredients, model)
    elif method == 'tfidf':
        user_embedding = tfidf_weighted_embeddings(user_ingredients, model, vectorizer)
    
    recipe_embeddings = []
    for recipe in recipes:
        if method == 'avg':
            recipe_embedding = avg_word_embeddings(recipe, model)
        elif method == 'tfidf':
            recipe_embedding = tfidf_weighted_embeddings(recipe, model, vectorizer)
        recipe_embeddings.append(recipe_embedding)
    
    similarities = cosine_similarity([user_embedding], recipe_embeddings).flatten()
    recommended_indices = similarities.argsort()[-5:][::-1]  # Top 5 recommendations
    return [(recipes[idx], similarities[idx]) for idx in recommended_indices]

# Streamlit App
st.title('Recipe Recommendation System')
st.header('Enter ingredients to get recipe recommendations')

user_input = st.text_area("Ingredients", "egg yolk, Parmesan cheese, garlic")

if st.button("Get Recommendations"):
    parsed_ingredients = ingredient_parser(user_input)
    parsed_ingredients = parsed_ingredients.split()
    recommendations = recommend_recipes(parsed_ingredients, recipes, model, vectorizer, method='avg')
    for recipe, score in recommendations:
        st.write(f"Recipe: {recipe}, Similarity Score: {score:.2f}")

# Run the Streamlit app with `streamlit run your_script.py`
