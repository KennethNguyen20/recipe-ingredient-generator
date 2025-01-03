from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load datasets
try:
    with open('archive/train.json', 'r') as train_file:
        train_data = json.load(train_file)
    with open('archive/test.json', 'r') as test_file:
        test_data = json.load(test_file)

    combined_data = train_data + test_data
    df = pd.DataFrame(combined_data)
    df = df[['cuisine', 'ingredients']]
    df['ingredients_str'] = df['ingredients'].apply(lambda x: ' '.join(x))

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['ingredients_str'])

    print("Datasets loaded successfully")
except Exception as e:
    print(f"Error loading datasets: {e}")

# Ingredient substitution dictionary
substitutions = {
    'butter': 'margarine',
    'milk': 'almond milk',
    'egg': 'flaxseed meal'
}

# API endpoints
@app.route('/recommend', methods=['POST'])
def recommend_recipes():
    data = request.json
    input_ingredients = data.get('ingredients', '')
    try:
        input_vec = vectorizer.transform([input_ingredients])
        cosine_sim = cosine_similarity(input_vec, tfidf_matrix)
        recommendations = cosine_sim.argsort()[0][-5:]
        recommended_recipes = df.iloc[recommendations][['cuisine', 'ingredients']].to_dict(orient='records')
        return jsonify({'status': 'success', 'data': recommended_recipes})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/substitute', methods=['POST'])
def get_substitutions():
    data = request.json
    ingredients = data.get('ingredients', '').split(',')
    substitutions_result = {ingredient.strip(): substitutions.get(ingredient.strip(), 'No substitution found') for ingredient in ingredients}
    return jsonify({'status': 'success', 'data': substitutions_result})

if __name__ == '__main__':
    app.run(debug=True)
