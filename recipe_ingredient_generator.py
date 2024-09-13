import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

try:
    
    with open('archive/train.json', 'r') as train_file:
        train_data = json.load(train_file)
    with open('archive/test.json', 'r') as test_file:
        test_data = json.load(test_file)


    combined_data = train_data + test_data
    
    print("Datasets loaded successfully")
except Exception as e:
    print(f"Error loading datasets: {e}")

df = pd.DataFrame(combined_data)

df = df[['cuisine', 'ingredients']]

df['ingredients_str'] = df['ingredients'].apply(lambda x: ' '.join(x))

print("Cleaned data:")
print(df.head())

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['ingredients_str'])

def recommend_recipes(input_ingredients):
    """
    Recommend recipes based on user input ingredients.
    """
    try:
        input_vec = vectorizer.transform([input_ingredients])
        
        cosine_sim = cosine_similarity(input_vec, tfidf_matrix)
        
        recommendations = cosine_sim.argsort()[0][-5:] 
        return df.iloc[recommendations][['cuisine', 'ingredients']] 
    
    except Exception as e:
        print(f"Error recommending recipes: {e}")
        return []

substitutions = {
    'butter': 'margarine',
    'milk': 'almond milk',
    'egg': 'flaxseed meal'
}

def get_substitution(ingredient):
    """
    Get ingredient substitution.
    """
    return substitutions.get(ingredient, 'No substitution found')

def main():
    """
    Main function for user interaction.
    """
    print("Welcome to the Recipe Ingredient Generator!")
    while True:
        user_input = input("Enter ingredients (or 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            break
        
        print("Processing your input...")
        
        print("\nRecommended recipes:")
        recipes = recommend_recipes(user_input)
        for i, (cuisine, ingredients) in enumerate(zip(recipes['cuisine'], recipes['ingredients']), 1):
            print(f"{i}. {cuisine} - {', '.join(ingredients)}")
        
        print("\nIngredient substitution:")
        for ingredient in user_input.split(','):
            ingredient = ingredient.strip()
            print(f"{ingredient}: {get_substitution(ingredient)}")

if __name__ == "__main__":
    main()
