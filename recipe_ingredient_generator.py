import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load the dataset from JSON files
try:
    # Load train and test datasets
    with open('archive/train.json', 'r') as train_file:
        train_data = json.load(train_file)
    with open('archive/test.json', 'r') as test_file:
        test_data = json.load(test_file)

    # Combine train and test datasets into one
    combined_data = train_data + test_data
    
    print("Datasets loaded successfully")
except Exception as e:
    print(f"Error loading datasets: {e}")

# Convert the JSON data to a Pandas DataFrame
# The JSON structure has 'id', 'cuisine', and 'ingredients' fields
df = pd.DataFrame(combined_data)

# Select only relevant columns ('ingredients' for recipe matching)
df = df[['cuisine', 'ingredients']]

# Convert the list of ingredients to a string for TF-IDF processing
df['ingredients_str'] = df['ingredients'].apply(lambda x: ' '.join(x))

# Preview the cleaned data to verify loading and cleaning worked
print("Cleaned data:")
print(df.head())

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['ingredients_str'])

def recommend_recipes(input_ingredients):
    """
    Recommend recipes based on user input ingredients.
    """
    try:
        # Transform the input ingredients into the TF-IDF vector space
        input_vec = vectorizer.transform([input_ingredients])
        
        # Calculate cosine similarity between input and all recipes
        cosine_sim = cosine_similarity(input_vec, tfidf_matrix)
        
        # Get the top 5 recommendations (recipes with the highest similarity)
        recommendations = cosine_sim.argsort()[0][-5:]  # Get indices of the top 5 matches
        return df.iloc[recommendations][['cuisine', 'ingredients']]  # Return the cuisine and ingredients of the recommended recipes
    
    except Exception as e:
        print(f"Error recommending recipes: {e}")
        return []

# Basic ingredient substitution model
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
    print("Welcome to the Recipe Ingredient Generator!")  # Verify the program started
    while True:
        # Get ingredients input from the user
        user_input = input("Enter ingredients (or 'exit' to quit): ")
        
        # Exit condition
        if user_input.lower() == 'exit':
            break
        
        print("Processing your input...")  # Debugging line to check input is received
        
        # Get recommended recipes
        print("\nRecommended recipes:")
        recipes = recommend_recipes(user_input)
        for i, (cuisine, ingredients) in enumerate(zip(recipes['cuisine'], recipes['ingredients']), 1):
            print(f"{i}. {cuisine} - {', '.join(ingredients)}")
        
        # Provide ingredient substitutions
        print("\nIngredient substitution:")
        for ingredient in user_input.split(','):
            ingredient = ingredient.strip()
            print(f"{ingredient}: {get_substitution(ingredient)}")

if __name__ == "__main__":
    main()
