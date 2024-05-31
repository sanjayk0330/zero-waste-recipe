import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import inflect
import ast
from scipy.sparse import lil_matrix, csr_matrix

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load the Recipes dataset
Recipes = pd.read_csv('../data/RAW_recipes_with_clusters.csv')

# Load the list of all ingredients
with open('../data/all_ingredients_list.txt', 'r') as f:
    all_ingredients_list = [line.strip() for line in f]

# Initialize inflect engine for singularizing ingredients
p = inflect.engine()

def to_singular(ingredient):
    return p.singular_noun(ingredient) or ingredient

# Define a function to process a list of ingredients
def process_ingredient_list(ingredient_list):
    return [to_singular(ingredient) for ingredient in ingredient_list]

def generate_random_date():
    today = datetime.now()
    random_days = random.randint(1, 20)
    expiration_date = today + timedelta(days=random_days)
    return expiration_date.strftime("%Y-%m-%d")

def get_user_openness():
    print("On a scale of 1 to 5, how open are you to trying different cuisines?")
    print("1: Not open at all")
    print("2: Slightly open")
    print("3: Moderately open")
    print("4: Very open")
    print("5: Extremely open")
    return int(input("Enter a number between 1 and 5: "))

def get_user_ingredients():
    ingredients = {}
    print("\nEnter your ingredients and expiration dates (format: ingredient;YYYY-MM-DD). Type 'done' when finished:")
    while True:
        user_input = input("Ingredient and Expiration Date: ").strip()
        if user_input.lower() == 'done':
            break
        try:
            ingredient, date_str = user_input.split(';')
            expiration_date = datetime.strptime(date_str, '%Y-%m-%d')  # Validate date format
            if expiration_date < datetime.now():
                print(f"{ingredient.strip()} is expired and will be ignored.")
            else:
                ingredients[ingredient.strip()] = date_str.strip()
        except ValueError:
            print("Invalid format. Please enter in the format: ingredient;YYYY-MM-DD")
    return ingredients

def display_ingredients_with_expiration(df_ingredients):
    print("\nIngredients with Expiration Dates:")
    df_display = df_ingredients[['Expiration_Date', 'Days_Left']].copy()
    df_display['Expiration_Date'] = df_display['Expiration_Date'].dt.strftime('%Y-%m-%d')
    # sort by Days_Left
    df_display = df_display.sort_values(by='Days_Left', ascending=True)
    print(df_display)

def display_recipe(index):
    recipe = Recipes_Preffered.iloc[index]
    print(f"\nRecipe Name: {recipe['name']}")
    print("\nIngredients:")
    for ingredient in ast.literal_eval(recipe['replaced_ingredients']):
        print(f"- {ingredient}")
    print("\nSteps:")
    for i, step in enumerate(ast.literal_eval(recipe['steps']), 1):
        print(f"Step {i}: {step}")

def main():
    global Recipes_Preffered  # Ensure Recipes_Preffered is accessible in display_recipe

    openness_to_different_cuisines = get_user_openness()
    
    print("\nWould you like to provide your own ingredients? (yes/no)")
    if input().strip().lower() == 'yes':
        ingredients_at_home = get_user_ingredients()
    else:
        random_ingredients = [
            'olive oil', 'chicken drumstick', 'beef', 'parsley', 'salmon',
            'bacon', 'sugar', 'onion', 'garlic', 'tomato', 'mayonnaise',
            'cucumber', 'lemon', 'yogurt', 'pepper', 'eggplant', 'milk', 'lamb',
            'chili', 'potato', 'carrot', 'cabbage', 'broccoli', 'lettuce'
        ]
        ingredients_at_home = {ingredient: generate_random_date() for ingredient in random_ingredients}
        print(f"\nUsing random ingredients: {list(ingredients_at_home.keys())}")
    
    ingredients_at_home_processed = process_ingredient_list(list(ingredients_at_home.keys()))
    ingredients_at_home_appended = [" ".join(ingredients_at_home_processed)]
    
    cuisine_ingredient_model = joblib.load('../model/classification_model_SVC_0530.sav')
    vectorizer = joblib.load('../model/tfidf_vectorizer_0529.sav')
    
    # Transform the ingredients at home
    ingredients_at_home_appended = vectorizer.transform(ingredients_at_home_appended)
    ingredients_at_home_cuisine_type = cuisine_ingredient_model.predict(ingredients_at_home_appended)[0]
    print(f"\nThe ingredients you have are similar to the ingredients used in {ingredients_at_home_cuisine_type} cuisine.")
    
    # Convert to a DataFrame
    df_ingredients = pd.DataFrame(list(ingredients_at_home.items()), columns=['Ingredient', 'Expiration_Date'])
    df_ingredients['Expiration_Date'] = pd.to_datetime(df_ingredients['Expiration_Date'])
    today = pd.Timestamp(datetime.now())
    df_ingredients['Days_Left'] = (df_ingredients['Expiration_Date'] - today).dt.days
    df_ingredients.set_index('Ingredient', inplace=True)
    
    # Display ingredients with expiration dates
    display_ingredients_with_expiration(df_ingredients)
    print("\nPress Enter to continue...")
    input("\nPress Enter to continue...")
    
    # Create the weighted pantry vector
    weighted_pantry_vector = np.array([1/(0.01+df_ingredients.at[ingredient, 'Days_Left']) if ingredient in df_ingredients.index else 0 for ingredient in all_ingredients_list])
    
    # Load the appropriate cuisine clusters file based on user openness
    if openness_to_different_cuisines == 1:
        with open('../data/cuisine_clusters30.json', 'r') as f:
            clusters_for_recipes = json.load(f)
    elif openness_to_different_cuisines == 2:
        with open('../data/cuisine_clusters20.json', 'r') as f:
            clusters_for_recipes = json.load(f)
    elif openness_to_different_cuisines == 3:
        with open('../data/cuisine_clusters15.json', 'r') as f:
            clusters_for_recipes = json.load(f)
    elif openness_to_different_cuisines == 4:
        with open('../data/cuisine_clusters10.json', 'r') as f:
            clusters_for_recipes = json.load(f)
    else:
        with open('../data/cuisine_clusters5.json', 'r') as f:
            clusters_for_recipes = json.load(f)
    
    # Define a dictionary to map each cuisine to its cluster number
    cuisine_to_cluster = {cuisine: cluster for cluster, cuisines in clusters_for_recipes.items() for cuisine in cuisines}
    
    def get_cluster_number(cuisine_tags):
        return np.int64(cuisine_to_cluster.get(cuisine_tags[0], None))
    
    Preferred_cuisine_number = get_cluster_number([ingredients_at_home_cuisine_type])
    
    # Filter the recipes based on the preferred cuisine
    if openness_to_different_cuisines == 1:
        Recipes_Preffered = Recipes[Recipes['Clusters30'] == Preferred_cuisine_number]
    elif openness_to_different_cuisines == 2:
        Recipes_Preffered = Recipes[Recipes['Clusters20'] == Preferred_cuisine_number]
    elif openness_to_different_cuisines == 3:
        Recipes_Preffered = Recipes[Recipes['Clusters15'] == Preferred_cuisine_number]
    elif openness_to_different_cuisines == 4:
        Recipes_Preffered = Recipes[Recipes['Clusters10'] == Preferred_cuisine_number]
    else:
        Recipes_Preffered = Recipes[Recipes['Clusters5'] == Preferred_cuisine_number]
    
    Recipes_Preffered.reset_index(drop=True, inplace=True)
    
    ingredients_of_recipes = Recipes_Preffered['replaced_ingredients'].apply(ast.literal_eval)
    Recipes_Preffered['Cuisine_Tags'] = Recipes_Preffered['Cuisine_Tags'].apply(ast.literal_eval)
    
    num_recipes = len(ingredients_of_recipes)
    num_ingredients = len(all_ingredients_list)
    
    ingredient_to_index = {ingredient: idx for idx, ingredient in enumerate(all_ingredients_list)}
    
    binary_matrix = lil_matrix((num_recipes, num_ingredients), dtype=int)
    
    for i, recipe_ingredients in enumerate(ingredients_of_recipes):
        for ingredient in recipe_ingredients:
            if ingredient in ingredient_to_index:
                j = ingredient_to_index[ingredient]
                binary_matrix[i, j] = 1
    
    binary_matrix_csr = binary_matrix.tocsr()
    weighted_pantry_vector_sparse = csr_matrix(weighted_pantry_vector)
    recipe_scores = binary_matrix_csr.dot(weighted_pantry_vector_sparse.T)
    recipe_scores = np.array(recipe_scores.toarray().flatten().tolist())
    
    # Find the indices of the 10 largest entries
    indices_of_largest_entries = np.argsort(recipe_scores)[-10:]
    
    index = 0
    while index < len(indices_of_largest_entries):
        display_recipe(indices_of_largest_entries[index])
        print("\nDo you like this recipe? (yes/no)")
        response = input().strip().lower()
        if response == 'yes':
            print("\nEnjoy your meal!")
            break
        index += 1
        if index >= len(indices_of_largest_entries):
            print("No more recipes to show.")
            
if __name__ == "__main__":
    main()