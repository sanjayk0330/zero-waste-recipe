import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def prioritize_ingredients(ingredients):
    """
    This function takes a list of ingredients with their expiry dates, 
    converts it into a DataFrame, sorts it by expiry date, and returns the sorted DataFrame.
    It also generates a string of ingredients sorted by their expiry date.

    Parameters:
    ingredients (list of tuples): List of ingredients with their expiry dates. 
                                   Each tuple contains ('item', 'expiry_date').

    Returns:
    pd.DataFrame: A DataFrame sorted by the expiry dates.
    str: A string of ingredients sorted by their expiry dates.
    """
    # Convert to DataFrame
    df = pd.DataFrame(ingredients, columns=['item', 'expiry_date'])
    # Convert expiry_date to datetime
    df['expiry_date'] = pd.to_datetime(df['expiry_date'])
    # Sort by expiry_date
    df = df.sort_values(by='expiry_date')

    # Generate a string of ingredients sorted by their expiry dates
    ingredients_str = ' '.join(df['item'].tolist())

    return df, ingredients_str

def load_model_and_vectorizer(model_path, vectorizer_path):
    """
    This function loads the classification model and TF-IDF vectorizer from the specified paths.

    Parameters:
    model_path (str): The file path to the classification model.
    vectorizer_path (str): The file path to the TF-IDF vectorizer.

    Returns:
    model: The loaded classification model.
    vectorizer: The loaded TF-IDF vectorizer.
    """
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_cuisine(ingredients_str, model_path, vectorizer_path):
    """
    This function predicts the cuisine based on the ingredients string.

    Parameters:
    ingredients_str (str): A string of ingredients.
    model_path (str): The file path to the classification model.
    vectorizer_path (str): The file path to the TF-IDF vectorizer.

    Returns:
    str: The predicted cuisine.
    """
    # Load the classification model and TF-IDF vectorizer
    loaded_model, tfidf_vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
    
    # Transform the ingredients string using the TF-IDF vectorizer
    ingredients_tfidf = tfidf_vectorizer.transform([ingredients_str])
    return loaded_model.predict(ingredients_tfidf)[0]

def load_clusters(json_file):
    """
    This function loads the cuisine clusters from a JSON file and processes the DataFrame.

    Parameters:
    json_file (str): The file path to the cuisine clusters JSON file.

    Returns:
    pd.DataFrame: A processed DataFrame with clusters and cuisines.
    """
    clusters_df = pd.read_json(json_file, orient='index')
    clusters_df = clusters_df.reset_index().rename(columns={'index': 'cluster'})
    clusters_df = clusters_df.melt(id_vars=['cluster'], value_name='cuisine')
    clusters_df = clusters_df.dropna(subset=['cuisine']).drop('variable', axis=1).reset_index(drop=True)
    return clusters_df

def find_cuisine_cluster(predicted_cuisine, clusters_df):
    """
    This function finds the cluster for the predicted cuisine and lists all cuisines in the identified cluster.

    Parameters:
    predicted_cuisine (str): The predicted cuisine.
    clusters_df (pd.DataFrame): The DataFrame containing cuisine clusters.

    Returns:
    list: A list of cuisines in the identified cluster.
    """
    predicted_cluster = clusters_df[clusters_df['cuisine'] == predicted_cuisine]['cluster'].values[0]
    cuisine_cluster = clusters_df[clusters_df['cluster'] == predicted_cluster]['cuisine'].tolist()
    return cuisine_cluster

def load_recipes(csv_file):
    """
    This function loads recipes from a CSV file.

    Parameters:
    csv_file (str): The file path to the recipes CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the recipes.
    """
    return pd.read_csv(csv_file)

def filter_recipes_by_cuisine(recipes_df, cuisine_cluster):
    """
    This function filters recipes based on the predicted cuisine cluster.

    Parameters:
    recipes_df (pd.DataFrame): The DataFrame containing the recipes.
    cuisine_cluster (list): A list of cuisines in the identified cluster.

    Returns:
    pd.DataFrame: A DataFrame containing filtered recipes.
    """
    filtered_recipes_df = recipes_df[recipes_df['Cuisine_Tags_str'].isin(cuisine_cluster)]
    filtered_recipes_df = filtered_recipes_df[['id', 'name', 'Cuisine_Tags_str', 'replaced_ingredients_str', 'steps_str', 'Cuisine_Tags', 'replaced_ingredients', 'steps']]
    return filtered_recipes_df

def match_ingredients_with_recipes(prioritized_ingredients, filtered_recipes_df):
    """
    This function matches ingredients with recipes and lists them in order of expiry date.

    Parameters:
    prioritized_ingredients (pd.DataFrame): A DataFrame containing prioritized ingredients.
    filtered_recipes_df (pd.DataFrame): A DataFrame containing filtered recipes.

    Returns:
    list: A list of matched recipes in dictionary format.
    """
    matched_recipes = []
    for index, row in prioritized_ingredients.iterrows():
        ingredient = row['item']
        # Filter recipes that contain the ingredient
        recipes_with_ingredient = filtered_recipes_df[filtered_recipes_df['replaced_ingredients_str'].str.contains(ingredient)]
        matched_recipes.extend(recipes_with_ingredient.to_dict(orient='records'))
    return matched_recipes

def get_file_paths():
    """
    This function returns the file paths for the model, vectorizer, clusters JSON file, and recipes CSV file.

    Returns:
    tuple: A tuple containing the file paths for the model, vectorizer, clusters JSON file, and recipes CSV file.
    """
    model_path = '../model/classification_model_SVC_0529.sav'
    vectorizer_path = '../model/tfidf_vectorizer_0529.sav'
    clusters_json_file = '../data/cuisine_clusters.json'
    recipes_csv_file = '../data/RAW_recipes_cleaned.csv'
    return model_path, vectorizer_path, clusters_json_file, recipes_csv_file

def main(ingredients):
    # Get file paths
    model_path, vectorizer_path, clusters_json_file, recipes_csv_file = get_file_paths()
    
    # Prioritize ingredients
    prioritized_ingredients, ingredients_str = prioritize_ingredients(ingredients)
    
    # Predict cuisine
    predicted_cuisine = predict_cuisine(ingredients_str, model_path, vectorizer_path)
    
    # Load clusters and find cuisine cluster
    clusters_df = load_clusters(clusters_json_file)
    cuisine_cluster = find_cuisine_cluster(predicted_cuisine, clusters_df)
    
    # Load and filter recipes
    recipes_df = load_recipes(recipes_csv_file)
    filtered_recipes_df = filter_recipes_by_cuisine(recipes_df, cuisine_cluster)
    
    # Match ingredients with recipes
    matched_recipes = match_ingredients_with_recipes(prioritized_ingredients, filtered_recipes_df)
    matched_recipes_df = pd.DataFrame(matched_recipes)
    
    return matched_recipes_df

def display_recipe(matched_recipes_df, recipe_index=0):
    matched_recipes_df = matched_recipes_df[['id', 'name', 'Cuisine_Tags', 'replaced_ingredients', 'steps']].head(10)
    matched_recipes_df.columns = ['ID', 'Name', 'Cuisine', 'Ingredients', 'Recipes']
    
    if recipe_index < len(matched_recipes_df):
        recipe_id = matched_recipes_df.iloc[recipe_index]['ID']
        recipe_name = matched_recipes_df.iloc[recipe_index]['Name']
        cuisine = matched_recipes_df.iloc[recipe_index]['Cuisine']
        ingredients = matched_recipes_df.iloc[recipe_index]['Ingredients']
        recipes = matched_recipes_df.iloc[recipe_index]['Recipes']
        
        ingredients = ingredients[1:-1]
        recipes = recipes[1:-1]
        cuisine = cuisine[1:-1]
        
        recipes = recipes.replace("'", "")
        ingredients = ingredients.replace("'", "")
        cuisine = cuisine.replace("'", "")
        
        ingredients_list = ingredients.split(',')
        recipes_list = recipes.split(', ')
        
        print('Ingredients:')
        for ingredient in ingredients_list:
            print(f"- {ingredient.strip()}")
        
        print('\nRecipe:')
        for step, instruction in enumerate(recipes_list, 1):
            print(f"Step {step}: {instruction.strip()}")
        
        print(f'\nRecipe ID: {recipe_id}')
        print(f'Recipe Name: {recipe_name}')
        print(f'Cuisine: {cuisine}')
    else:
        print("Recipe index out of range")