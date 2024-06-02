# Intelligent Recipe Suggestion System For Zero-Waste
Our main goal was to determine and implement a method to help reduce the amount of food wasted by individual households. By using machine learning, we managed to achieve this goal by creating a program which recommends recipes based on at home ingredients prioritized by their expiration dates. 

This repository contains our work in creating the program from its inception to its implementation. 

## Authors 
- Chun-hao (Larry) Chen &nbsp;<a href="https://www.linkedin.com/in/larrychencpa/"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="height: 1em; width:auto;"/></a> &nbsp; <a href="https://github.com/LarryChenCode"> <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" style="height: 1em; width: auto;"/></a>
- Deniz Genlik
- Sanjay Kumar &nbsp;<a href="https://www.linkedin.com/in/sanjay-kumar0330/"><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="height: 1em; width:auto;"/></a> &nbsp; <a href="https://github.com/sanjayk0330"> <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" style="height: 1em; width: auto;"/></a>
- Sevim Polat Genlik

## Summary
We will follow with further details; however, our work can be summarized into the following steps:
1. Preprocessing of the dataset from [Food.com](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=PP_users.csv).
2. Cross-validation of various models on the cleaned dataset to determine the ideal model.
3. Training our ideal model on the cleaned dataset.
4. Determining the correlation between cuisines from various cultures based on ingredients.
5. Using the correlation between cuisines as a metric for clustering.
6. Based on the expiration dates of the ingredients, creating a method for scoring recipes.
7. By using the model and clustering, we construct a function which outputs a recipe from a cuisine which uses similar ingredients and prioritizes their expiration dates.

## Final Product

```
Hi! Welcome to Intelligent Recipe Suggestion System For Zero-Waste! We use machine learning to recommend recipes based
on the ingredients you have at home and your openness to different cuisines. Let's get started!

Would you like to provide your own ingredients? (yes/no)

Note: If you choose 'no', we will use a random set of ingredients for you.

You chose: yes

Enter your ingredients and expiration dates (format: ingredient;YYYY-MM-DD). Type 'done' when finished:
milk is expired and will be ignored.
Invalid format. Please enter in the format: ingredient;YYYY-MM-DD
Added: egg (Expires on: 2024-06-05)
Added: salt (Expires on: 2026-12-01)
Added: pork (Expires on: 2024-06-12)
Added: oil (Expires on: 2026-12-01)

The ingredients you have are similar to the ingredients used in CHINESE cuisine based on our system.

Next, how open are you to trying different cuisines?

Please rate your openness on a scale of 1 to 5, where 1 means 'very slightly open' and 5 means 'extremely open'
1: Very slightly open
2: Slightly open
3: Moderately open
4: Very open
5: Extremely open

Your openness to different cuisines: 3

Ingredients with Expiration Dates:
           Expiration_Date  Days_Left
Ingredient                           
egg             2024-06-05          4
pork            2024-06-12         11
salt            2026-12-01        913
oil             2026-12-01        913

Press Enter to continue...

Here is a recipe you might like:

Recipe Name: chin jao ro su  bamboo and bell pepper stir fry

Ingredients:
- pork
- egg
- sake
- soy sauce
- cornstarch
- garlic clove
- fresh ginger
- oil
- bell pepper
- bamboo shoot
- oyster sauce
- sugar

Steps:
Step 1: combine first 5 ingredients in a bowl
Step 2: stir fry in oil with the garlic and ginger until brown
Step 3: add bell peppers and bamboo shoots
Step 4: stir fry until done , about 5 minutes
Step 5: add sauce during the last minute or so of cooking time

Do you like this recipe? (yes/no)

You chose: no

Here is a recipe you might like:

Recipe Name: needle noodles

Ingredients:
- noodle
- shrimp
- pork
- onion
- ginger
- garlic clove
- green pepper
- scallion
- egg
- bean sprout
- soy sauce
- salt
- sugar
- sesame oil
- chicken stock
- oyster sauce
- vegetable oil

Steps:
Step 1: heat wok
Step 2: add 1 tbsp oil
Step 3: crack eggs into hot oil and stir fry to scramble , remove
Step 4: add additional oil to wok if needed , add onions , ginger , garlic , and light green parts of scallion
Step 5: stir fry for 1-2 minutes until softened
Step 6: add shrimp , pork and peppers , stir fry until heated through
Step 7: add needle noodles and stir fry to mix well
Step 8: add sauce ingredients , continue stirring
Step 9: add green parts of scallions , bean sprouts , eggs , mix well , serve

Do you like this recipe? (yes/no)

You chose: no

Here is a recipe you might like:

Recipe Name: mandarin hot and sour pork soup

Ingredients:
- soup stock
- square bean curd
- dried black mushroom
- wood ear mushroom
- ham
- chili oil
- salt
- sugar
- egg
- pork
- bamboo shoot
- button mushroom
- scallion
- vinegar
- white pepper
- sesame oil
- soy sauce
- cornstarch

Steps:
Step 1: bring soup stock to a boil , add
Step 2: shredded pork , black mushrooms and wood ears
Step 3: cook 2-3 minutes
Step 4: add remainder of ingredients and seasonings reduce heat and simmer for 2 minutes
Step 5: thicken with cornstarch and turn off heat
Step 6: slowly pour in beaten eggs in a thin stream while stirring
Step 7: serve immediately
Step 8: garnish with green onion
Step 9: if soup is to be prepared ahead of time , do not add cornstarch and eggs until serving time
Step 10: otherwise the egg will be overcooked and spoil the appearance
Step 11: soup should be quite hot and sour
Step 12: adjust the hotness with varying amount of white pepper and the sourness with different amounts of vinegar

Do you like this recipe? (yes/no)

You chose: yes

Enjoy your meal!

Thank you for using Intelligent Recipe Suggestion System For Zero-Waste!
```

## Demo Video

We have included a demo video of using the final product below:

https://github.com/sanjayk0330/zero-waste-recipe/assets/146881479/93ad2e61-7adb-4061-af20-94abad51bc3f

# Process

We will now discuss our overall process in constructing the program. More details can be found in the documents provided in the notebooks. 

## Data Preprocessing 

Our initial data from [Food.com](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=PP_users.csv) required preprocessing. For example, one of our goals was to suggest recipes based on the users' preference. In order to deal with this, we decided to train a model to predict the cuisine type based on the ingredients. Then depending on the users' openness to try new cuisines, the function will make suggestions from recipes of similar cuisines. In our data set, each recipe is associated a list of tags which we used to determine the cuisine type. After cleaning our data, we vectorize the ingredients so that we can train our model. 

## Cross-Validation

In order to determine which classification model to use, we initially used cross-validation on a cleaned dataset provided by [Kaggle](https://www.kaggle.com/datasets/kaggle/recipe-ingredients-dataset). From our results shown below, we decided to implement LinearSVC.

<p align="center">
<img src="/image/EDA_Kaggle.png" width="600"/>

We proceeded by applying the same cross-validation on our cleaned data set shown in the following diagram:

<p align="center">
<img src="/image/EDA_Cleaned_Data.png" width="600"/>

Our decrease in accuracy is due to the larger number of cuisines in our cleaned dataset where the Kaggle dataset contains 20 cuisines and our cleaned dataset contains 58 cuisines. Because of the number of cuisines as well as that many cuisines share ingredients, it is expected that the accuracy would decrease. For our purposes, this is not an issue since our program also clusters cuisines with similar ingredients based on the users' preference. 

## Cuisine Correlation and Clustering

In order to consider the users' preference, we created a metric for determining the distance between two cuisines based on their ingredients. This can be calculated through their correlation where cuisines with high correlation have low distance between them. Below, we are attaching a heat map which shows the relation between the cuisines. 

<p align="center">
<img src="/image/cuisine_similarity_heatmap.png" width="600"/>

This can be further illustrated with the following dendrogram:

<p align="center">
<img src="/image/cuisine_similarity_dendogram.png" width="600"/>

From reading the dendrogram, cuisines which create earlier branches are more similar to each other. By using this metric, we are able to cluster the cuisines. 

## Scoring Recipes
In order to prioritize ingredients based on their expiration dates, we defined a metric which was inversely proportional to the number of days until the ingredients expired. This allowed us to create a weighted vector which we applied to our recipes to score them. This can be illustrated by the following picture:

<p align="center">
<img src="/image/Scoring_Recipes.png" width="600"/>

## Program Implementation

By combining our previous steps, we are able to construct our program. We first consider the users' initial ingredients and the ingredients' expiration dates. We then weight the ingredients based on their expiration date. In parallel, the program predicts the users' main cuisine type as well as similar cuisines depending on their openness to try new dishes. Lastly, the program matches recipes of the cuisine types based on the weighted ingredients. The process can be summarized in the picture listed below:

<p align="center">
<img src="/image/product_flowchart.png" width="600"/>
