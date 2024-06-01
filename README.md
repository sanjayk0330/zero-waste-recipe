# Zero Waste Recipe Recommender
Our main goal was to determine and implement a method to help reduce the amount of food wasted by individual households. By using machine learning, we managed to achieve this goal by creating a program which recommends recipes based on at home ingredients prioritized by their expiration dates. 

This repository contains our work in creating the program from its inception to its implemention. 

## Summary
We will follow with further details; however, our work can be summarized into the following steps:
1. Preprocessing of the dataset from [Food.com](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=PP_users.csv).
2. Cross-validation of various models on the cleaned dataset to determine the ideal model.
3. Training our ideal model on the cleaned dataset.
4. Determining the correlation between cuisines from various cultures based on ingredients.
5. Using the correlation between cuisines as a metric for clustering. 
6. By using the model and clustering, we construct a function which outputs a reicpe from a cuisine which uses similar ingredients and prioritizes their expiration dates.

## Final Product

```
Hi! Welcome to the Recipe Recommender System! We use machine learning to recommend recipes based
on the ingredients you have at home and your openness to different cuisines. Let's get started!

Would you like to provide your own ingredients? (yes/no)

Note: If you choose 'no', we will use a random set of ingredients for you.

You chose: yes

Enter your ingredients and expiration dates (format: ingredient;YYYY-MM-DD). Type 'done' when finished:
milk is expired and will be ignored.
bread is expired and will be ignored.
Added: egg (Expires on: 2024-06-05)
Invalid format. Please enter in the format: ingredient;YYYY-MM-DD
Added: mayonnaise (Expires on: 2024-12-01)
Added: red wine (Expires on: 2025-12-01)
Added: vinegar (Expires on: 2025-12-01)
Added: salt (Expires on: 2026-12-01)
Added: chicken breast (Expires on: 2024-06-12)
Added: pork (Expires on: 2024-06-12)
Added: black pepper (Expires on: 2026-12-01)
Added: oil (Expires on: 2026-12-01)

The ingredients you have are similar to the ingredients used in CHINESE cuisine based on our system.

Next, how open are you to trying different cuisines?

Please rate your openness on a scale of 1 to 5, where 1 means 'very slightly open' and 5 means 'extremely open.
1: Very slightly open
2: Slightly open
3: Moderately open
4: Very open
5: Extremely open

Your openness to different cuisines: 2

Ingredients with Expiration Dates:
               Expiration_Date  Days_Left
Ingredient                               
egg                 2024-06-05          4
chicken breast      2024-06-12         11
pork                2024-06-12         11
mayonnaise          2024-12-01        183
red wine            2025-12-01        548
vinegar             2025-12-01        548
salt                2026-12-01        913
black pepper        2026-12-01        913
oil                 2026-12-01        913

Press Enter to continue...

Here is a recipe you might like:

Recipe Name: crispie orange chicken rice pilaf salad

Ingredients:
- chicken breast
- egg
- oil
- cornstarch
- flmy
- salt
- black pepper
- orange marmalade
- orzo pastum
- olive oil
- onion
- celery
- long grain rice
- minute rice
- chicken bouillon
- red pepper flake
- boiling water
- honey
- carrot
- broccoli
- green bean
- red pepper
- snow pea pod
- slivered almond
- chinese noodle

Steps:
Step 1: rice pilaf:
Step 2: brown orzo in skillet with 1 / 8 c olive oil
Step 3: saute onions and celery
Step 4: mix with rice , bouillion , red pepper flakes , water and honey in a casserole dish
Step 5: bake 350 degrees covered for 30-40 minutes
Step 6: while the pilaf is cooking , fry the chicken in oil 375 degrees for 3-4 minutes till golden brown
Step 7: cut chicken into 1" pieces
Step 8: beat 2 eggs , mix in 2 tblsp oil
Step 9: dip chicken in egg mixture and coat with mixture of 1c cornstarch and 1 / 2 c flour , salt and pepper
Step 10: when golden brown , drain on paper towels
Step 11: keep warm in oven , till time to assemble salad
Step 12: steam vegetables or microwave 3-4 minutes
Step 13: assemble salad 4 salad bowls
Step 14: add to each
Step 15: divide rice pilaf between 4 bowls -- top off with tender crisp vegetables 5 broccoli florets , 3-4 strips of red pepper , 5 green beans , snapped in half , 6 snow pea pods , 1 tblsp toasted almonds1 tblsp shredded carrots
Step 16: top off with 1 tblsp chinese noodles
Step 17: dip hot chicken pieces into 3 / 4 cup warmed orange marmalade
Step 18: place 7-8 pieces of orange chicken on top of salad
Step 19: serve while still hot --

Do you like this recipe? (yes/no)

You chose: no

Here is a recipe you might like:

Recipe Name: chilli chicken   chinese style

Ingredients:
- chicken breast
- cornstarch
- egg
- vinegar
- soya sauce
- salt
- green chili
- garlic
- green pepper
- shallot
- water
- msg
- oil

Steps:
Step 1: make a batter with the'marinade' ingredients
Step 2: put chicken pieces into it and mix well
Step 3: set aside for 1 / 2 hour
Step 4: heat oil , about 2 inches deep , in a wok / frypan
Step 5: fry chicken pieces , in batches of 6-8 pieces , till cooked through
Step 6: drain fried chicken on a paper towel
Step 7: when all the chicken pieces are fried , remove oil from the pan , letting about 2 tablespoons remain in it
Step 8: in a small bowl mix together cornstarch and water
Step 9: put back the wok on fire and heat the 2 tblsps of oil
Step 10: add green chillies , garlic , green pepper and shallot and stir fry for 30 secsonds
Step 11: pour in soya sauce and vinegar
Step 12: give it two three quick whisks and immediately add the cornstarch-water mixture
Step 13: the sauce will thicken
Step 14: if you find it too thick , add some more water
Step 15: when the bubbles appear , add the fried chicken pieces
Step 16: add salt and a pinch of msg
Step 17: stir fry chicken for 2 minutes and remove from fire
Step 18: serve hot with chinese fried rice / plain rice or noodles

Do you like this recipe? (yes/no)

You chose: yes

Enjoy your meal!

Thank you for using the Recipe Recommender System!
```

# Process

We will now discuss our overall process in constructing the program. More details can be found in the documents provided in the notebooks. 

## Data Preprocessing 

Our initial data from [Food.com](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=PP_users.csv) required preprocessing. For example, one of our goals was to suggest recipes based on the users' preference. In order to deal with this, we decided to train a model to predict the cuisine type based on the ingredients. Then depending on the users' openness to try new cuisines, the function will make suggestions from recipes of similar cuisines. In our data set, each recipe is associated a list of tags which we used to determine the cuisine type. After cleaning our data, we vectorize the ingredients so that we can train our model. 

## Cross-Validation

In order to determine which classification model to use, we initialliy used cross-validation on a cleaned dataset provided by [Kaggle](https://www.kaggle.com/datasets/kaggle/recipe-ingredients-dataset). From our results shown below, we decided to implement LinearSVC.

<p align="center">
<img src="/image/EDA_Kaggle.png" width="600"/>

We proceeded by applying the same cross-validation on our cleaned data set shown in the following diagram:

<p align="center">
<img src="/image/EDA_Cleaned_Data.png" width="600"/>

Our decrease in accuracy is due to the larger number of cuisines in our cleaned dataset where the Kaggle dataset contains 20 cuisines and our cleaned dataset contains 58 cuisines. Because of the number of cuisines as well as that many cuisines share ingredients, it is expected that the accuracy would decrease. For our purposes, this is not an issue since our program also clusters cuisines with similar ingredients based on the users' preference. 

## Cuisine Correlation and Clustering

In order to consider the users' preference, we created a metric for determining the distance between two cuisines based on their ingredients. This can by calculated through their correlation where cuisines with high correlation have low distance between them. Below, we are attaching a heat map which shows the relation between the cuisines. 

<p align="center">
<img src="/image/cuisine_similarity_heatmap.png" width="600"/>

This can be further illustrated with the following dendrogram:

<p align="center">
<img src="/image/cuisine_similarity_dendogram.png" width="600"/>

From reading the dendrogram, cuisines which create earlier branches are more similar to each other. By using this metric, we are able to cluster the cuisines. 

## Program Implementation

By combining our previous steps, we are able to construct our program. We first consider the users' initial ingredients and the ingredients' expiration dates. We then weight the ingredients based on their expiration date. In parallel, the program predicts the users' main cuisine type as well as similar cuisines depending on their openness to try new dishes. Lastly, the program matches recipes of the cuisine types based on the weighted ingredients. The process can be summarized in the picture listed below:

<p align="center">
<img src="/image/Product_Process.png" width="600"/>
