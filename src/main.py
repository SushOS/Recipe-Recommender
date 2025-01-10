import os
import sys
import time
from tqdm import tqdm
import pandas as pd
pd.options.mode.chained_assignment = None

from data_preprocessing import FoodComDataPreprocessor
from recommendation_engine import RecipeRecommender
from feedback_manager import FeedbackManager
from nlp_preference_extraction import NLPPreferenceExtractor
from utils import validate_user_preferences, format_recipe_output
from visualization import plot_top_recommendations, show_recommendations_cuisine_breakdown, plot_cuisine_distribution, plot_rating_distribution,  plot_difficulty_distribution, plot_diet_flags_breakdown_dataset, plot_diet_flags_breakdown_recommendations, plot_correlation_matrix

def incorporate_nlp_preferences_for_user(user_id: int, user_preferences: dict) -> dict:
    """
    1. Load the user's feedback from JSON.
    2. Parse each comment with the NLP extractor.
    3. Aggregate discovered preferences and combine with user_preferences.
    """
    feedback_manager = FeedbackManager(feedback_file='data/feedback.json')
    user_reviews = feedback_manager.get_user_feedback(user_id)

    if not user_reviews:
        print(f"No feedback/reviews found for user {user_id}.")
        return user_preferences

    extractor = NLPPreferenceExtractor()
    discovered = extractor.aggregate_preferences(user_reviews)

    # For demonstration, let's say if they're "concerned_about_calories",
    # we automatically limit them to "low calorie" or "max calories" in the nutrition preferences.
    # or we just store it as a boolean that might influence the recommendation approach.
    if discovered["concerned_about_calories"]:
        print(f"User {user_id} seems concerned about calories (from reviews).")
        # For example, we could reduce their 'calories' target in user_preferences['nutrition_preferences']
        if 'nutrition_preferences' not in user_preferences:
            user_preferences['nutrition_preferences'] = {}
        # If they didn't set a numeric target, let's pick 300 or 400 as a guess:
        user_preferences['nutrition_preferences']['calories'] = 400

    if discovered["healthy_focus"]:
        print(f"User {user_id} mentions healthy or nutrient aspect.")
        # Maybe we set a 'protein' target or something else:
        if 'nutrition_preferences' not in user_preferences:
            user_preferences['nutrition_preferences'] = {}
        user_preferences['nutrition_preferences']['protein'] = 30
        user_preferences['nutrition_preferences']['fat'] = 40
        # Or set a preference for 'fat' to be lower, etc.

    return user_preferences

def main():
    try:
        # ======================
        # 1. Initialize Components
        # ======================
        print("\n1. Initializing components...")
        preprocessor = FoodComDataPreprocessor()
        feedback_manager = FeedbackManager()
        
        # Define file paths
        recipes_path = 'data/raw/RAW_recipes.csv'
        ratings_path = 'data/raw/RAW_interactions.csv'
        
        # ======================
        # 2. Load & Preprocess Data
        # ======================
        sample_size = 70000
        print(f"\nLoading data with sample size: {sample_size}")
        start_time = time.time()
        
        # Load the full dataset, then sample
        recipes_df_full = pd.read_csv(recipes_path)
        recipes_df_sample = recipes_df_full.sample(n=sample_size, random_state=42)
        print(f"Sampled {len(recipes_df_sample)} recipes")
        
        # Preprocess
        recipes_df, recipe_embeddings = preprocessor.load_and_preprocess_data(
            recipes_path=recipes_path,
            ratings_path=ratings_path,
            sample_df=recipes_df_sample
        )
        
        print(f"\nData loading took {time.time() - start_time:.2f} seconds")
        print(f"Processed {len(recipes_df)} recipes")
        
        # ======================
        # 3. Build Recommendation Engine
        # ======================
        print("\n2. Initializing recommendation engine...")
        recommender = RecipeRecommender(recipes_df, recipe_embeddings)
        
        print("Building similarity matrices...")
        with tqdm(total=1, desc="Building matrices") as pbar:
            recommender.build_similarity_matrices()
            pbar.update(1)
            
        print(f"Recommendation engine initialization took {time.time() - start_time:.2f} seconds")
        
        # ======================
        # 4. Dynamically Gather User Preferences
        # ======================
        print("\n3. Please enter your user preferences.")
        
        # user_id (integer)
        while True:
            try:
                user_id_str = input("Enter a user ID (integer): ")
                user_id = int(user_id_str)
                break
            except ValueError:
                print("Invalid integer. Please try again.")
        
        # cuisine_preference (string)
        cuisine_pref = input("Enter cuisine preference (e.g., 'Indian', 'Italian', 'Mexican', 'Chinese', American'): ").strip()
        if not cuisine_pref:
            cuisine_pref = "Indian"  # fallback or empty string
        
        # dietary_restrictions (comma-separated)
        diet_str = input("Enter dietary restrictions (comma-separated, e.g. 'vegetarian, non_vegetarian, gluten_free', 'vegan'): ")
        if diet_str.strip():
            dietary_restrictions = [d.strip() for d in diet_str.split(',') if d.strip()]
        else:
            dietary_restrictions = []
        
        # available_time (integer)
        while True:
            try:
                time_str = input("Enter available time in minutes (integer): ")
                available_time = int(time_str)
                break
            except ValueError:
                print("Invalid integer. Please try again.")
        
        # available_ingredients (comma-separated)
        ing_str = input("Enter available ingredients (comma-separated, e.g. 'tomatoes, onions, etc..'): ")
        if ing_str.strip():
            available_ingredients = [x.strip() for x in ing_str.split(',') if x.strip()]
        else:
            available_ingredients = []
        
        # max_difficulty (float or int)
        while True:
            try:
                diff_str = input("Enter max difficulty (1-5, float allowed): ")
                max_difficulty = float(diff_str)
                break
            except ValueError:
                print("Invalid number. Please try again.")
        
        # nutrition_preferences (enter separate values for calories, protein, fat)
        while True:
            try:
                cal_str = input("Enter target calories (float): ")
                calories = float(cal_str)
                break
            except ValueError:
                print("Invalid float. Please try again.")
        
        while True:
            try:
                prot_str = input("Enter target protein in grams (float): ")
                protein = float(prot_str)
                break
            except ValueError:
                print("Invalid float. Please try again.")
        
        while True:
            try:
                fat_str = input("Enter target fat in grams (float): ")
                fat = float(fat_str)
                break
            except ValueError:
                print("Invalid float. Please try again.")
        
        user_preferences = {
            'user_id': user_id,
            'cuisine_preference': cuisine_pref,
            'dietary_restrictions': dietary_restrictions,
            'available_time': available_time,
            'available_ingredients': available_ingredients,
            'max_difficulty': max_difficulty,
            'nutrition_preferences': {
                'calories': calories,
                'protein': protein,
                'fat': fat
            }
        }
        
        # Validate the user preferences
        try:
            validate_user_preferences(user_preferences)
        except ValueError as ve:
            print(f"Validation Error: {ve}")
            # Could exit or continue with fallback
            sys.exit(1)
        
        # ======================
        # 5. Generate Recommendations
        # ======================
        print("\n4. Generating recommendations...")
        start_time = time.time()
        recommendations = recommender.get_recommendations(
            user_id=user_preferences['user_id'],
            preferences=user_preferences
        )
        
        print(f"\nRecommendation generation took {time.time() - start_time:.2f} seconds")
        
        # Display recommendations
        print("\n5. Top Recommendations:")
        for idx, (_, recipe) in enumerate(recommendations.iterrows(), 1):
            print(f"\nRecipe {idx}:")
            formatted_recipe = format_recipe_output(recipe)
            for key, value in formatted_recipe.items():
                print(f"{key}: {value}")
            print("-" * 50)
            
            # Collect feedback
            try:
                rating_str = input("Please rate this recipe (1-5), or press Enter to skip: ")
                if rating_str.strip():
                    rating = float(rating_str)
                else:
                    rating = None

                comments = input("Any comments? (optional): ")
                
                if rating is not None:
                    feedback_manager.save_feedback(
                        user_id=user_preferences['user_id'],
                        recipe_id=recipe['recipe_id'],
                        rating=rating,
                        comments=comments
                    )
            except KeyboardInterrupt:
                print("\nFeedback collection interrupted by user")
                break
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

    updated_prefs = incorporate_nlp_preferences_for_user(user_id, user_preferences)
    print("Updated user prefs after NLP extraction:", updated_prefs)
    
    plot_rating_distribution(recipes_df, 'avg_rating', "Distribution of Average Ratings in Sample Dataset")
    plot_cuisine_distribution(recipes_df, "Cuisine Distribution in Sample Dataset")
    plot_diet_flags_breakdown_dataset(recipes_df, ['is_vegetarian','is_vegan','is_gluten_free','is_dairy_free', 'is_non_vegetarian'])
    plot_correlation_matrix(recipes_df, ['cook_time_minutes','avg_rating','difficulty','calories','protein','fat'])
    plot_top_recommendations(recommendations, top_n=5) # top recommendations based on the average rating
    show_recommendations_cuisine_breakdown(recommendations) # cusine breakdown among the recommendations
    plot_difficulty_distribution(recommendations, "Difficulty Distribution in the Recommendations")
    plot_diet_flags_breakdown_recommendations(recommendations, ['is_vegetarian','is_vegan','is_gluten_free','is_dairy_free', 'is_non_vegetarian'])


if __name__ == "__main__":
    main()