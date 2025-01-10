from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List, Optional
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix

class RecipeRecommender:
    def __init__(self, recipes_df: pd.DataFrame, recipe_embeddings):
        self.recipes_df = recipes_df.reset_index(drop=True)
        self.recipe_embeddings = recipe_embeddings
        self.content_similarity_matrix = None
        
        # Collaborative filtering
        self.user_recipe_matrix = None
        self.user_to_idx = {}
        self.recipe_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_recipe = {}
        
        self.feedback_history = defaultdict(list)
    
    def build_similarity_matrices(self):
        print("\nBuilding similarity matrices...")
        
        # === CONTENT-BASED SIMILARITY ===
        print("Computing content-based similarity in chunks...")
        n_samples = self.recipe_embeddings.shape[0]
        chunk_size = 1000
        self.content_similarity_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
        
        for start_idx in tqdm(range(0, n_samples, chunk_size), desc="Computing similarity"):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = self.recipe_embeddings[start_idx:end_idx]
            chunk_sim = cosine_similarity(chunk, self.recipe_embeddings, dense_output=False)
            self.content_similarity_matrix[start_idx:end_idx] = chunk_sim.toarray()
        
        # === COLLABORATIVE FILTERING MATRIX ===
        if 'avg_rating' in self.recipes_df.columns or 'rating_count' in self.recipes_df.columns:
            self._build_collaborative_matrix()
    
    def _build_collaborative_matrix(self):
        print("Building collaborative filtering matrix (if possible)...")
        
        if 'user_id' not in self.recipes_df.columns or 'avg_rating' not in self.recipes_df.columns:
            print("Not enough columns to build CF matrix. Skipping CF.")
            return
        
        unique_users = self.recipes_df['user_id'].unique()
        unique_recipes = self.recipes_df['recipe_id'].unique()
        
        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.recipe_to_idx = {rid: idx for idx, rid in enumerate(unique_recipes)}
        self.idx_to_user = {v: k for k, v in self.user_to_idx.items()}
        self.idx_to_recipe = {v: k for k, v in self.recipe_to_idx.items()}
        
        user_indices = self.recipes_df['user_id'].map(self.user_to_idx).values
        recipe_indices = self.recipes_df['recipe_id'].map(self.recipe_to_idx).values
        
        ratings_values = self.recipes_df['avg_rating'].values
        
        self.user_recipe_matrix = csr_matrix(
            (ratings_values, (user_indices, recipe_indices)),
            shape=(len(unique_users), len(unique_recipes))
        )
    
    def get_recommendations(self,
                            user_id: Optional[int] = None,
                            preferences: Dict = None,
                            n_recommendations: int = 5) -> pd.DataFrame:
        print("\nGenerating recommendations...")
        if preferences is None:
            preferences = {}
        
        # 1. Content-based partial score
        content_scores = self._get_content_based_scores(preferences)
        
        # 2. Collaborative partial score
        if user_id is not None and self.user_recipe_matrix is not None and user_id in self.user_to_idx:
            cf_scores = self._get_collaborative_scores(user_id)
            final_scores = 0.7 * content_scores + 0.3 * cf_scores
        else:
            final_scores = content_scores
        
        # 3. Apply full filters (cuisine, dietary, time, difficulty, ingredients)
        filtered_df = self._apply_filters(preferences)
        
        if filtered_df.empty:
            # === Fallback when no recipes match ===
            print("\nNo recipes match your constraints, but here are some of the closest dishes you can make! SInce I am recommending you the dishes using the fallback mechanism, these recipes might not meet your preferences and constraints. Thanks and Enjoy!")
            
            # Relax some constraints in a fallback approach:
            fallback_df = self._apply_filters_fallback(preferences)
            
            # If fallback is still empty, show top-scoring recipes from entire dataset
            if fallback_df.empty:
                print("\nEven fallback constraints are too strict! Showing any top recipes:")
                fallback_df = self.recipes_df
            
            # Re-rank the fallback_df using final_scores
            fallback_indices = fallback_df.index.values
            fallback_subset_scores = final_scores[fallback_indices]
            top_idx_in_fallback = np.argsort(fallback_subset_scores)[-n_recommendations:]
            recommended_indices = fallback_indices[top_idx_in_fallback]
            
            return self.recipes_df.iloc[recommended_indices].sort_values(by='recipe_id')
        
        # If not empty, proceed as normal
        filtered_indices = filtered_df.index.values
        subset_scores = final_scores[filtered_indices]
        top_idx_in_filtered = np.argsort(subset_scores)[-n_recommendations:]
        
        recommended_indices = filtered_indices[top_idx_in_filtered]
        return self.recipes_df.iloc[recommended_indices].sort_values(by='recipe_id')
    
    def _apply_filters(self, preferences: Dict) -> pd.DataFrame:
        """ Applies the strict, full set of user constraints. """
        filtered_df = self.recipes_df.copy()
        
        # Cuisine
        if preferences.get('cuisine_preference'):
            if 'cuisine' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['cuisine'].str.lower() == preferences['cuisine_preference'].lower()
                ]
            else:
                print("No 'cuisine' column found. Cannot filter by cuisine.")
        
        # Dietary
        if preferences.get('dietary_restrictions'):
            for restr in preferences['dietary_restrictions']:
                col_name = f'is_{restr}'
                if col_name in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[col_name] == True]
        
        # Difficulty
        if preferences.get('max_difficulty'):
            filtered_df = filtered_df[
                filtered_df['difficulty'] <= preferences['max_difficulty']
            ]
        
        # Cooking time
        if preferences.get('available_time'):
            if 'cook_time_minutes' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['cook_time_minutes'] <= preferences['available_time']
                ]
            elif 'minutes' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['minutes'] <= preferences['available_time']
                ]
        
        # Must-have ingredients
        if preferences.get('available_ingredients'):
            req_set = set(x.lower() for x in preferences['available_ingredients'])
            
            def has_required(ing_list):
                ing_lower = set(i.lower() for i in ing_list)
                return req_set.issubset(ing_lower)
            
            filtered_df = filtered_df[filtered_df['ingredients_list'].apply(has_required)]
        
        return filtered_df
    
    def _apply_filters_fallback(self, preferences: Dict) -> pd.DataFrame:
        """
        A more relaxed approach:
          - Skips the 'cuisine' filter or certain dietary restrictions
          - Possibly still respects time, difficulty, or partial ingredient matching
          Adjust as needed to decide which constraints to relax.
        """
        relaxed_df = self.recipes_df.copy()
        
        # Example fallback:
        # 1. Skip cuisine filter entirely
        # 2. Skip dietary restrictions
        # 3. Keep time, difficulty, must-have ingredients
        # (Feel free to tweak any of these as you see fit)

        # Skip cuisine
        # (no check for 'cuisine' even if it exists)
        
        # Skip dietary restrictions
        # (do nothing for is_vegetarian, is_gluten_free, etc.)

        # Difficulty
        if preferences.get('max_difficulty'):
            relaxed_df = relaxed_df[
                relaxed_df['difficulty'] <= preferences['max_difficulty']
            ]
        
        # Cooking time
        if preferences.get('available_time'):
            if 'cook_time_minutes' in relaxed_df.columns:
                relaxed_df = relaxed_df[
                    relaxed_df['cook_time_minutes'] <= preferences['available_time']
                ]
            elif 'minutes' in relaxed_df.columns:
                relaxed_df = relaxed_df[
                    relaxed_df['minutes'] <= preferences['available_time']
                ]
        
        # Must-have ingredients
        if preferences.get('available_ingredients'):
            req_set = set(x.lower() for x in preferences['available_ingredients'])
            
            def has_required(ing_list):
                ing_lower = set(i.lower() for i in ing_list)
                return req_set.issubset(ing_lower)
            
            relaxed_df = relaxed_df[relaxed_df['ingredients_list'].apply(has_required)]
        
        return relaxed_df
    
    def _get_content_based_scores(self, preferences: Dict) -> np.ndarray:
        """ Same as before, or as you customized. """
        # Example weighting
        ING_WEIGHT = 5.0
        NUT_WEIGHT = 3.0
        TIME_WEIGHT = 1.0
        
        n = len(self.recipes_df)
        scores = np.zeros(n, dtype=np.float32)
        
        # Ingredient overlap
        if preferences.get('available_ingredients'):
            def ingredient_overlap(ing_list):
                if not ing_list:
                    return 0.0
                overlap = set(i.lower() for i in ing_list) & set(a.lower() for a in preferences['available_ingredients'])
                return len(overlap) / float(len(ing_list))
            overlap_vals = self.recipes_df['ingredients_list'].apply(ingredient_overlap).values
            scores += overlap_vals * ING_WEIGHT
        
        # Nutrition
        if preferences.get('nutrition_preferences'):
            nut_scores = self._calculate_nutrition_scores(preferences['nutrition_preferences'])
            scores += nut_scores * NUT_WEIGHT
        
        # Time
        if preferences.get('available_time'):
            time_vals = np.zeros(n, dtype=np.float32)
            if 'cook_time_minutes' in self.recipes_df.columns:
                ratio = 1.0 - (self.recipes_df['cook_time_minutes'] / preferences['available_time'])
                time_vals = ratio.clip(lower=0).values
            elif 'minutes' in self.recipes_df.columns:
                ratio = 1.0 - (self.recipes_df['minutes'] / preferences['available_time'])
                time_vals = ratio.clip(lower=0).values
            
            scores += time_vals * TIME_WEIGHT
        
        # Normalize
        min_val, max_val = scores.min(), scores.max()
        if max_val > min_val:
            scores = (scores - min_val) / (max_val - min_val)
        else:
            scores = np.zeros_like(scores)
        
        return scores

    def _calculate_nutrition_scores(self, nutrition_prefs: Dict) -> np.ndarray:
        """ Sample difference-based approach for nutrition. """
        scores = np.zeros(len(self.recipes_df), dtype=np.float32)
        
        for nut, target_val in nutrition_prefs.items():
            if nut in self.recipes_df.columns:
                current_vals = self.recipes_df[nut].fillna(0).astype(float)
                max_val = current_vals.max() if current_vals.max() != 0 else 1
                diff = abs(current_vals - float(target_val))
                inv_diff = (max_val - diff).clip(lower=0)
                scores += inv_diff.values
        
        return scores
    
    def _get_collaborative_scores(self, user_id: int) -> np.ndarray:
        user_idx = self.user_to_idx.get(user_id)
        if user_idx is None:
            return np.zeros(len(self.recipes_df), dtype=np.float32)
        
        user_ratings = self.user_recipe_matrix[user_idx, :]
        user_sim = cosine_similarity(user_ratings, self.user_recipe_matrix)
        cf_scores_matrix = user_sim.dot(self.user_recipe_matrix)
        cf_scores = cf_scores_matrix.toarray().flatten()
        
        final_scores = np.zeros(len(self.recipes_df), dtype=np.float32)
        for i, row in self.recipes_df.iterrows():
            rid = row['recipe_id']
            if rid in self.recipe_to_idx:
                r_idx = self.recipe_to_idx[rid]
                final_scores[i] = cf_scores[r_idx]
        
        max_cf = final_scores.max()
        if max_cf > 0:
            final_scores = final_scores / max_cf
        
        return final_scores
    
    def add_user_feedback(self, user_id: int, recipe_id: int, rating: float):
        self.feedback_history[user_id].append({
            'recipe_id': recipe_id,
            'rating': rating,
        })