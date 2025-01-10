import pandas as pd
import numpy as np
import ast
import random  # <-- We use random.choice to assign cuisines
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from typing import Tuple, Dict, List
from tqdm import tqdm

class RecipeDataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.mlb = MultiLabelBinarizer()
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.scaler = StandardScaler()
        
        # Expanded dietary patterns for vegetarian, etc.
        self.dietary_patterns = {
            'vegetarian': [
                'egg', 'eggs', 'egg white', 'beef', 'veal', 'pork', 'bacon', 'ham', 'pepperoni', 'salami',
                'prosciutto', 'sausage', 'mutton', 'lamb', 'goat', 'venison',
                'bison', 'buffalo', 'corned beef', 'spam', 'baloney', 'bologna',
                'chorizo', 'kielbasa', 'haggis', 'blood sausage', 'black pudding',
                'chicken', 'turkey', 'duck', 'goose', 'quail', 'pheasant',
                'fish', 'cod', 'haddock', 'salmon', 'tuna', 'anchovy', 'anchovies',
                'sardine', 'sardines', 'shellfish', 'shrimp', 'prawn', 'lobster',
                'crab', 'crawfish', 'crayfish', 'squid', 'calamari', 'octopus',
                'clams', 'mussels', 'oysters', 'scallops',
                'gelatin', 'lard', 'broth', 'stock',
                'worcestershire sauce', 'fish sauce', 'oyster sauce', 'shrimp paste',
                'rabbit', 'horse', 'alligator', 'crocodile', 'frog', 'frog legs',
                'snails', 'escargot', 'chicken', 'turkey', 'duck', 'goose', 'quail', 'pheasant',
                'fish', 'cod', 'haddock', 'salmon', 'tuna', 'anchovy', 'anchovies',
                'sardine', 'sardines', 'shellfish', 'shrimp', 'prawn', 'lobster',
                'crab', 'crawfish', 'crayfish', 'squid', 'calamari', 'octopus',
                'clams', 'mussels', 'oysters', 'scallops',
                'gelatin', 'lard', 'broth', 'stock',
                'worcestershire sauce', 'fish sauce', 'oyster sauce', 'shrimp paste',
                'rabbit', 'horse', 'alligator', 'crocodile', 'frog', 'frog legs',
                'snails', 'escargot', 'steak'

            ],
            'vegan': [
                'beef', 'veal', 'pork', 'bacon', 'ham', 'pepperoni', 'salami',
                'prosciutto', 'sausage', 'mutton', 'lamb', 'goat', 'venison',
                'bison', 'buffalo', 'corned beef', 'spam', 'baloney', 'bologna',
                'chorizo', 'kielbasa', 'haggis', 'blood sausage', 'black pudding',
                'chicken', 'turkey', 'duck', 'goose', 'quail', 'pheasant',
                'fish', 'cod', 'haddock', 'salmon', 'tuna', 'anchovy', 'anchovies',
                'sardine', 'sardines', 'shellfish', 'shrimp', 'prawn', 'lobster',
                'crab', 'crawfish', 'crayfish', 'squid', 'calamari', 'octopus',
                'clams', 'mussels', 'oysters', 'scallops',
                'gelatin', 'lard', 'broth', 'stock',
                'worcestershire sauce', 'fish sauce', 'oyster sauce', 'shrimp paste',
                'rabbit', 'horse', 'alligator', 'crocodile', 'frog', 'frog legs',
                'snails', 'escargot',
                'milk', 'cheese', 'cream', 'yogurt', 'butter', 'ghee',
                'custard', 'casein', 'whey', 'lactose', 'milk powder',
                'egg', 'eggs', 'egg whites', 'mayonnaise',
                'honey', 'beeswax', 'royal jelly',
                'shellac', 'carmine', 'cochineal', 'isinglass'
            ],
            'gluten_free': [
                'wheat', 
                'flour', 
                'all-purpose flour',
                'bread', 
                'pasta',
                'cracker',
                'breadcrumbs',
                'cake flour',
                'self-rising flour',
                'whole wheat',
                'wheat germ',
                'wheat bran',
                'spelt',
                'semolina',
                'durum',
                'rye',
                'barley',
                'malt',
                'farro',
                'triticale',
                'oats'
            ],
            'dairy_free': [
                'milk',
                'milk powder',
                'skim milk',
                'evaporated milk',
                'condensed milk',
                'cheese',
                'cream',
                'heavy cream',
                'light cream',
                'whipped cream',
                'sour cream',
                'cream cheese',
                'yogurt',
                'butter',
                'ghee',
                'buttermilk',
                'custard',
                'casein',
                'caseinate',
                'whey',
                'lactose',
                'kefir',
                'paneer',
                'ricotta',
                'mascarpone'
            ]
        }


    def load_and_preprocess_data(
            self, recipes_path: str, ratings_path: str = None, sample_df: pd.DataFrame = None
        ) -> Tuple[pd.DataFrame, np.ndarray]:
            """
            Load and preprocess recipe + ratings data.
            """
            try:
                if sample_df is not None:
                    print(f"Using provided sample of {len(sample_df)} recipes")
                    df = sample_df.copy()
                else:
                    print(f"Loading recipes from {recipes_path}")
                    df = pd.read_csv(recipes_path, encoding='utf-8')

                df = self._clean_data(df)  # Clean data (including random 'cuisine' assignment)

                # If steps are stored in 'steps', parse them into lists
                if 'steps' in df.columns:
                    df['steps_list'] = df['steps'].apply(self._safe_parse_list)
                
                # Process ingredients and create dietary flags
                df = self._process_ingredients(df)
                
                # Process nutrition if it exists
                if 'nutrition' in df.columns:
                    df = self._process_nutrition(df)
                
                # Calculate difficulty
                df = self._calculate_difficulty(df)
                
                # Create TF-IDF embeddings
                recipe_embeddings = self._create_recipe_embeddings(df)
                
                # Merge with ratings if provided
                if ratings_path:
                    print(f"Loading ratings from {ratings_path}")
                    ratings_df = pd.read_csv(ratings_path)
                    df = self._add_ratings_features(df, ratings_df)
                
                return df, recipe_embeddings
            
            except Exception as e:
                print(f"Error during data preprocessing: {str(e)}")
                if 'df' in locals():
                    print("DataFrame info after rename:")
                    print(df.info())
                raise

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning and random cuisine assignment.
        """
        print("\nCleaning data...")
        original_len = len(df)
        
        # Drop rows missing both ingredients and steps
        columns_to_check = ['ingredients', 'steps']
        existing_cols = [col for col in columns_to_check if col in df.columns]
        df = df.dropna(subset=existing_cols, how='all')
        print(f"Dropped {original_len - len(df)} rows with missing data in 'ingredients' or 'steps'")
        
        # Basic cleaning: lower-case recipe names
        if 'name' in df.columns:
            df['name'] = df['name'].astype(str).str.lower()
        
        # === Adding the cuisine column randomly ===
        # We pick from 5 possible cuisines
        CUISINE_TYPES = ['Indian', 'Italian', 'Mexican', 'Chinese', 'American']
        if 'cuisine' not in df.columns:
            df['cuisine'] = [random.choice(CUISINE_TYPES) for _ in range(len(df))]
        
        return df
    
    def _safe_parse_list(self, val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except:
                return []
        elif isinstance(val, list):
            return val
        else:
            return []

    def _process_ingredients(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert string -> list
        if 'ingredients' not in df.columns:
            return df
        
        def parse_ingredient_list(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except:
                    return []
            elif isinstance(x, list):
                return x
            else:
                return []
        
        df['ingredients_list'] = df['ingredients'].apply(parse_ingredient_list)
        
        # Create dietary restriction flags
        for diet, excluded_items in self.dietary_patterns.items():
            df[f'is_{diet}'] = ~df['ingredients_list'].apply(
                # lambda ing_list: any(ex_item in " ".join(ing_list).lower() for ex_item in excluded_items)
                lambda ing_list: any(
                                    any(ex_item in ingredient.lower() for ex_item in excluded_items)
                                    for ingredient in ing_list)
            )
        df['is_non_vegetarian'] = ~df['is_vegetarian']
        
        return df

    def _process_nutrition(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\nProcessing nutrition column...")
        
        def parse_nutrition_list(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)  # e.g. '[51.5, 0.0, 13.0, ...]'
                except:
                    return [0,0,0,0,0,0,0]
            elif isinstance(x, list):
                return x
            else:
                return [0,0,0,0,0,0,0]
        
        df['nutrition_list'] = df['nutrition'].apply(parse_nutrition_list)
        
        # Format: [calories, total_fat, sugar, sodium, protein, sat_fat, carbs]
        nut_cols = ['calories','total_fat','sugar','sodium','protein','saturated_fat','carbohydrates']
        df[nut_cols] = pd.DataFrame(df['nutrition_list'].tolist(), index=df.index)
        
        df.drop(columns=['nutrition','nutrition_list'], inplace=True)
        
        return df

    def _calculate_difficulty(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\nCalculating recipe difficulty...")
        df['difficulty'] = 0.0
        
        # #ingredients
        if 'n_ingredients' in df.columns:
            df['difficulty'] += df['n_ingredients'] / 10.0
        elif 'ingredients_list' in df.columns:
            df['difficulty'] += df['ingredients_list'].apply(len) / 10.0
        
        # cooking time
        if 'cook_time_minutes' in df.columns:
            df['difficulty'] += df['cook_time_minutes'].clip(0,180)/60.0
        if 'minutes' in df.columns:
            df['difficulty'] += df['minutes'].clip(0,180)/60.0
        
        # #steps
        if 'n_steps' in df.columns:
            df['difficulty'] += df['n_steps']/5.0
        elif 'steps' in df.columns:
            df['difficulty'] += df['steps'].apply(lambda s: len(re.split(r'[.!?]', str(s)))) / 5.0
        
        df['difficulty'] = df['difficulty'].clip(1,5)
        print("Difficulty calculation done.")
        return df

    def _create_recipe_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        print("\nCreating text embeddings...")
        
        if 'ingredients_list' not in df.columns or 'steps' not in df.columns:
            # minimal fallback
            text_data = df['name'].fillna("").astype(str).tolist()
        else:
            text_data = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing text data"):
                ing_str = " ".join(row['ingredients_list']) if isinstance(row['ingredients_list'], list) else ""
                step_str = str(row.get('steps', ''))
                combined = f"{ing_str} {step_str}"
                text_data.append(combined)
        
        print("Computing TF-IDF matrix...")
        recipe_embeddings = self.tfidf.fit_transform(text_data)
        print("TF-IDF embedding created.")
        
        return recipe_embeddings

    def _add_ratings_features(self, recipes_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        print("\nMerging ratings data with recipes...")
        if 'recipe_id' not in recipes_df.columns:
            return recipes_df
        
        if 'recipe_id' not in ratings_df.columns or 'rating' not in ratings_df.columns:
            print("Ratings file missing 'recipe_id' or 'rating'. Cannot merge.")
            return recipes_df
        
        rating_stats = ratings_df.groupby('recipe_id').agg({
            'rating': ['mean','count','std']
        }).reset_index()
        rating_stats.columns = ['recipe_id','avg_rating','rating_count','rating_std']
        
        recipes_df = recipes_df.merge(rating_stats, on='recipe_id', how='left')
        
        recipes_df['avg_rating'] = recipes_df['avg_rating'].fillna(0.0)
        recipes_df['rating_count'] = recipes_df['rating_count'].fillna(0)
        recipes_df['rating_std'] = recipes_df['rating_std'].fillna(0.0)
        
        return recipes_df

class FoodComDataPreprocessor(RecipeDataPreprocessor):
    """
    Child class for the Food.com dataset specifically,
    with custom column mappings.
    """
    def __init__(self):
        super().__init__()
        self.column_mapping = {
            'id': 'recipe_id',
            'minutes': 'cook_time_minutes',
            'description': 'description',
            'n_steps': 'n_steps',
            'n_ingredients': 'n_ingredients',
            'contributor_id': 'user_id'
        }
        self.tfidf = TfidfVectorizer(stop_words='english')

    def load_and_preprocess_data(
        self, recipes_path: str, ratings_path: str = None, sample_df: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load & preprocess with custom Food.com logic (renaming columns, etc.).
        """
        try:
            if sample_df is not None:
                print(f"Using provided sample of {len(sample_df)} recipes")
                df = sample_df.copy()
            else:
                print(f"Loading recipes from {recipes_path}")
                df = pd.read_csv(recipes_path)
            
            # Rename columns
            df = df.rename(columns=self.column_mapping)
            
            # Now call parent for the rest
            df, recipe_embeddings = super().load_and_preprocess_data(
                recipes_path=recipes_path, 
                ratings_path=ratings_path, 
                sample_df=df
            )
            
            return df, recipe_embeddings
        
        except Exception as e:
            print(f"Error during data preprocessing: {str(e)}")
            if 'df' in locals():
                print("DataFrame info after rename:")
                print(df.info())
            raise