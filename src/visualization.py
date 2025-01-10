from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cuisine_distribution(df: pd.DataFrame, title: str = "Cuisine Distribution"):
    """
    Plots a bar chart showing how many recipes are labeled 
    for each cuisine in the DataFrame.
    """
    if 'cuisine' not in df.columns:
        print("No 'cuisine' column in dataframe. Cannot plot cuisine distribution.")
        return
    
    cuisine_counts = df['cuisine'].value_counts()
    
    plt.figure(figsize=(8,6))
    sns.barplot(x=cuisine_counts.index, y=cuisine_counts.values, palette="viridis")
    plt.title(title)
    plt.xlabel("Cuisine")
    plt.ylabel("Count of Recipes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_top_recommendations(recommended_df: pd.DataFrame, top_n: int = 5):
    """
    Plots or prints a simple bar chart of the top N recommended recipes
    by some measure (e.g., final_scores or rating).
    You can adapt this to show rating or final scoring if stored.
    """
    if len(recommended_df) == 0:
        print("No recommended recipes to plot.")
        return
    
    # If you have a 'final_score' column or similar, you can plot that.
    # For now, let's assume we just plot 'avg_rating' or a placeholder.
    
    # If 'avg_rating' doesn't exist, let's create a dummy column to plot.
    if 'avg_rating' not in recommended_df.columns:
        recommended_df['avg_rating'] = 0
    
    top_recipes = recommended_df.head(top_n)
    
    plt.figure(figsize=(8,5))
    sns.barplot(
        x=top_recipes['name'],
        y=top_recipes['avg_rating'],
        palette="rocket"
    )
    plt.title(f"Top {top_n} Recommended Recipes (by avg_rating as example)")
    plt.xlabel("Recipe Name")
    plt.ylabel("Average Rating")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

def show_recommendations_cuisine_breakdown(recommended_df: pd.DataFrame):
    """
    Example function to visualize how many recommended recipes
    fall into each cuisine category.
    """
    if 'cuisine' not in recommended_df.columns:
        print("No 'cuisine' column in recommended_df.")
        return
    
    rec_counts = recommended_df['cuisine'].value_counts()
    
    plt.figure(figsize=(6,6))
    plt.pie(
        rec_counts.values,
        labels=rec_counts.index,
        autopct="%1.1f%%",
        startangle=140
    )
    plt.title("Recommended Recipes by Cuisine")
    plt.tight_layout()
    plt.show()

def plot_difficulty_distribution(df: pd.DataFrame, title: str = "Difficulty Distribution in the Recommendations"):
    """
    Plots a histogram showing how many recipes fall into each difficulty level (1..5).
    """
    if 'difficulty' not in df.columns:
        print("No 'difficulty' column in dataframe. Cannot plot difficulty distribution.")
        return
    
    plt.figure(figsize=(7,5))
    # Because difficulty is presumably a float 1..5, we can bin them
    sns.histplot(df['difficulty'], bins=[1,2,3,4,5,6], kde=False, color="teal")
    plt.title(title)
    plt.xlabel("Difficulty (1 = easiest, 5 = hardest)")
    plt.ylabel("Number of Recipes")
    plt.tight_layout()
    plt.show()



def plot_rating_distribution(df: pd.DataFrame, rating_col='avg_rating', title: str = "Ratings Distribution of the Dataset"):
    """
    Plots a histogram of average ratings among recipes, if available.
    """
    if rating_col not in df.columns:
        print(f"No '{rating_col}' column found. Cannot plot rating distribution.")
        return
    
    plt.figure(figsize=(7,5))
    sns.histplot(df[rating_col], bins=10, kde=False, color="purple")
    plt.title(title)
    plt.xlabel("Average Rating")
    plt.ylabel("Number of Recipes")
    plt.tight_layout()
    plt.show()


def plot_diet_flags_breakdown_dataset(df: pd.DataFrame, flags: List[str] = None):
    """
    Plots a bar chart showing how many recipes pass each diet flag 
    (e.g., 'is_vegetarian', 'is_vegan', 'is_gluten_free').
    """
    if flags is None:
        flags = ['is_vegetarian', 'is_vegan', 'is_gluten_free', 'is_dairy_free', 'is_non_vegetarian']
    
    # Collect counts
    diet_counts = {}
    for flag in flags:
        if flag in df.columns:
            count_true = df[flag].sum()  # True is 1, so sum() = number of True
            diet_counts[flag] = count_true
        else:
            diet_counts[flag] = 0  # or skip if not present
    
    # Convert to DF for easy plotting
    diet_df = pd.DataFrame(list(diet_counts.items()), columns=['Diet Flag', 'Count'])
    
    plt.figure(figsize=(6,4))
    sns.barplot(x='Diet Flag', y='Count', data=diet_df, palette="Set2")
    plt.title("Dietary Flags Breakdown in the Sample Dataset")
    plt.xlabel("Dietary Flag")
    plt.ylabel("Number of Recipes Passing This Flag")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_diet_flags_breakdown_recommendations(df: pd.DataFrame, flags: List[str] = None):
    """
    Plots a bar chart showing how many recipes pass each diet flag 
    (e.g., 'is_vegetarian', 'is_vegan', 'is_gluten_free').
    """
    if flags is None:
        flags = ['is_vegetarian', 'is_vegan', 'is_gluten_free', 'is_dairy_free']
    
    # Collect counts
    diet_counts = {}
    for flag in flags:
        if flag in df.columns:
            count_true = df[flag].sum()  # True is 1, so sum() = number of True
            diet_counts[flag] = count_true
        else:
            diet_counts[flag] = 0  # or skip if not present
    
    # Convert to DF for easy plotting
    diet_df = pd.DataFrame(list(diet_counts.items()), columns=['Diet Flag', 'Count'])
    
    plt.figure(figsize=(6,4))
    sns.barplot(x='Diet Flag', y='Count', data=diet_df, palette="Set2")
    plt.title("Dietary Flags Breakdown in the Recommendations")
    plt.xlabel("Dietary Flag")
    plt.ylabel("Number of Recipes Passing This Flag")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def plot_correlation_matrix(df: pd.DataFrame, numeric_cols: List[str] = None, title="Correlation Matrix"):
    """
    Plots a correlation heatmap for specified numeric columns, explicitly excluding 'cook_time_minutes'.
    """
    if numeric_cols is None:
        numeric_cols = ['avg_rating', 'difficulty', 'calories', 'protein', 'fat']
    excluded_col = 'cook_time_minutes'
    valid_cols = [
        col for col in numeric_cols
        if col in df.columns and col != excluded_col
    ]
    
    if not valid_cols:
        print("No valid numeric columns to correlate after excluding 'cook_time_minutes'.")
        return
    
    corr_df = df[valid_cols].corr()
    
    plt.figure(figsize=(6,5))
    sns.heatmap(corr_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()