# nlp_preference_extraction.py

from typing import Dict, List
import re

class NLPPreferenceExtractor:
    """
    A simple rule-based approach to parse textual reviews/comments
    for dietary or nutritional cues.
    """

    def __init__(self):
        # Example sets of keywords
        self.calorie_keywords = {"calorie", "calories", "low-calorie", "fat"}
        self.healthy_keywords = {"nutrient", "vitamin", "healthy", "low fat", "low-fat", "protein"}
        # You can define more sets for "spicy", "sweet", "taste", etc.

    def extract_preferences(self, comments: str) -> Dict[str, bool]:
        """
        Given a single user comment, return booleans indicating certain preference flags.
        For instance:
          {
            "concerned_about_calories": True,
            "healthy_focus": False
          }
        You can expand with more complex logic or multiple flags.
        """
        text = comments.lower()

        found_calorie = any(kw in text for kw in self.calorie_keywords)
        found_healthy = any(kw in text for kw in self.healthy_keywords)

        return {
            "concerned_about_calories": found_calorie,
            "healthy_focus": found_healthy
        }

    def aggregate_preferences(self, user_reviews: List[Dict]) -> Dict[str, bool]:
        """
        Given all reviews for a user, unify the flags across them.
        E.g., if they mention 'calorie' in ANY comment, we set concerned_about_calories = True.
        """
        # Start all flags as False
        agg_prefs = {
            "concerned_about_calories": False,
            "healthy_focus": False
        }

        for review in user_reviews:
            # 'comments' might be the key for textual review
            comment = review.get("comments", "")
            extracted = self.extract_preferences(comment)

            # If user mentions it in ANY comment, we keep it True
            if extracted["concerned_about_calories"]:
                agg_prefs["concerned_about_calories"] = True

            if extracted["healthy_focus"]:
                agg_prefs["healthy_focus"] = True
        
        return agg_prefs