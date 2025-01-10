import pandas as pd
from typing import Dict, List
import json
import os
from datetime import datetime

class FeedbackManager:
    def __init__(self, feedback_file: str = 'data/feedback.json'):
        self.feedback_file = feedback_file
        self.feedback_data = self._load_feedback()
    
    def _load_feedback(self) -> Dict:
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_feedback(self, user_id: int, recipe_id: int, 
                     rating: float, comments: str = None):
        if str(user_id) not in self.feedback_data:
            self.feedback_data[str(user_id)] = []
        
        feedback = {
            'recipe_id': int(recipe_id),
            'rating': float(rating),
            'timestamp': datetime.now().isoformat(),
            'comments': comments
        }
        
        self.feedback_data[str(user_id)].append(feedback)
        
        # Persist to file
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
    
    def get_user_feedback(self, user_id: int) -> List[Dict]:
        return self.feedback_data.get(str(user_id), [])
    
    def get_recipe_ratings(self, recipe_id: int) -> List[Dict]:
        ratings = []
        for user_key, user_feedback_list in self.feedback_data.items():
            for record in user_feedback_list:
                if record['recipe_id'] == recipe_id:
                    ratings.append(record)
        return ratings
    
    def load_all_feedback(self) -> Dict[str, List[Dict]]:
        """
        Returns the entire feedback data structure, e.g.:
        {
          "123": [
             { "recipe_id": 26067, "rating":2.0, "comments":"..." },
             ...
          ],
          ...
        }
        """
        return self.feedback_data