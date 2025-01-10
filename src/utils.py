def validate_user_preferences(preferences):
    """Validate user preference input."""
    required_fields = ['cuisine_preference', 'dietary_restrictions', 'available_time']
    
    for field in required_fields:
        if field not in preferences:
            raise ValueError(f"Missing required field: {field}")
    
    if preferences['available_time'] <= 0:
        raise ValueError("Available time must be positive")

def format_recipe_output(recipe):
    """Format recipe details for the Food.com dataset."""
    cook_time_val = recipe.get('cook_time_minutes', recipe.get('minutes', 'N/A'))
    
    # Steps
    steps_info = recipe.get('steps_list', [])
    if isinstance(steps_info, list):
        steps_text = " | ".join(steps_info)
    else:
        steps_text = str(steps_info)
    
    return {
        'name': recipe.get('name', 'unknown'),
        'cook_time': f"{cook_time_val} minutes",
        'ingredients': recipe.get('ingredients_list', []),
        'instructions': steps_text,
        'nutrition': {
            'calories': recipe.get('calories', 0.0),
            'protein': recipe.get('protein', 0.0),
            'fat': recipe.get('total_fat', 0.0),
            'carbohydrates': recipe.get('carbohydrates', 0.0)
        },
        'difficulty': f"{recipe.get('difficulty', 1)} (1=easiest, 5=hardest)"
    }