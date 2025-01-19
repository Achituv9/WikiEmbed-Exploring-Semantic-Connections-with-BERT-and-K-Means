import json
import os
import pandas as pd

# Function to extract categories from a string
def extract_categories(category_list, valid_categories):
    """
    Extract categories from a list of category strings, filtering
    by the valid categories in the DataFrame column.
    
    Parameters:
    - category_list: List of category strings from the JSON data
    - valid_categories: Set of valid categories to filter against

    Returns:
    - A list of categories that are present in the valid_categories set
    """
    categories = []
    for category_str in category_list:
        # Split and clean category strings
        extracted = [cat.strip() for cat in category_str.split('Category:') if cat.strip()]
        # Filter categories against the valid set
        categories.extend([cat for cat in extracted if cat in valid_categories])
    return categories

# Load valid categories from 'category_louvain_network_cluster.csv'
valid_categories_df = pd.read_csv('category_louvain_network_cluster.csv')
valid_categories = set(valid_categories_df['Category'])  # Assuming 'Category' is the column name

# Process JSON files and create category DataFrame
category_data = []
json_dir = 'filtered_wikipedia_data'

for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(json_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for article in data:
                # Extract and filter categories
                categories = extract_categories(article.get('categories', []), valid_categories)
                for category in categories:
                    category_data.append({'ID': article['id'], 'Category': category})

# Convert to DataFrame
category_df = pd.DataFrame(category_data)

# Save to CSV
output_file = 'id_category_data.csv'
category_df.to_csv(output_file, index=False)
print(f"Category data saved to '{output_file}'")

