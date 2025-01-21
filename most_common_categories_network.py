import os
import json
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
#### ZOHAR

# Louvain algorithm for community detection
# pip install python-louvain
import community as community_louvain

##############################################################################
# CONFIGURATIONS
##############################################################################

# Set this to True if "filtered files" are already created and
# you want to skip re-processing the original files.
SKIP_PHASE_1 = False

# Paths
ORIGINAL_FILES_DIR = "wikipedia_data"
FILTERED_FILES_DIR = "filtered_wikipedia_data"

# Minimum word count in text
MIN_WORDS = 300

# Words to exclude in categories (case-insensitive)
EXCLUDE_WORDS = [
    "births",
    "deaths",
    "living",
    "(living",
    "unknown",
    "missing",
    "uncertain",
    "articles containing video clips",
    "taxonomy articles created by polbot",
]

# Number of top categories
TOP_N = 50

# Output plot filename
NETWORK_PLOT_FILE = "category_network.png"

# Output CSV
TOP_CATEGORIES_CSV = "category_louvain_network_cluster.csv"


##############################################################################
# HELPER FUNCTIONS
##############################################################################


def extract_text_and_categories(article_text):
    """
    Given the full article text, split off the 'Category:' portion if present.
    Returns:
      (cleaned_text, categories_list)
    """
    # Find the first occurrence of "Category:"
    cat_index = article_text.find("Category:")
    if cat_index == -1:
        # No 'Category:' found
        return article_text, []

    # Text up to (but not including) 'Category:'
    cleaned_text = article_text[:cat_index]

    # The chunk that starts from the first 'Category:'
    categories_text = article_text[cat_index:]

    # Split by "Category:" to isolate category strings
    split_categories = categories_text.split("Category:")

    # Extract non-empty stripped items
    cat_list = [c.strip() for c in split_categories if c.strip()]

    return cleaned_text, cat_list


def has_enough_words(text, min_words=MIN_WORDS):
    """Return True if text has at least min_words."""
    return len(text.split()) >= min_words


def should_exclude(category, exclude_words=EXCLUDE_WORDS):
    """
    Return True if the category contains any of the exclude words
    (case-insensitive match).
    """
    cat_lower = category.lower()
    return any(word in cat_lower for word in exclude_words)


def load_json_file(filepath):
    """Utility to load a JSON file and return the data."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data, filepath):
    """Utility to save data to a JSON file with UTF-8 encoding."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


##############################################################################
# PHASE 1: PROCESS & FILTER ARTICLES
##############################################################################

if not SKIP_PHASE_1:
    print("=== Phase 1: Processing original files and creating filtered files ===")

    # Counters to track total articles
    total_articles_before = 0
    total_articles_after = 0

    # Ensure the "filtered files" directory exists
    if not os.path.exists(FILTERED_FILES_DIR):
        os.makedirs(FILTERED_FILES_DIR)

    # Process each JSON in "original files"
    for filename in os.listdir(ORIGINAL_FILES_DIR):
        if not filename.endswith(".json"):
            continue  # skip non-JSON files

        original_path = os.path.join(ORIGINAL_FILES_DIR, filename)
        data = load_json_file(original_path)

        # Filtered articles list for this file
        filtered_articles = []

        # Count how many articles are in this file
        file_article_count = len(data)
        total_articles_before += file_article_count

        for article in data:
            article_id = article.get("id", "")
            article_title = article.get("title", "")
            article_text = article.get("text", "")

            # 1) Extract the portion of text before "Category:" and the categories list
            cleaned_text, categories_list = extract_text_and_categories(article_text)

            # 2) Check word count
            if not has_enough_words(cleaned_text, MIN_WORDS):
                continue

            # 3) Check if categories is empty
            if not categories_list:
                continue

            # 4) Construct new article object
            new_article = {
                "id": article_id,
                "title": article_title,
                "text": cleaned_text,
                "categories": categories_list,
            }

            filtered_articles.append(new_article)

        # Update total articles after filtering
        total_articles_after += len(filtered_articles)

        # Save filtered data to "filtered files" with the same filename
        filtered_path = os.path.join(FILTERED_FILES_DIR, filename)
        save_json_file(filtered_articles, filtered_path)

        # Clear from memory
        del data
        del filtered_articles

    print(f"Total articles before filtering (all files): {total_articles_before}")
    print(f"Total articles after filtering (all files):  {total_articles_after}")
    print("=== Phase 1 complete ===\n")
else:
    print("=== Phase 1 skipped. Using existing filtered files. ===\n")

##############################################################################
# PHASE 2: CATEGORY ANALYSIS & NETWORK GRAPH
##############################################################################

print("=== Phase 2: Analyzing categories from filtered files ===")

# 1) Gather and count all categories from "filtered files"
category_counter = Counter()

# Temporary storage of articles' categories (for co-occurrence)
articles_categories = []

for filename in os.listdir(FILTERED_FILES_DIR):
    if not filename.endswith(".json"):
        continue

    filtered_path = os.path.join(FILTERED_FILES_DIR, filename)
    data = load_json_file(filtered_path)

    for article in data:
        cats = article.get("categories", [])
        # Accumulate for global counting
        category_counter.update(cats)
        # Keep track for co-occurrence analysis
        articles_categories.append(cats)

    # Clear memory for this file
    del data


def is_valid_category(cat):
    """Exclude categories containing certain substrings."""
    return not should_exclude(cat, EXCLUDE_WORDS)


# Build a sorted list of categories by frequency, excluding undesired ones
sorted_categories = [
    (cat, cnt)
    for cat, cnt in category_counter.most_common()
    if is_valid_category(cat)
]

# 3) Take the top 50
top_categories = [cat for cat, cnt in sorted_categories[:TOP_N]]

print(f"Found {len(top_categories)} top categories (after exclusion).")

# 4) Build co-occurrence network
co_occurrences = {}  # dict of ((cat1, cat2) -> weight)

for cats in articles_categories:
    # Keep only categories that are in top_categories
    relevant_cats = [c for c in cats if c in top_categories]
    # Sort them so we can consistently form (catA, catB) as a key
    relevant_cats = sorted(set(relevant_cats))

    # Count pairwise co-occurrences
    for i in range(len(relevant_cats)):
        for j in range(i + 1, len(relevant_cats)):
            pair = (relevant_cats[i], relevant_cats[j])
            co_occurrences[pair] = co_occurrences.get(pair, 0) + 1

# Create an undirected graph
G = nx.Graph()

# Add nodes (the top categories)
for cat in top_categories:
    G.add_node(cat)

# Add edges based on co-occurrence weights
for (cat1, cat2), weight in co_occurrences.items():
    G.add_edge(cat1, cat2, weight=weight)

# 5) Cluster detection using Louvain
print("Running Louvain community detection...")
partition = community_louvain.best_partition(G)

# partition is a dict: {node: community_id}, e.g. {'CategoryA': 0, 'CategoryB': 1, ...}
num_communities = len(set(partition.values()))
print(f"Louvain found {num_communities} communities.")

# 6) Plot the network with different colors per cluster
plt.figure(figsize=(20, 20))

# We can compute a layout
pos = nx.spring_layout(G, k=0.6, iterations=50)

# To color nodes by their community, map each community to a color
community_ids = list(partition.values())
unique_communities = sorted(set(community_ids))

# Build a color map (can adjust or use a palette)
import matplotlib.cm as cm

colors = cm.get_cmap("tab20", len(unique_communities))

# Node sizes based on frequency
node_sizes = []
node_colors = []

for node in G.nodes():
    freq = category_counter[node]
    node_sizes.append(freq)
    comm_id = partition[node]
    color = colors(comm_id)
    node_colors.append(color)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=[(size / 100) for size in node_sizes],
    node_color=node_colors,
    alpha=0.9,
)

# Draw edges
edge_widths = []
max_weight = max(co_occurrences.values()) if co_occurrences else 1
for (u, v, w) in G.edges(data=True):
    edge_widths.append(0.1 + 0.9 * (w["weight"] / max_weight))

nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)

# Labels
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title(
    "Co-occurrence Network of Top 50 Categories (Louvain Clustering)", fontsize=16
)
plt.axis("off")
plt.tight_layout()

# Save plot
plt.savefig(NETWORK_PLOT_FILE, dpi=300, bbox_inches="tight")
plt.close()
print(f"Network plot saved as '{NETWORK_PLOT_FILE}'.")

# 7) Save the top 50 categories and their counts + cluster to CSV
top_categories_df = pd.DataFrame(top_categories, columns=["Category"])
top_categories_df["Count"] = top_categories_df["Category"].map(category_counter)
top_categories_df["Cluster"] = top_categories_df["Category"].map(partition)
top_categories_df.to_csv(TOP_CATEGORIES_CSV, index=False)
print(
    f"CSV with top 50 categories, counts, and cluster saved as '{TOP_CATEGORIES_CSV}'."
)

print("=== Phase 2 complete ===")
