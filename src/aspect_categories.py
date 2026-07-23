"""
aspect_categories.py
---------------------
Maps the ~1000 raw, free-text aspect terms in SemEval-Restaurants
(e.g. "wine list", "waiter", "prices", "ambiance") down to a small set
of business-relevant categories, so the "insights" layer of the
project can answer questions like:

    "Which category drives the most negative sentiment?"

instead of drowning in hundreds of raw noun phrases. This mirrors what
a real Data Analyst does: taxonomy design + rollups, on top of a raw
NLP signal.

The keyword lists were built by inspecting the most frequent aspect
terms in the training data (see notebooks/01_eda.ipynb) and are
intentionally simple/transparent rather than a black box, which keeps
this step explainable for a report/viva.
"""

CATEGORY_KEYWORDS = {
    "Food Quality": [
        "food", "dish", "meal", "menu", "taste", "flavor", "flavour",
        "sushi", "pizza", "sauce", "meat", "fish", "chicken", "steak",
        "dessert", "appetizer", "portion", "ingredient", "spicy",
        "cuisine", "curry", "noodle", "rice", "bread", "cook",
        "lunch", "dinner", "brunch", "entree", "salad", "burger",
        "soup", "sandwich", "seafood", "pasta", "crust", "cake",
        "selection",
    ],
    "Service": [
        "service", "staff", "waiter", "waitress", "server", "host",
        "hostess", "manager", "attitude", "attentive", "rude",
        "friendly", "wait time", "reservation", "bartender", "owner",
        "serve", "waitstaff", "waiting",
    ],
    "Price/Value": [
        "price", "cost", "value", "expensive", "cheap", "bill",
        "money", "worth", "overpriced", "affordable",
    ],
    "Ambience": [
        "ambiance", "ambience", "atmosphere", "decor", "interior",
        "music", "noise", "seating", "space", "view", "lighting",
        "vibe", "crowd", "romantic", "place", "dining experience",
        "decoration",
    ],
    "Drinks": [
        "wine", "cocktail", "beer", "drink", "beverage", "bar",
        "margarita", "sake", "coffee", "tea",
    ],
    "Location/Wait": [
        "location", "parking", "wait", "line", "queue", "crowded",
        "delivery", "takeout", "speed",
    ],
}


def categorize_aspect(aspect: str) -> str:
    """Return the best-matching business category for a raw aspect term.
    Uses simple substring matching (not strict word boundaries) so that
    plural/inflected forms like 'prices', 'waiters', 'drinks' still match
    their singular keyword ('price', 'waiter', 'drink'). Falls back to
    'Other' if no keyword matches.
    """
    a = aspect.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in a:
                return category
    return "Other"


def add_category_column(df, aspect_col: str = "aspect", new_col: str = "aspect_category"):
    df = df.copy()
    df[new_col] = df[aspect_col].apply(categorize_aspect)
    return df
