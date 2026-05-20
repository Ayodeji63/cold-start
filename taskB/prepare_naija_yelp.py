import argparse
import json
import random
from pathlib import Path

import pandas as pd


TARGET_CITIES = ["Philadelphia", "Tampa", "Nashville"]


SIGNALS = {
    "spice_signal": [
        "pepper",
        "peppery",
        "spicy",
        "hot",
        "chilli",
        "chili",
        "suya",
        "jerk",
        "cajun",
    ],
    "value_signal": [
        "portion",
        "portions",
        "large",
        "small",
        "price",
        "cheap",
        "expensive",
        "value",
        "worth",
    ],
    "service_signal": [
        "service",
        "staff",
        "friendly",
        "rude",
        "warm",
        "hospitality",
        "waiter",
        "waitress",
    ],
    "family_style_signal": [
        "family",
        "group",
        "sharing",
        "communal",
        "platter",
        "ambience",
        "atmosphere",
    ],
}

WEST_AFRICAN_TERMS = [
    "african",
    "west african",
    "nigerian",
    "ghanaian",
    "ethiopian",
    "caribbean",
    "jollof",
    "suya",
    "egusi",
    "fufu",
    "plantain",
]

CUISINE_MAP = {
    "west_african": ["african", "nigerian", "ghanaian", "jollof", "suya"],
    "caribbean": ["caribbean", "jamaican", "haitian"],
    "middle_eastern": ["middle eastern", "turkish", "mediterranean", "halal", "kebab"],
    "asian": ["chinese", "japanese", "thai", "korean", "vietnamese", "sushi", "ramen"],
    "american": ["american", "burgers", "barbeque", "bbq", "fast food", "chicken wings"],
    "latin": ["mexican", "latin", "tacos", "spanish"],
    "italian": ["italian", "pizza", "pasta"],
}


def load_user_history(path):
    with open(path, "r") as f:
        return json.load(f)


def flatten_reviews(user_history):
    rows = []
    for user_id, reviews in user_history.items():
        for idx, review in enumerate(reviews):
            business_id = review["business_id"]
            rows.append(
                {
                    "review_id": review.get("review_id", f"{user_id}_{business_id}_{idx}"),
                    "user_id": user_id,
                    "city": review.get("city", "naija_yelp"),
                    "rest_id": business_id,
                    "date": review.get("date", ""),
                    "rating": review.get("stars_review", review.get("rating")),
                    "text": review.get("text", ""),
                }
            )
    return pd.DataFrame(rows)


def filter_target_cities(review_df, target_cities):
    if not target_cities:
        return review_df
    return review_df[review_df["city"].isin(target_cities)].copy()


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).lower()


def contains_any(text, terms):
    return any(term in text for term in terms)


def infer_cuisine_family(categories):
    text = normalize_text(categories)
    for family, terms in CUISINE_MAP.items():
        if contains_any(text, terms):
            return family
    return "other"


def build_semantic_profile(row):
    name = str(row.get("name", row.get("business_id", "restaurant")))
    city = str(row.get("city", "")).strip()
    categories = str(row.get("categories", "")).strip()
    cuisine_family = row.get("cuisine_family", "other")
    price = row.get("price", "?")
    halal = row.get("halal", False)

    signals = []
    if row.get("west_african_similarity", False):
        signals.append("West African or African-diaspora similarity")
    if row.get("spice_signal", False):
        signals.append("spice or pepper cues")
    if row.get("value_signal", False):
        signals.append("portion/value cues")
    if row.get("service_signal", False):
        signals.append("service and hospitality cues")
    if row.get("family_style_signal", False):
        signals.append("family or group dining cues")
    if bool(halal):
        signals.append("halal signal")

    signal_text = ", ".join(signals) if signals else "general dining cues"
    return (
        f"{name} is a {cuisine_family} restaurant in {city}. "
        f"Categories: {categories}. Price tier: {price}. "
        f"Notable recommendation signals: {signal_text}."
    )


def infer_preference_terms(text, term_groups):
    normalized = normalize_text(text)
    return [
        signal_name.replace("_signal", "").replace("_", " ")
        for signal_name, terms in term_groups.items()
        if contains_any(normalized, terms)
    ]


def infer_cuisine_preferences(text):
    normalized = normalize_text(text)
    return [
        family.replace("_", " ")
        for family, terms in CUISINE_MAP.items()
        if contains_any(normalized, terms)
    ]


def build_cold_start_preference_text(profile_text):
    cuisines = infer_cuisine_preferences(profile_text)
    signals = infer_preference_terms(profile_text, SIGNALS)
    normalized = normalize_text(profile_text)
    if contains_any(normalized, WEST_AFRICAN_TERMS):
        signals.append("west african similarity")
    if "halal" in normalized:
        signals.append("halal")

    cuisine_text = ", ".join(cuisines) if cuisines else "unknown cuisine"
    signal_text = ", ".join(dict.fromkeys(signals)) if signals else "general dining"
    return (
        "Cold-start preference profile. "
        f"Likely cuisine preferences: {cuisine_text}. "
        f"Likely dining signals: {signal_text}. "
        f"Evidence text: {profile_text}"
    )


def enrich_restaurant_metadata(restaurant_detail, review_df):
    detail = restaurant_detail.copy()
    if "business_id" not in detail.columns and detail.index.name == "business_id":
        detail = detail.reset_index()

    if "business_id" not in detail.columns:
        return detail

    if "categories" not in detail.columns:
        detail["categories"] = ""

    categories = detail["categories"].map(normalize_text)
    detail["cuisine_family"] = detail["categories"].map(infer_cuisine_family)
    detail["west_african_similarity"] = categories.map(lambda x: contains_any(x, WEST_AFRICAN_TERMS))

    if "halal" not in detail.columns:
        detail["halal"] = categories.map(lambda x: "halal" in x)

    review_text = (
        review_df.groupby("rest_id")["text"]
        .apply(lambda values: " ".join(map(str, values)).lower())
        .to_dict()
    )
    for signal_name, terms in SIGNALS.items():
        detail[signal_name] = detail["business_id"].map(
            lambda business_id: contains_any(review_text.get(business_id, ""), terms)
        )

    detail["semantic_profile"] = detail.apply(build_semantic_profile, axis=1)

    return detail


def temporal_history_holdout(review_df, holdout_per_user):
    df = review_df.copy()
    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["user_id", "_date", "review_id"], na_position="first")

    history_parts = []
    holdout_parts = []
    for _, group in df.groupby("user_id", sort=False):
        n_holdout = min(holdout_per_user, max(len(group) - 1, 0))
        if n_holdout == 0:
            history_parts.append(group)
            continue
        history_parts.append(group.iloc[:-n_holdout])
        holdout_parts.append(group.iloc[-n_holdout:])

    history_df = pd.concat(history_parts, ignore_index=True).drop(columns=["_date"])
    if holdout_parts:
        holdout_df = pd.concat(holdout_parts, ignore_index=True).drop(columns=["_date"])
    else:
        holdout_df = df.iloc[0:0].drop(columns=["_date"])
    return history_df, holdout_df


def split_users(review_df, train_ratio, dev_ratio, seed):
    rng = random.Random(seed)
    users = sorted(review_df["user_id"].dropna().unique().tolist())
    rng.shuffle(users)

    train_end = int(len(users) * train_ratio)
    dev_end = train_end + int(len(users) * dev_ratio)

    return {
        "train": users[:train_end],
        "dev": users[train_end:dev_end],
        "test": users[dev_end:],
    }


def build_cross_city_splits(review_df, holdout_city, train_ratio, dev_ratio, seed):
    if not holdout_city:
        return None
    city_col = review_df["city"].fillna("")
    test_users = sorted(review_df.loc[city_col == holdout_city, "user_id"].unique().tolist())
    train_dev_users = sorted(review_df.loc[city_col != holdout_city, "user_id"].unique().tolist())
    train_dev_users = [user_id for user_id in train_dev_users if user_id not in set(test_users)]

    rng = random.Random(seed)
    rng.shuffle(train_dev_users)
    dev_count = int(len(train_dev_users) * dev_ratio / max(train_ratio + dev_ratio, 1e-9))
    return {
        "train": train_dev_users[dev_count:],
        "dev": train_dev_users[:dev_count],
        "test": test_users,
    }


def build_cold_start_profiles(history_df, personas_path=None, max_profile_reviews=2):
    rows = []
    persona_lookup = {}
    if personas_path:
        path = Path(personas_path)
        if path.is_file():
            with open(path, "r") as f:
                persona_lookup = json.load(f)
        elif path.is_dir():
            for persona_file in path.glob("persona_*.txt"):
                user_key = persona_file.stem.replace("persona_", "")
                persona_lookup[user_key] = persona_file.read_text()

    for user_id, group in history_df.groupby("user_id"):
        persona_text = persona_lookup.get(user_id, "")
        sample_reviews = group.sort_values("date").head(max_profile_reviews)
        snippets = " ".join(sample_reviews["text"].astype(str).tolist())
        profile_text = f"{persona_text} {snippets}".strip()
        if not profile_text:
            profile_text = snippets
        profile_text = build_cold_start_preference_text(profile_text)
        first = sample_reviews.iloc[0]
        rows.append(
            {
                "review_id": f"cold_start_profile_{user_id}",
                "user_id": user_id,
                "city": first.get("city", "naija_yelp"),
                "rest_id": first["rest_id"],
                "date": first.get("date", ""),
                "rating": first.get("rating", 5),
                "text": profile_text,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(history_df, holdout_df, splits, restaurant_detail, out_dir, dataset_name, cold_start_df=None):
    reviews_dir = out_dir / "reviews"
    meta_dir = out_dir / "metadata"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    review_path = reviews_dir / f"{dataset_name}.csv"
    holdout_path = reviews_dir / f"{dataset_name}_holdout.csv"
    split_path = reviews_dir / f"{dataset_name}_splits.json"
    meta_path = meta_dir / f"{dataset_name}_restaurant_detail.csv"
    cold_start_dataset = f"{dataset_name}_cold_start"
    cold_start_path = reviews_dir / f"{cold_start_dataset}.csv"
    cold_start_split_path = reviews_dir / f"{cold_start_dataset}_splits.json"

    history_df.to_csv(review_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)
    with open(split_path, "w") as f:
        json.dump({dataset_name: splits}, f, indent=2)

    if restaurant_detail is not None:
        restaurant_detail.to_csv(meta_path, index=False)
    if cold_start_df is not None:
        cold_start_df.to_csv(cold_start_path, index=False)
        with open(cold_start_split_path, "w") as f:
            json.dump({cold_start_dataset: splits}, f, indent=2)

    return {
        "history": review_path,
        "holdout": holdout_path,
        "splits": split_path,
        "metadata": meta_path if restaurant_detail is not None else None,
        "cold_start": cold_start_path if cold_start_df is not None else None,
        "cold_start_splits": cold_start_split_path if cold_start_df is not None else None,
    }


def write_paper_protocol_outputs(review_df, splits, restaurant_detail, out_dir, dataset_name):
    reviews_dir = out_dir / "reviews"
    meta_dir = out_dir / "metadata"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    review_path = reviews_dir / f"{dataset_name}.csv"
    split_path = reviews_dir / f"{dataset_name}_splits.json"
    meta_path = meta_dir / f"{dataset_name}_restaurant_detail.csv"

    review_df.to_csv(review_path, index=False)
    with open(split_path, "w") as f:
        json.dump({dataset_name: splits}, f, indent=2)
    if restaurant_detail is not None:
        restaurant_detail.to_csv(meta_path, index=False)

    return {
        "reviews": review_path,
        "splits": split_path,
        "metadata": meta_path if restaurant_detail is not None else None,
    }


def reviews_for_users(review_df, users):
    user_set = set(users)
    return review_df[review_df["user_id"].isin(user_set)].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Convert Nigerian/Yelp notebook outputs into Kalm4Rec Task B files."
    )
    parser.add_argument("--user_history", required=True, help="Path to user_review_history.json")
    parser.add_argument("--restaurant_detail", help="Path to restaurant_detail.csv")
    parser.add_argument("--dataset_name", default="naija_yelp")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--min_reviews", type=int, default=5)
    parser.add_argument("--holdout_per_user", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--holdout_city", help="Optional city name for cross-city testing")
    parser.add_argument("--target_cities", nargs="+", default=TARGET_CITIES, help="Cities to keep from the Yelp notebook output")
    parser.add_argument("--personas", help="Optional persona JSON file or persona_*.txt directory")
    parser.add_argument("--cold_start", action="store_true", help="Write a profile-only cold-start input file")
    parser.add_argument(
        "--paper_protocol_dataset",
        help="Also write a Kalm4Rec-paper-style user-disjoint dataset with all test-user reviews as relevance labels",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    user_history = load_user_history(args.user_history)
    review_df = flatten_reviews(user_history)
    review_df = filter_target_cities(review_df, args.target_cities)
    review_df = review_df.dropna(subset=["user_id", "rest_id", "rating", "text"])

    if args.holdout_city and args.holdout_city not in args.target_cities:
        raise ValueError(f"--holdout_city must be one of {args.target_cities}")

    counts = review_df.groupby("user_id").size()
    keep_users = counts[counts >= args.min_reviews].index
    review_df = review_df[review_df["user_id"].isin(keep_users)].copy()

    history_df, holdout_df = temporal_history_holdout(review_df, args.holdout_per_user)

    splits = build_cross_city_splits(history_df, args.holdout_city, args.train_ratio, args.dev_ratio, args.seed)
    if splits is None:
        splits = split_users(history_df, args.train_ratio, args.dev_ratio, args.seed)

    restaurant_detail = None
    raw_restaurant_detail = None
    if args.restaurant_detail:
        raw_restaurant_detail = pd.read_csv(args.restaurant_detail)
        if "city" in raw_restaurant_detail.columns:
            raw_restaurant_detail = raw_restaurant_detail[raw_restaurant_detail["city"].isin(args.target_cities)].copy()
        restaurant_detail = enrich_restaurant_metadata(raw_restaurant_detail, history_df)

    cold_start_df = None
    if args.cold_start:
        cold_start_df = build_cold_start_profiles(history_df, args.personas)

    paths = write_outputs(
        history_df=history_df,
        holdout_df=holdout_df,
        splits=splits,
        restaurant_detail=restaurant_detail,
        out_dir=Path(args.out_dir),
        dataset_name=args.dataset_name,
        cold_start_df=cold_start_df,
    )

    print(f"Users: {review_df['user_id'].nunique():,}")
    print(f"Target cities: {', '.join(args.target_cities)}")
    print(f"Reviews: {len(review_df):,}")
    print(f"History reviews: {len(history_df):,}")
    print(f"Holdout reviews: {len(holdout_df):,}")
    print(f"Restaurants: {review_df['rest_id'].nunique():,}")
    print(f"Train/dev/test users: {len(splits['train']):,}/{len(splits['dev']):,}/{len(splits['test']):,}")
    print(f"Wrote history reviews: {paths['history']}")
    print(f"Wrote holdout labels: {paths['holdout']}")
    print(f"Wrote splits: {paths['splits']}")
    if paths["metadata"] is not None:
        print(f"Wrote metadata: {paths['metadata']}")
    if paths["cold_start"] is not None:
        print(f"Wrote cold-start profiles: {paths['cold_start']}")
        print(f"Wrote cold-start splits: {paths['cold_start_splits']}")

    if args.paper_protocol_dataset:
        paper_splits = build_cross_city_splits(review_df, args.holdout_city, args.train_ratio, args.dev_ratio, args.seed)
        if paper_splits is None:
            paper_splits = split_users(review_df, args.train_ratio, args.dev_ratio, args.seed)
        paper_restaurant_detail = None
        if raw_restaurant_detail is not None:
            paper_train_df = reviews_for_users(review_df, paper_splits["train"])
            paper_restaurant_detail = enrich_restaurant_metadata(raw_restaurant_detail, paper_train_df)
        paper_paths = write_paper_protocol_outputs(
            review_df=review_df,
            splits=paper_splits,
            restaurant_detail=paper_restaurant_detail,
            out_dir=Path(args.out_dir),
            dataset_name=args.paper_protocol_dataset,
        )
        print("Paper protocol: user-disjoint cold-start split, no temporal holdout.")
        print(f"Wrote paper-protocol reviews/labels: {paper_paths['reviews']}")
        print(f"Wrote paper-protocol splits: {paper_paths['splits']}")
        if paper_paths["metadata"] is not None:
            print(f"Wrote paper-protocol metadata: {paper_paths['metadata']}")


if __name__ == "__main__":
    main()
