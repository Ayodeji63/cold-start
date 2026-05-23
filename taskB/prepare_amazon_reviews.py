import argparse
import gzip
import json
import random
from pathlib import Path

import pandas as pd


REVIEW_COLUMN_ALIASES = {
    "review_id": ["review_id", "reviewID", "id"],
    "user_id": ["user_id", "reviewerID", "reviewer_id", "user"],
    "rest_id": ["rest_id", "asin", "parent_asin", "item_id", "product_id"],
    "rating": ["rating", "overall", "stars"],
    "text": ["full_text", "text", "reviewText", "review_text", "body"],
    "date": ["date", "reviewTime", "timestamp", "unixReviewTime"],
}

METADATA_COLUMN_ALIASES = {
    "rest_id": ["rest_id", "business_id", "asin", "parent_asin", "item_id", "product_id"],
    "title": ["title", "name", "product_title"],
    "categories": ["categories", "category", "main_category"],
    "brand": ["brand"],
    "description": ["description", "details", "features"],
}


def read_table(path):
    path = Path(path)
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".json.gz"):
        rows = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            with open(path, "r") as f:
                data = json.load(f)
            return pd.DataFrame(data)
    return pd.read_csv(path)


def first_present(df, aliases):
    for col in aliases:
        if col in df.columns:
            return col
    return None


def normalize_date(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)) and value > 10_000:
        return pd.to_datetime(value, unit="s", errors="coerce").strftime("%Y-%m-%d")
    return str(value)


def normalize_reviews(raw_df, dataset_name):
    source_cols = {
        target: first_present(raw_df, aliases)
        for target, aliases in REVIEW_COLUMN_ALIASES.items()
    }
    missing = [target for target in ["user_id", "rest_id", "rating", "text"] if source_cols[target] is None]
    if missing:
        raise ValueError(f"Missing required Amazon review columns: {missing}. Available columns: {list(raw_df.columns)}")

    df = pd.DataFrame()
    df["user_id"] = raw_df[source_cols["user_id"]].astype(str)
    df["rest_id"] = raw_df[source_cols["rest_id"]].astype(str)
    df["rating"] = pd.to_numeric(raw_df[source_cols["rating"]], errors="coerce")
    df["text"] = raw_df[source_cols["text"]].fillna("").astype(str)
    if source_cols["date"]:
        df["date"] = raw_df[source_cols["date"]].map(normalize_date)
    else:
        df["date"] = ""

    if source_cols["review_id"]:
        df["review_id"] = raw_df[source_cols["review_id"]].astype(str)
    else:
        df["review_id"] = [
            f"{dataset_name}_{user_id}_{item_id}_{idx}"
            for idx, (user_id, item_id) in enumerate(zip(df["user_id"], df["rest_id"]))
        ]

    df["city"] = dataset_name
    df = df[["review_id", "user_id", "city", "rest_id", "date", "rating", "text"]]
    df = df.dropna(subset=["user_id", "rest_id", "rating", "text"])
    df = df[df["text"].str.strip().astype(bool)]
    return df


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


def filter_by_category(raw_df, category_contains):
    if not category_contains:
        return raw_df
    category_col = first_present(raw_df, METADATA_COLUMN_ALIASES["categories"])
    title_col = first_present(raw_df, METADATA_COLUMN_ALIASES["title"])
    if category_col is None and title_col is None:
        raise ValueError("--category_contains was provided, but no category/title column exists.")

    text = pd.Series("", index=raw_df.index, dtype="object")
    if category_col is not None:
        text = text + " " + raw_df[category_col].fillna("").astype(str)
    if title_col is not None:
        text = text + " " + raw_df[title_col].fillna("").astype(str)

    terms = [term.lower() for term in category_contains]
    mask = text.str.lower().map(lambda value: any(term in value for term in terms))
    return raw_df[mask].copy()


def filter_reviews_by_metadata_category(raw_reviews, raw_meta, category_contains):
    if not category_contains:
        return raw_reviews

    review_item_col = first_present(raw_reviews, REVIEW_COLUMN_ALIASES["rest_id"])
    meta_item_col = first_present(raw_meta, METADATA_COLUMN_ALIASES["rest_id"])
    category_col = first_present(raw_meta, METADATA_COLUMN_ALIASES["categories"])
    title_col = first_present(raw_meta, METADATA_COLUMN_ALIASES["title"])

    if review_item_col is None or meta_item_col is None:
        raise ValueError("--category_contains needs item ids in both reviews and metadata.")
    if category_col is None and title_col is None:
        raise ValueError("--category_contains was provided, but no category/title column exists in reviews or metadata.")

    text = pd.Series("", index=raw_meta.index, dtype="object")
    if category_col is not None:
        text = text + " " + raw_meta[category_col].fillna("").astype(str)
    if title_col is not None:
        text = text + " " + raw_meta[title_col].fillna("").astype(str)

    terms = [term.lower() for term in category_contains]
    keep_mask = text.str.lower().map(lambda value: any(term in value for term in terms))
    keep_items = set(raw_meta.loc[keep_mask, meta_item_col].astype(str))
    return raw_reviews[raw_reviews[review_item_col].astype(str).isin(keep_items)].copy()


def filter_dense_items(review_df, min_item_reviews):
    if min_item_reviews <= 1:
        return review_df
    item_counts = review_df.groupby("rest_id").size()
    keep_items = item_counts[item_counts >= min_item_reviews].index
    return review_df[review_df["rest_id"].isin(keep_items)].copy()


def filter_top_items(review_df, max_items):
    if not max_items:
        return review_df
    top_items = review_df["rest_id"].value_counts().head(max_items).index
    return review_df[review_df["rest_id"].isin(top_items)].copy()


def filter_dense_users(review_df, min_reviews):
    if min_reviews <= 1:
        return review_df
    user_counts = review_df.groupby("user_id").size()
    keep_users = user_counts[user_counts >= min_reviews].index
    return review_df[review_df["user_id"].isin(keep_users)].copy()


def filter_k_core(review_df, min_user_reviews, min_item_reviews):
    if min_user_reviews <= 1 and min_item_reviews <= 1:
        return review_df

    filtered = review_df.copy()
    while True:
        before = len(filtered)
        if min_item_reviews > 1:
            item_counts = filtered.groupby("rest_id").size()
            keep_items = item_counts[item_counts >= min_item_reviews].index
            filtered = filtered[filtered["rest_id"].isin(keep_items)].copy()
        if min_user_reviews > 1:
            user_counts = filtered.groupby("user_id").size()
            keep_users = user_counts[user_counts >= min_user_reviews].index
            filtered = filtered[filtered["user_id"].isin(keep_users)].copy()
        if len(filtered) == before:
            return filtered


def normalize_metadata(raw_meta, review_df, dataset_name):
    source_cols = {
        target: first_present(raw_meta, aliases)
        for target, aliases in METADATA_COLUMN_ALIASES.items()
    }
    if source_cols["rest_id"] is None:
        raise ValueError(f"Missing product id column in metadata. Available columns: {list(raw_meta.columns)}")

    raw_meta = raw_meta.drop_duplicates(subset=[source_cols["rest_id"]]).copy()

    meta = pd.DataFrame()
    meta["business_id"] = raw_meta[source_cols["rest_id"]].astype(str)
    meta["city"] = dataset_name
    meta["name"] = raw_meta[source_cols["title"]].fillna("").astype(str) if source_cols["title"] else meta["business_id"]
    meta["categories"] = raw_meta[source_cols["categories"]].fillna("").astype(str) if source_cols["categories"] else ""
    meta["brand"] = raw_meta[source_cols["brand"]].fillna("").astype(str) if source_cols["brand"] else ""
    meta["description"] = raw_meta[source_cols["description"]].fillna("").astype(str) if source_cols["description"] else ""

    review_text = (
        review_df.groupby("rest_id")["text"]
        .apply(lambda values: " ".join(map(str, values.head(5))))
        .to_dict()
    )
    meta["semantic_profile"] = meta.apply(
        lambda row: (
            f"{row['name']} is an Amazon product in {dataset_name}. "
            f"Brand: {row['brand']}. Categories: {row['categories']}. "
            f"Description: {row['description']}. "
            f"Representative reviews: {review_text.get(row['business_id'], '')[:800]}"
        ),
        axis=1,
    )
    return meta[meta["business_id"].isin(set(review_df["rest_id"]))]


def write_outputs(review_df, splits, metadata_df, out_dir, dataset_name):
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
    if metadata_df is not None:
        metadata_df.to_csv(meta_path, index=False)

    return review_path, split_path, meta_path if metadata_df is not None else None


def main():
    parser = argparse.ArgumentParser(description="Convert Amazon reviews into Kalm4Rec-compatible cross-domain files.")
    parser.add_argument("--reviews", required=True, help="Amazon reviews file: csv, json, jsonl, or json.gz")
    parser.add_argument("--metadata", help="Optional Amazon product metadata file")
    parser.add_argument("--dataset_name", required=True, help="Example: amazonBaby or amazonVideo")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--min_reviews", type=int, default=5)
    parser.add_argument("--min_item_reviews", type=int, default=1, help="Keep only items with at least this many reviews")
    parser.add_argument("--max_items", type=int, help="Keep only the most-reviewed N items after other filters")
    parser.add_argument("--category_contains", nargs="+", help="Keep rows whose category/title contains any of these terms")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_reviews = read_table(args.reviews)
    raw_meta = read_table(args.metadata) if args.metadata else None
    if args.category_contains:
        try:
            raw_reviews = filter_by_category(raw_reviews, args.category_contains)
        except ValueError:
            if raw_meta is None:
                raise
            raw_reviews = filter_reviews_by_metadata_category(raw_reviews, raw_meta, args.category_contains)
    review_df = normalize_reviews(raw_reviews, args.dataset_name)
    review_df = filter_top_items(review_df, args.max_items)
    review_df = filter_k_core(review_df, args.min_reviews, args.min_item_reviews)

    if review_df.empty:
        raise ValueError("No reviews remain after filtering. Relax --min_reviews, --min_item_reviews, --max_items, or --category_contains.")

    splits = split_users(review_df, args.train_ratio, args.dev_ratio, args.seed)

    metadata_df = None
    if raw_meta is not None:
        metadata_df = normalize_metadata(raw_meta, review_df, args.dataset_name)

    review_path, split_path, meta_path = write_outputs(
        review_df=review_df,
        splits=splits,
        metadata_df=metadata_df,
        out_dir=Path(args.out_dir),
        dataset_name=args.dataset_name,
    )

    print(f"Dataset: {args.dataset_name}")
    print(f"Users: {review_df['user_id'].nunique():,}")
    print(f"Reviews: {len(review_df):,}")
    print(f"Items: {review_df['rest_id'].nunique():,}")
    print(f"Train/dev/test users: {len(splits['train']):,}/{len(splits['dev']):,}/{len(splits['test']):,}")
    print(f"Wrote reviews/labels: {review_path}")
    print(f"Wrote splits: {split_path}")
    if meta_path is not None:
        print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
