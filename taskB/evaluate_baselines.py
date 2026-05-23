import argparse
import json
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def ndcg_at_k(ranked_items, relevant_items, k):
    ranked_items = ranked_items[:k]
    dcg = 0.0
    for idx, item in enumerate(ranked_items):
        if item in relevant_items:
            dcg += 1.0 / np.log2(idx + 2)
    ideal_hits = min(len(relevant_items), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg


def hit_rate_at_k(ranked_items, relevant_items, k):
    return float(bool(set(ranked_items[:k]).intersection(relevant_items)))


def build_groundtruth(holdout_df):
    truth = defaultdict(set)
    for user_id, rest_id in zip(holdout_df["user_id"], holdout_df["rest_id"]):
        truth[user_id].add(rest_id)
    return truth


def popular_rankings(history_df):
    counts = Counter(history_df["rest_id"])
    return [rest_id for rest_id, _ in counts.most_common()]


def filter_holdout_by_split(holdout_df, splits_file, dataset_name, split_name):
    if not splits_file:
        return holdout_df
    with open(splits_file, "r") as f:
        splits = json.load(f)
    if dataset_name is None:
        dataset_name = next(iter(splits))
    users = set(splits[dataset_name][split_name])
    return holdout_df[holdout_df["user_id"].isin(users)].copy()


def filter_history_by_split(history_df, splits_file, dataset_name, split_name):
    if not splits_file or split_name is None:
        return history_df
    with open(splits_file, "r") as f:
        splits = json.load(f)
    if dataset_name is None:
        dataset_name = next(iter(splits))
    users = set(splits[dataset_name][split_name])
    return history_df[history_df["user_id"].isin(users)].copy()


def evaluate(truth, rankings_by_user, k):
    ndcgs = []
    hits = []
    for user_id, relevant_items in truth.items():
        ranked_items = rankings_by_user[user_id]
        ndcgs.append(ndcg_at_k(ranked_items, relevant_items, k))
        hits.append(hit_rate_at_k(ranked_items, relevant_items, k))
    return float(np.mean(ndcgs)), float(np.mean(hits))


def main():
    parser = argparse.ArgumentParser(description="Task B popularity/random baseline evaluator.")
    parser.add_argument("--history", required=True, help="History/profile csv")
    parser.add_argument("--holdout", required=True, help="Temporal holdout csv")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", help="Optional splits json for train/dev/test filtering")
    parser.add_argument("--dataset_name", help="Dataset key inside --splits")
    parser.add_argument("--split", default="test", help="Split name to evaluate when --splits is provided")
    parser.add_argument("--history_split", help="Optional split name to use for building popularity/random item universe")
    args = parser.parse_args()

    history_df = pd.read_csv(args.history)
    holdout_df = pd.read_csv(args.holdout)
    history_df = filter_history_by_split(history_df, args.splits, args.dataset_name, args.history_split)
    holdout_df = filter_holdout_by_split(holdout_df, args.splits, args.dataset_name, args.split)
    truth = build_groundtruth(holdout_df)

    all_items = sorted(history_df["rest_id"].dropna().unique().tolist())
    popular_items = popular_rankings(history_df)
    rng = random.Random(args.seed)

    popular_by_user = {user_id: popular_items for user_id in truth}
    random_by_user = {}
    for user_id in truth:
        items = all_items[:]
        rng.shuffle(items)
        random_by_user[user_id] = items

    pop_ndcg, pop_hit = evaluate(truth, popular_by_user, args.k)
    rand_ndcg, rand_hit = evaluate(truth, random_by_user, args.k)

    print(f"Users: {len(truth):,}")
    print(f"Items: {len(all_items):,}")
    print(f"Popularity NDCG@{args.k}: {pop_ndcg:.6f}")
    print(f"Popularity HitRate@{args.k}: {pop_hit:.6f}")
    print(f"Random NDCG@{args.k}: {rand_ndcg:.6f}")
    print(f"Random HitRate@{args.k}: {rand_hit:.6f}")


if __name__ == "__main__":
    main()
