import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "reRanker"))

from utils import prepare_user2rests, quick_eval


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_candidate_mapping(candidate_data, map_item_id2int):
    next_id = max(map_item_id2int.values(), default=-1) + 1
    for user_data in candidate_data.values():
        for item_id in user_data.get("candidate", []):
            if item_id not in map_item_id2int:
                map_item_id2int[item_id] = next_id
                next_id += 1
    return map_item_id2int


def raw_rankings(candidate_data, map_item_id2int, top_k):
    return {
        user_id: [map_item_id2int[item_id] for item_id in user_data["candidate"][:top_k]]
        for user_id, user_data in candidate_data.items()
    }


def combine_rankings(raw_rank, llm_rank, alpha, top_k):
    """Interpolate MPG retrieval order and LLM order.

    alpha=1.0 gives raw MPG. alpha=0.0 gives LLM-only. Middle values preserve
    graph evidence while allowing LLM semantic preferences to move items.
    """
    items = []
    for item in raw_rank + llm_rank:
        if item not in items:
            items.append(item)

    norm = max(len(items), top_k, 1)
    scores = {}
    for idx, item in enumerate(raw_rank):
        scores[item] = scores.get(item, 0.0) + alpha * (norm - idx) / norm
    for idx, item in enumerate(llm_rank):
        scores[item] = scores.get(item, 0.0) + (1.0 - alpha) * (norm - idx) / norm

    return sorted(items, key=lambda item: scores.get(item, 0.0), reverse=True)[:top_k]


def evaluate(rankings, groundtruth, k):
    prec, rec, f1, ndcg, hit = [], [], [], [], []
    for user_id, rank in rankings.items():
        p, r, f, n = quick_eval(rank[:k], groundtruth[user_id])
        relevant = {item for item, _ in groundtruth[user_id]}
        prec.append(p)
        rec.append(r)
        f1.append(f)
        ndcg.append(n)
        hit.append(float(bool(relevant.intersection(rank[:k]))))
    return {
        f"precision@{k}": float(np.mean(prec)),
        f"recall@{k}": float(np.mean(rec)),
        f"f1@{k}": float(np.mean(f1)),
        f"ndcg@{k}": float(np.mean(ndcg)),
        f"hitrate@{k}": float(np.mean(hit)),
    }


def main():
    parser = argparse.ArgumentParser(description="Hybrid MPG + LLM reranker for Task B.")
    parser.add_argument("--city", default="naija_yelp_paper")
    parser.add_argument("--candidates", help="Kalm4Rec candidate json")
    parser.add_argument("--llm_rank", help="LLM reranker json")
    parser.add_argument("--groundtruth", help="Ground-truth review csv")
    parser.add_argument("--alpha", type=float, default=0.3, help="Weight for MPG order; 1.0=MPG only, 0.0=LLM only")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--eval_k", type=int, default=10)
    parser.add_argument("--output", help="Where to save hybrid rankings")
    args = parser.parse_args()

    candidate_path = args.candidates or f"data/out2LLMs/{args.city}_knn2rest.json"
    llm_path = args.llm_rank or f"reRanker/results_rerank/{args.city}/zeroshot_3_5_12.json"
    groundtruth_path = args.groundtruth or f"data/reviews/{args.city}.csv"
    output_path = args.output or f"reRanker/results_rerank/{args.city}/hybrid_alpha_{args.alpha:g}_top{args.top_k}.json"

    candidate_data = read_json(candidate_path)
    llm_rank = {user_id: [int(item) for item in rank] for user_id, rank in read_json(llm_path).items()}
    _, groundtruth, map_item_id2int = prepare_user2rests(groundtruth_path, is_tripAdvisor=False)
    map_item_id2int = ensure_candidate_mapping(candidate_data, map_item_id2int)

    raw_rank = raw_rankings(candidate_data, map_item_id2int, args.top_k)
    hybrid = {
        user_id: combine_rankings(raw_rank[user_id], llm_rank[user_id], args.alpha, args.top_k)
        for user_id in raw_rank
        if user_id in llm_rank
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(hybrid, f)

    print(f"Wrote hybrid rankings: {output_path}")
    for k in [1, 3, 5, 10, 20]:
        if k <= args.top_k:
            print(evaluate(hybrid, groundtruth, k))


if __name__ == "__main__":
    main()
