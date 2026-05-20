# Task B: Kalm4Rec for Nigerian-Focused Restaurant Recommendation

## Fit

Kalm4Rec is a good backbone for Task B because the challenge rewards contextual recommendation, cold-start behavior, and reasoning before ranking. Its original pipeline already does three useful things:

1. Extracts user and item keywords from review text.
2. Retrieves candidate restaurants from sparse user preference signals.
3. Uses an LLM reranker to reason over the user keywords and candidate item keywords.

Your notebook adds the missing hackathon angle: culturally grounded Nigerian diaspora personas across Yoruba, Igbo, and Hausa preference patterns.

This Task B setup uses only the Yelp restaurant subset from:

- Philadelphia
- Tampa
- Nashville

`tripAdvisor` is part of the original Kalm4Rec repository examples and is not part of this experiment or recommendation domain.

## Recommended Architecture

Use a two-stage recommender:

1. **Candidate retrieval**
   Run Kalm4Rec keyword retrieval over your Yelp restaurant subset. This gives a candidate pool from behavior signals instead of asking the LLM to search over 8,958 restaurants.

2. **Persona-aware reranking**
   Feed the reranker:
   - generated Nigerian persona blocks,
   - user keywords from history,
   - candidate restaurant keywords,
   - restaurant metadata such as city, cuisine, price, halal, rating, and review count.

3. **Cold-start path**
   For new users, skip historical collaborative signals and build query keywords directly from the persona, such as `peppery`, `halal`, `portion size`, `suya`, `jollof`, `family-style`, `value for money`, or `warm service`.

4. **Cross-domain/cross-city path**
   Treat Philadelphia, Tampa, and Nashville as separate domains. Train/retrieve from two cities and evaluate on held-out users or restaurants in the third city. In the paper, call this cross-city transfer unless you add another product domain.

## Data Adapter

After the notebook writes `user_review_history.json` and `restaurant_detail.csv`, convert them into Kalm4Rec format with temporal holdout labels:

```bash
python taskB/prepare_naija_yelp.py \
  --user_history /path/to/user_review_history.json \
  --restaurant_detail /home/dell/projects/Agent4Rec/restaurant_detail.csv \
  --dataset_name naija_yelp \
  --target_cities Philadelphia Tampa Nashville \
  --holdout_per_user 1 \
  --cold_start
```

This creates:

- `data/reviews/naija_yelp.csv` for history/profile reviews only
- `data/reviews/naija_yelp_holdout.csv` for next-item labels
- `data/reviews/naija_yelp_splits.json`
- `data/metadata/naija_yelp_restaurant_detail.csv`
- `data/reviews/naija_yelp_cold_start.csv` and `data/reviews/naija_yelp_cold_start_splits.json` when `--cold_start` is used

Then the normal Kalm4Rec stages can run with `--city naija_yelp`.

## Two Evaluation Protocols

There are two valid but different ways to evaluate this project:

1. **Temporal Task B protocol**
   This uses `naija_yelp.csv` as earlier user history and `naija_yelp_holdout.csv` as each user's next restaurant. It is stricter and produces lower absolute scores because each user has only one relevant held-out item.

2. **Kalm4Rec paper protocol**
   This uses a user-disjoint cold-start split. Test users have no reviews in the train graph, and all of their reviewed restaurants are treated as relevant labels. This is the protocol behind the higher scores in the Kalm4Rec paper.

To create the paper-style protocol files:

```bash
python taskB/prepare_naija_yelp.py \
  --user_history user_review_history.json \
  --restaurant_detail restaurant_detail.csv \
  --dataset_name naija_yelp \
  --target_cities Philadelphia Tampa Nashville \
  --holdout_per_user 1 \
  --cold_start \
  --paper_protocol_dataset naija_yelp_paper
```

This creates:

- `data/reviews/naija_yelp_paper.csv`
- `data/reviews/naija_yelp_paper_splits.json`
- `data/metadata/naija_yelp_paper_restaurant_detail.csv`

Run Kalm4Rec-paper-style retrieval with:

```bash
python extractor.py --city naija_yelp_paper --edgeType IUF --kwExtractor kw_NLTK
python retrieval.py \
  --city naija_yelp_paper \
  --edgeType IUF \
  --quantity 50 \
  --validTopK 20 \
  --export2LLMs
```

Do not pass `--groundtruth_file data/reviews/naija_yelp_holdout.csv` for the paper protocol. The ground truth is `data/reviews/naija_yelp_paper.csv`, matching the original Kalm4Rec setup.

## Hybrid Reranking

The LLM reranker improves the first ranks, but it can push some relevant MPG candidates lower in the list. To preserve graph evidence while keeping semantic reranking, combine both rankings:

```bash
python taskB/hybrid_rerank.py \
  --city naija_yelp_paper \
  --alpha 0.3
```

`alpha` controls the MPG weight:

- `alpha=1.0`: raw MPG order only
- `alpha=0.0`: LLM order only
- `alpha=0.3`: current NDCG@10-oriented blend

The main reranker also supports this directly:

```bash
python reRanker/rerank.py \
  --city naija_yelp_paper \
  --api_key "$GOOGLE_API_KEY" \
  --metadata_file data/metadata/naija_yelp_paper_restaurant_detail.csv \
  --rerank_mode scored \
  --candidate_pool_k 50 \
  --rerank_top_k 20 \
  --hybrid_alpha 0.3
```

Current paper-protocol results:

| Method | P@1 | Recall@10 | NDCG@10 | HitRate@10 | Recall@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPG retrieval | 0.4480 | 0.2413 | 0.6198 | 0.8597 | 0.3270 |
| Gemini reranker | 0.4802 | 0.2242 | 0.6170 | 0.8430 | 0.3270 |
| Hybrid, alpha=0.3 | 0.4969 | 0.2333 | 0.6297 | 0.8430 | 0.3270 |

The alpha choice depends on the target metric. `alpha=0.3` gives the best measured NDCG@10 among the tested blends, while `alpha=0.6` gives the best measured Recall@10/HitRate@10 balance. Neither should be tuned on the final test set in a formal paper; use a dev split or report it as an analysis.

## ColdRAG-Style KG Notes

ColdRAG uses LLM-generated item profiles, entity/relation extraction, vector indexing, and LLM-guided multi-hop traversal. This repository does not use an external ontology-style knowledge graph. It uses Kalm4Rec's review-derived keyword graph:

```text
user -> keyword -> restaurant
```

For this project, we added a lightweight ColdRAG-inspired `semantic_profile` field to restaurant metadata. It standardizes each restaurant into a concise profile using name, city, categories, cuisine family, halal, spice, value, service, family dining, and West African similarity signals.

Full LLM entity/relation graph extraction is not enabled by default because:

- It would require thousands of profile/extraction calls for 2,608 restaurants.
- LLM-scored edge traversal per user would be expensive for 962 test users.
- Our strongest measured improvement came from hybrid ranking, not from adding more generated text.
- The current keyword graph already reproduces Kalm4Rec-level paper results on `naija_yelp_paper`.

## Run Commands

Normal history-aware recommendation:

```bash
python taskB/evaluate_baselines.py \
  --history data/reviews/naija_yelp.csv \
  --holdout data/reviews/naija_yelp_holdout.csv \
  --k 10
python extractor.py --city naija_yelp --edgeType IUF --kwExtractor kw_NLTK
python retrieval.py \
  --city naija_yelp \
  --edgeType IUF \
  --quantity 50 \
  --export2LLMs \
  --groundtruth_file data/reviews/naija_yelp_holdout.csv
python reRanker/rerank.py \
  --city naija_yelp \
  --api_key "$GOOGLE_API_KEY" \
  --groundtruth_file data/reviews/naija_yelp_holdout.csv \
  --metadata_file data/metadata/naija_yelp_restaurant_detail.csv \
  --rerank_mode scored \
  --candidate_pool_k 50 \
  --rerank_top_k 20 \
  --hybrid_alpha 0.3
```

Cold-start recommendation, using profile/persona text instead of full history:

```bash
python extractor.py --city naija_yelp_cold_start --edgeType IUF --kwExtractor kw_NLTK
python retrieval.py \
  --city naija_yelp_cold_start \
  --edgeType IUF \
  --quantity 50 \
  --export2LLMs \
  --groundtruth_file data/reviews/naija_yelp_holdout.csv
python reRanker/rerank.py \
  --city naija_yelp_cold_start \
  --api_key "$GOOGLE_API_KEY" \
  --groundtruth_file data/reviews/naija_yelp_holdout.csv \
  --metadata_file data/metadata/naija_yelp_restaurant_detail.csv \
  --rerank_mode scored \
  --candidate_pool_k 50 \
  --rerank_top_k 20 \
  --hybrid_alpha 0.3
```

Cross-city transfer, for example holding out Tampa users:

```bash
python taskB/prepare_naija_yelp.py \
  --user_history /path/to/user_review_history.json \
  --restaurant_detail /home/dell/projects/Agent4Rec/restaurant_detail.csv \
  --dataset_name naija_yelp_tampa_transfer \
  --target_cities Philadelphia Tampa Nashville \
  --holdout_city Tampa \
  --holdout_per_user 1
```

Then run the same extractor/retrieval/reranker commands with `--city naija_yelp_tampa_transfer` and `--groundtruth_file data/reviews/naija_yelp_tampa_transfer_holdout.csv`.

Amazon cross-domain setup:

```bash
python taskB/prepare_amazon_reviews.py \
  --reviews /path/to/amazon_reviews.json.gz \
  --metadata /path/to/amazon_metadata.json.gz \
  --dataset_name amazonBaby

python extractor.py --city amazonBaby --edgeType IUF --kwExtractor kw_NLTK
python retrieval.py \
  --city amazonBaby \
  --edgeType IUF \
  --quantity 20 \
  --validTopK 20 \
  --export2LLMs
python reRanker/rerank.py \
  --city amazonBaby \
  --api_key "$GOOGLE_API_KEY" \
  --metadata_file data/metadata/amazonBaby_restaurant_detail.csv \
  --rerank_top_k 20
```

Use the same pattern for `amazonVideo`.

For Amazon data, `asin` or `parent_asin` becomes `rest_id`; each product is treated as an item. This lets the same Kalm4Rec keyword graph and LLM reranker run across Yelp restaurants and Amazon products. That is a cross-domain robustness/evaluation setup. It will not directly improve Yelp restaurant NDCG unless you add a supervised transfer model or use Amazon as few-shot prompt evidence, because the item space and user histories are different.

## Important Notes

- Do not evaluate cold-start using keywords extracted from the same reviews you are trying to predict. This adapter writes earlier history/profile inputs separately from later holdout labels.
- Restaurant metadata signals are now enriched from history/profile reviews for temporal evaluation, and from train users for paper-protocol user-disjoint evaluation. This avoids using held-out labels as metadata evidence.
- The notebook currently contains API keys. Rotate those keys and move secrets to environment variables before submitting the repository.
- Few-shot prompt templates in `reRanker/prompts.yaml` still need a Nigerian restaurant variant if you use `1_shot`, `2_shots`, or `3_shots`; the `zeroshot` path now has a `naija_yelp` restaurant branch.
