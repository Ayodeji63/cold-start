# Chapter 3 and 4 README: Kalm4Rec for Cold-Start and Cross-Domain Recommendation

This README summarizes the methodology and experimental findings for the adapted Kalm4Rec system. It is written to support Chapter 3 (methodology) and Chapter 4 (results and discussion).

## Chapter 3: Methodology

### Research Goal

The goal is to adapt Kalm4Rec for cold-start and cross-domain recommendation, using a Nigerian-focused restaurant recommendation task as the main domain and Amazon reviews as an additional product-domain evaluation setting.

The main restaurant dataset is a Yelp-derived subset focused on three cities:

- Philadelphia
- Tampa
- Nashville

Although the internal dataset name is `naija_yelp`, this does not mean the restaurants are located in Nigeria. The name refers to the Nigerian-focused recommendation framing: the system uses restaurant reviews, metadata, and culturally relevant dining signals to support Nigerian diaspora food preferences.

TripAdvisor is not part of this experiment. It only exists in the original Kalm4Rec codebase as a legacy example.

### Dataset Summary

The Yelp restaurant data prepared from the notebook contains:

| Component | Count |
| --- | ---: |
| Users | 9,614 |
| Reviews | 101,708 |
| Restaurants | about 2,600 after filtering |
| Target cities | Philadelphia, Tampa, Nashville |

The Amazon Grocery cross-domain dataset was also prepared:

| Component | Count |
| --- | ---: |
| Users | 14,176 |
| Reviews | 114,306 |
| Items | 48,297 |
| Train/dev/test users | 11,340 / 1,417 / 1,419 |

A density-controlled Amazon Grocery subset was also prepared for a fairer controlled cross-domain experiment:

| Component | Count |
| --- | ---: |
| Users | 780 |
| Reviews | 5,052 |
| Items | 1,195 |
| Train/dev/test users | 624 / 78 / 78 |

The full Amazon Grocery dataset and the dense Amazon Grocery subset answer different questions. The full dataset is a large-catalog stress test with 48k products. The dense subset is a controlled cross-domain setting where each retained item has enough review/title evidence to form a meaningful keyword profile.

Amazon products are mapped into the same schema as restaurants:

```text
Amazon user_id       -> user_id
Amazon item_id/asin  -> rest_id
Amazon full_text     -> text
Amazon rating        -> rating
```

`full_text` combines product-title evidence with review evidence, which is preferred over raw `review_text` because product titles expose transferable product/category signals such as organic, spicy, gluten-free, coffee, candy, tea, or sauce. This allows the same Kalm4Rec pipeline to run on restaurants and Amazon products.

### Data Preparation

The Yelp notebook output is converted into Kalm4Rec-compatible files with:

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

This creates two evaluation setups:

| Protocol | Files | Purpose |
| --- | --- | --- |
| Temporal Task B protocol | `naija_yelp.csv`, `naija_yelp_holdout.csv` | Next-item recommendation with one held-out future item per user |
| Kalm4Rec paper protocol | `naija_yelp_paper.csv`, `naija_yelp_paper_splits.json` | User-disjoint cold-start protocol matching the Kalm4Rec paper style |

The paper protocol is the main protocol used for comparing with Kalm4Rec-style results, because it matches the original paper’s evaluation logic: test users are not in the training graph, and all reviewed items for a test user are treated as relevant labels.

### Model Pipeline

The system uses a three-stage architecture:

```text
review text -> keyword extraction -> MPG graph retrieval -> LLM/hybrid reranking
```

#### Stage 1: Keyword Extraction

Keywords are extracted from user and item reviews using NLTK noun-phrase extraction:

```bash
python extractor.py \
  --city naija_yelp_paper \
  --edgeType IUF \
  --kwExtractor kw_NLTK
```

The extractor creates:

```text
data/keywords/{city}-keywords_train.json
data/keywords/{city}-keywords_test.json
data/score/{city}-keywords-TFIUF.json
data/embedding/{city}_kwSenEB_pad_train.npy
data/embedding/{city}_kwSenEB_pad_test.npy
```

#### Stage 2: MPG Candidate Retrieval

The retrieval stage uses MPG, the graph retrieval method built into the Kalm4Rec codebase:

```bash
python retrieval.py \
  --city naija_yelp_paper \
  --edgeType IUF \
  --quantity 20 \
  --validTopK 20 \
  --export2LLMs
```

MPG operates on a review-derived heterogeneous keyword graph:

```text
user -> keyword -> restaurant/item
```

The graph does not use an external ontology-style knowledge graph. Instead, it represents users and restaurants/items through extracted review keywords, weighted by TF-IUF.

#### Why `quantity=20`

`--quantity` does not mean the dataset size or number of users. It controls how many top keyword signals are used in graph retrieval.

The best observed result came from:

```text
quantity=20
```

Increasing the value added noise:

| Retrieval Setting | NDCG@20 |
| --- | ---: |
| `quantity=20` | 0.611271 |
| `quantity=50` | 0.550238 |

The reason is that the first 20 keywords contain the strongest preference signals, while additional keywords often include generic or weak review terms. More keywords therefore reduced ranking quality.

The retrieval script now also saves quantity-tagged candidate files, for example:

```text
data/out2LLMs/naija_yelp_paper_q20_knn2rest.json
data/out2LLMs/naija_yelp_paper_q20_user2candidate.json
```

This prevents later runs with different quantities from silently overwriting the strongest candidate files.

#### Stage 3: LLM and Hybrid Reranking

The LLM reranker receives:

- user keywords,
- candidate restaurant/item IDs,
- candidate keywords,
- optional metadata and semantic profiles.

The reranker can run in listwise or scored mode. In practice, pure scored Gemini reranking was not reliable enough, because it often assigned many candidates very similar low scores. This caused relevant MPG candidates to move down the ranking.

The strongest result came from hybrid reranking:

```bash
python taskB/hybrid_rerank.py \
  --city naija_yelp_paper \
  --alpha 0.3
```

The hybrid score blends MPG retrieval order and LLM ranking:

```text
hybrid_score = alpha * MPG_score + (1 - alpha) * LLM_score
```

`alpha=0.3` gave the best measured NDCG@10 and NDCG@20. `alpha=0.6` gave slightly better Recall@10 and HitRate@10.

### Cold-Start Design

Cold-start recommendation is supported by replacing or reducing historical collaborative signals with profile text and semantic preference signals. The cold-start input file is:

```text
data/reviews/naija_yelp_cold_start.csv
```

The cold-start profile includes preference cues such as:

- pepper/spice preference,
- jollof/rice dishes,
- suya or grilled meat,
- halal needs,
- portion size,
- value for money,
- family-style dining,
- warm service,
- Nigerian or West African similarity.

This allows the system to retrieve restaurants from text and metadata-derived preference signals even when full user history is unavailable.

### Cross-Domain Design

The cross-domain extension uses Amazon reviews. The first preparation pass can read the raw enriched Amazon CSV where product metadata still appears beside each review:

```bash
python taskB/prepare_amazon_reviews.py \
  --reviews data/reviews/amazonGrocery.csv \
  --metadata data/reviews/amazonGrocery.csv \
  --dataset_name amazonGrocery
```

After this step, `data/reviews/amazonGrocery.csv` is the normalized Kalm4Rec review file and product metadata is stored separately in `data/metadata/amazonGrocery_restaurant_detail.csv`. Any later filtered Amazon dataset should therefore use the metadata file:

```bash
python taskB/prepare_amazon_reviews.py \
  --reviews data/reviews/amazonGrocery.csv \
  --metadata data/metadata/amazonGrocery_restaurant_detail.csv \
  --dataset_name amazonGrocery_dense \
  --min_reviews 5 \
  --min_item_reviews 10 \
  --max_items 10000
```

Then the same pipeline can be run:

```bash
python extractor.py --city amazonGrocery --edgeType IUF --kwExtractor kw_NLTK --overwrite

python retrieval.py \
  --city amazonGrocery \
  --edgeType IUF \
  --quantity 20 \
  --validTopK 20 \
  --export2LLMs
```

The Amazon setup should be interpreted as cross-domain robustness/evaluation. It does not directly improve Yelp restaurant NDCG because the item spaces are different. Yelp restaurants and Amazon grocery products do not share item IDs or interaction graphs. The benefit is that the same keyword-graph and LLM-reranking method can be evaluated across domains.

For a denser Amazon controlled experiment, create a filtered dataset instead of evaluating only over the full sparse 48k-item catalog:

```bash
python taskB/prepare_amazon_reviews.py \
  --reviews data/reviews/amazonGrocery.csv \
  --metadata data/metadata/amazonGrocery_restaurant_detail.csv \
  --dataset_name amazonGrocery_dense \
  --min_reviews 5 \
  --min_item_reviews 10 \
  --max_items 10000
```

This is not the same task as full Amazon Grocery. It is a density-controlled experiment that tests whether metadata-enriched cross-domain retrieval improves when each product has enough textual evidence. It should be reported separately from the full Amazon stress test, not as a replacement for it.

### Knowledge Graph Position

ColdRAG-style systems often use LLM-generated item profiles, entity-relation extraction, vector databases, and LLM-guided graph traversal.

This project uses a lighter and more efficient Kalm4Rec-style knowledge graph:

```text
user -> keyword -> restaurant/item
```

A `semantic_profile` field was added to restaurant metadata to provide a ColdRAG-inspired textual profile, but full LLM entity/relation graph construction was not used by default.

The full LLM knowledge graph approach was not adopted because:

- it would require thousands of item-profile and entity-extraction calls,
- it would add high API cost and latency,
- LLM-guided graph edge scoring for 962 test users would be expensive,
- the existing MPG keyword graph already produced Kalm4Rec-level results,
- the strongest observed improvement came from hybrid ranking, not from generating more graph text.

### Why MPG Is Used Instead of LightGCN

MPG is used as the main retrieval method because it is native to Kalm4Rec and directly supports text-driven cold-start recommendation.

MPG uses:

```text
user -> keyword -> item
```

This makes it suitable for users or items with sparse interaction history, because the system can still use review keywords, profile text, and metadata-derived semantic signals.

LightGCN is a collaborative filtering model based on a user-item interaction graph:

```text
user -> item
```

LightGCN can be a useful baseline, especially in warm-start settings with enough interaction data. However, it is not the main method for this project because:

- it does not directly use review text or extracted keywords,
- it struggles with new users that have little or no interaction history,
- it struggles with new items that have few or no interactions,
- it does not naturally transfer between Yelp restaurants and Amazon products,
- adding textual features to LightGCN would require a separate hybrid model not present in this codebase.

Therefore, LightGCN should be described as a possible collaborative-filtering baseline, not as the main method.

Recommended framing:

```text
LightGCN tests whether interaction-only collaborative filtering is sufficient.
Kalm4Rec tests whether keyword-based graph retrieval and LLM reranking improve cold-start and cross-domain recommendation.
```

## Chapter 4: Results and Discussion

### Temporal Task B Baseline Results

The temporal Task B protocol is strict because each user has only one held-out future item. This makes absolute scores much lower than the paper-style protocol.

Popularity and random baselines on the temporal test split:

| Method | NDCG@20 | HitRate@20 |
| --- | ---: | ---: |
| Popularity | 0.042748 | 0.102911 |
| Random | 0.002236 | 0.005198 |

This confirms that the temporal protocol is much harder and should not be directly compared with Kalm4Rec paper scores.

### Amazon Cross-Domain Results

The Amazon Grocery experiment should be reported in two parts.

#### Full Amazon Grocery Stress Test

The full Amazon Grocery setup contains 48,297 products and is therefore much harder than the Yelp restaurant setup. It tests whether the same Kalm4Rec keyword graph can operate in a large sparse product catalog.

The first review-only Amazon run produced:

| Amazon Input Text | NDCG@20 |
| --- | ---: |
| Review text only | 0.070276 |

After replacing raw review text with `full_text`, which includes product-title evidence plus review evidence, the Amazon result improved substantially:

| Amazon Input Text | NDCG@20 |
| --- | ---: |
| Product title + review text (`full_text`) | about 0.320000 |

This shows that metadata-enriched text is important for cross-domain recommendation. Product titles introduce transferable product/category signals such as coffee, tea, gluten-free, organic, spicy, candy, sauce, or beverage, while raw reviews often overemphasize isolated complaints such as packaging or delivery.

The full Amazon result should not be directly compared with the Yelp NDCG score because the item spaces are very different:

| Dataset | Items |
| --- | ---: |
| Yelp restaurant paper protocol | about 2,558 |
| Full Amazon Grocery | 48,297 |

#### Dense Amazon Grocery Controlled Test

The dense Amazon Grocery subset was created with:

```bash
python taskB/prepare_amazon_reviews.py \
  --reviews data/reviews/amazonGrocery.csv \
  --metadata data/metadata/amazonGrocery_restaurant_detail.csv \
  --dataset_name amazonGrocery_dense \
  --min_reviews 5 \
  --min_item_reviews 10 \
  --max_items 10000
```

This produced:

| Component | Count |
| --- | ---: |
| Users | 780 |
| Reviews | 5,052 |
| Items | 1,195 |
| Train/dev/test users | 624 / 78 / 78 |

This subset is not used to hide the full Amazon difficulty. It controls for item sparsity by keeping products with at least 10 reviews and users with at least 5 reviews. The full Amazon result is the stress test; the dense Amazon result is the controlled cross-domain experiment.

### Protocol Justification and Sources

The full-vs-dense Amazon design follows established recommender-system evaluation practice rather than being an arbitrary reduction of the dataset.

1. **k-core filtering is standard for Amazon recommendation datasets.**
   The public McAuley Amazon datasets explicitly provide dense `5-core` subsets, where remaining users and items have at least 5 reviews. The older Amazon data page describes k-cores as dense subsets where each remaining user and item has `k` reviews, and the current Amazon review data page also provides a `5-core` subset. Our `amazonGrocery_dense` setting follows this same logic, but uses a stricter item threshold of 10 reviews to ensure stronger product keyword profiles.

2. **Data split and filtering choices affect recommender rankings.**
   Meng et al. (2020) show that recommender-system results are strongly affected by splitting strategy and that evaluation protocols can change the apparent ranking of models. This supports reporting the full sparse Amazon stress test separately from the dense controlled Amazon test, rather than mixing the two as if they were the same task.

3. **Top-K recommendation is sensitive to catalog size and sparsity.**
   A top-20 ranking over 48,297 products is not comparable to a top-20 ranking over about 2,558 restaurants. The full Amazon setup primarily measures large-catalog sparse retrieval; the dense subset measures whether metadata-enriched keyword transfer works when products have enough textual evidence.

4. **Collaborative-filtering baselines such as LightGCN operate on user-item interaction graphs.**
   LightGCN learns user and item embeddings by propagating signals over the user-item interaction graph. This makes it a relevant baseline for interaction-only recommendation, but it does not directly use review text, product titles, categories, or semantic profiles unless additional feature modeling is added.

These sources justify the chapter design:

- McAuley Amazon review datasets: the Amazon data pages provide product reviews, metadata, and k-core/5-core subsets for experimentation and reproducibility.
- Meng et al. (2020), *Exploring Data Splitting Strategies for the Evaluation of Recommendation Models*: evaluation split choices can substantially affect recommender results.
- He et al. (2020), *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*: LightGCN is a strong collaborative-filtering baseline based on user-item graph propagation.
- He et al. (2017), *Neural Collaborative Filtering*: collaborative filtering models user preference from past user-item interactions, which explains why interaction-only baselines are different from text-enriched cold-start retrieval.

### Kalm4Rec Paper-Protocol MPG Retrieval

Fair Random and Popularity baselines were evaluated on the same `naija_yelp_paper` test users. Popularity was computed from train users only to avoid leaking test-user interactions into the baseline:

```bash
python taskB/evaluate_baselines.py \
  --history data/reviews/naija_yelp_paper.csv \
  --holdout data/reviews/naija_yelp_paper.csv \
  --k 20 \
  --splits data/reviews/naija_yelp_paper_splits.json \
  --dataset_name naija_yelp_paper \
  --split test \
  --history_split train
```

| Baseline | NDCG@10 | HitRate@10 | NDCG@20 | HitRate@20 |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.004646 | 0.035343 | 0.005746 | 0.068607 |
| Popularity | 0.088333 | 0.374220 | 0.100273 | 0.489605 |

The main retrieval result was obtained with:

```bash
python retrieval.py \
  --city naija_yelp_paper \
  --edgeType IUF \
  --quantity 20 \
  --validTopK 20 \
  --export2LLMs
```

Result:

| Method | Precision@20 | Recall@20 | F1@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: |
| MPG retrieval | 0.145166 | 0.326992 | 0.184961 | 0.611271 |

This shows that the adapted Yelp restaurant dataset can reproduce Kalm4Rec-level paper-protocol performance.

### LLM Reranking Findings

A pure Gemini reranker improved some early-rank metrics but slightly reduced other metrics:

| Method | P@1 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPG retrieval | 0.448025 | 0.241273 | 0.619777 | 0.326992 | 0.611271 |
| Gemini reranker | 0.480249 | 0.224172 | 0.617021 | 0.326992 | 0.607893 |

Interpretation:

- The LLM improves the first recommendation position.
- However, it can move relevant MPG candidates lower in the top-20 ranking.
- This slightly hurts recall-oriented and full-list ranking metrics.

The scored reranker was worse when used directly:

| Method | NDCG@10 | NDCG@20 | Recall@20 |
| --- | ---: | ---: | ---: |
| Scored Gemini reranker | 0.578696 | 0.564693 | 0.290734 |

The likely reason is that the model assigned many candidates nearly identical low scores, which weakened the original MPG order.

### Hybrid Reranking Results

Hybrid reranking gave the best overall ranking quality.

#### Hybrid, `alpha=0.3`

```bash
python taskB/hybrid_rerank.py \
  --city naija_yelp_paper \
  --alpha 0.3
```

| Metric | Score |
| --- | ---: |
| Precision@1 | 0.496881 |
| Recall@1 | 0.064816 |
| NDCG@1 | 0.496881 |
| Precision@3 | 0.336452 |
| Recall@3 | 0.124793 |
| NDCG@3 | 0.602886 |
| Precision@5 | 0.270270 |
| Recall@5 | 0.163436 |
| NDCG@5 | 0.623895 |
| Precision@10 | 0.197505 |
| Recall@10 | 0.233253 |
| NDCG@10 | 0.629689 |
| HitRate@10 | 0.843035 |
| Precision@20 | 0.145166 |
| Recall@20 | 0.326992 |
| NDCG@20 | 0.620736 |
| HitRate@20 | 0.913721 |

#### Hybrid, `alpha=0.6`

```bash
python taskB/hybrid_rerank.py \
  --city naija_yelp_paper \
  --alpha 0.6
```

| Metric | Score |
| --- | ---: |
| Precision@1 | 0.467775 |
| Recall@1 | 0.060811 |
| NDCG@1 | 0.467775 |
| Precision@3 | 0.346847 |
| Recall@3 | 0.128970 |
| NDCG@3 | 0.592476 |
| Precision@5 | 0.284200 |
| Recall@5 | 0.171665 |
| NDCG@5 | 0.616881 |
| Precision@10 | 0.207380 |
| Recall@10 | 0.242829 |
| NDCG@10 | 0.628581 |
| HitRate@10 | 0.860707 |
| Precision@20 | 0.145166 |
| Recall@20 | 0.326992 |
| NDCG@20 | 0.620657 |
| HitRate@20 | 0.913721 |

### Best Result Summary

The result is metric-specific. Raw MPG is the strongest simple retrieval baseline and should not be described as weaker overall. Hybrid reranking improves ranking-quality metrics such as NDCG, but raw MPG remains competitive and is stronger than `alpha=0.3` on Recall@10 and HitRate@10.

| Method | P@10 | Recall@10 | NDCG@10 | HitRate@10 | NDCG@20 | HitRate@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | - | - | 0.004646 | 0.035343 | 0.005746 | 0.068607 |
| Popularity | - | - | 0.088333 | 0.374220 | 0.100273 | 0.489605 |
| MPG retrieval | 0.208524 | 0.241273 | 0.619777 | 0.859667 | 0.611271 | 0.913721 |
| Hybrid, `alpha=0.3` | 0.197505 | 0.233253 | 0.629689 | 0.843035 | 0.620736 | 0.913721 |
| Hybrid, `alpha=0.6` | 0.207380 | 0.242829 | 0.628581 | 0.860707 | 0.620657 | 0.913721 |

The most defensible reading is:

- Use **MPG retrieval** as the strongest non-LLM baseline.
- Use **Hybrid, `alpha=0.3`** when optimizing NDCG@10/NDCG@20.
- Use **Hybrid, `alpha=0.6`** when optimizing Recall@10/HitRate@10 while keeping NDCG close to the best value.

Hybrid `alpha=0.3` improves NDCG@20 from:

```text
0.611271 -> 0.620736
```

This is an absolute improvement of about:

```text
+0.009465
```

or approximately:

```text
+0.95 percentage points
```

However, this NDCG gain comes with lower Recall@10 and HitRate@10 than raw MPG. Therefore, the Chapter 4 discussion should report MPG and Hybrid side by side rather than claiming that Hybrid is universally better.

### Why NDCG@10 Did Not Reach 80%

The target of 80% NDCG@10 is ambitious for a non-leaky recommender. The current best measured NDCG@10 is about 63%.

An oracle reorder of the candidate set can produce much higher results, which means the candidate pool contains many relevant items. However, reaching 80% in a real evaluation would likely require one of the following:

- a supervised reranker trained on relevance labels,
- more explicit user preference labels,
- additional item metadata,
- stronger domain-specific semantic profiles,
- or label leakage, which should not be used.

Therefore, the current result should be reported honestly as strong but not 80%. Hybrid reranking improves over the Kalm4Rec retrieval baseline on NDCG, while raw MPG remains stronger on some top-10 coverage metrics.

### HitRate Interpretation

HitRate@K measures whether at least one relevant item appears in the top K.

For example:

```text
HitRate@10 = 0.843035
```

means about 84.3% of test users received at least one relevant restaurant in their top 10 recommendations.

HitRate differs from Recall because Recall measures the fraction of all relevant items recovered, while HitRate only checks whether there is at least one hit.

### Final Interpretation

The results support four main findings:

1. MPG is a strong retrieval method for this cold-start setup because it uses review-derived keyword evidence rather than only user-item interactions.
2. Increasing the retrieval keyword quantity beyond 20 hurts performance because it introduces noisy keyword signals.
3. Pure LLM reranking is not consistently better than MPG; it improves some early-rank metrics but can damage graph-based relevance ordering.
4. Hybrid MPG + LLM reranking gives the best overall result by preserving graph evidence while adding semantic reranking.

The recommended final system is therefore:

```text
Kalm4Rec keyword extraction + MPG retrieval with quantity=20 + hybrid reranking with alpha=0.3
```

### Recommended Chapter 4 Claim

The adapted Kalm4Rec system achieved:

```text
NDCG@10 = 62.97%
NDCG@20 = 62.07%
HitRate@10 = 84.30%
HitRate@20 = 91.37%
```

on the Kalm4Rec paper-style user-disjoint restaurant recommendation protocol.

The result demonstrates that a keyword-graph retrieval system, enhanced with lightweight LLM reranking, can support cold-start and cross-domain recommendation more naturally than interaction-only collaborative filtering methods.

## References

- McAuley Lab. Amazon Review Data, including product reviews, metadata, and k-core/5-core subsets: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
- McAuley Lab. Older Amazon Product Data page describing k-core dense subsets: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
- Meng, Z., McCreadie, R., Macdonald, C., and Ounis, I. (2020). *Exploring Data Splitting Strategies for the Evaluation of Recommendation Models*: https://arxiv.org/abs/2007.13237
- He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., and Wang, M. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*: https://arxiv.org/abs/2002.02126
- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., and Chua, T.-S. (2017). *Neural Collaborative Filtering*: https://hexiangnan.github.io/papers/www17-ncf.pdf
