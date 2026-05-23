# NaijaRec_ColdStart Containerized Recommendation API

This is the submission-facing application for the BCT/DSN Task B requirement:

> The system should take user persona as input and produce personalized recommendations as output.

The API uses the project metadata profiles generated from the Kalm4Rec experiments. It supports:

- Nigerian-focused Yelp restaurant recommendations.
- Full Amazon Grocery product recommendations.
- Dense Amazon Grocery product recommendations.

The API is deterministic and does not require a Gemini API key at runtime. Gemini was used only in offline reranking experiments.

## Run Locally

```bash
pip install -r requirements-api.txt
uvicorn app.NaijaRec_ColdStart:app --host 0.0.0.0 --port 8001
```

Open:

```text
http://localhost:8001/docs
```

## Run With Docker

```bash
docker build -t naijarec-coldstart .
docker run --rm -p 8001:8001 naijarec-coldstart
```

## Endpoints

### Health

```bash
curl http://localhost:8001/health
```

### Available Datasets

```bash
curl http://localhost:8001/datasets
```

### Recommend Restaurants

```bash
curl -X POST http://localhost:8001/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "restaurants",
    "top_k": 5,
    "city": "Philadelphia",
    "persona": "A Nigerian student in Philadelphia who likes spicy jollof, halal meat, generous portions, affordable food, and warm service for group dinners."
  }'
```

### Recommend Amazon Grocery Products

```bash
curl -X POST http://localhost:8001/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "amazon_grocery",
    "top_k": 5,
    "persona": "A health-conscious shopper who likes organic gluten-free snacks, spicy sauces, coffee, tea, and good-value pantry staples."
  }'
```

### Recommend From Dense Amazon Grocery

```bash
curl -X POST http://localhost:8001/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "amazon_grocery_dense",
    "top_k": 5,
    "persona": "A parent buying affordable snacks, sugar-free candy, tea, coffee pods, and healthy pantry items."
  }'
```

## Request Schema

```json
{
  "persona": "natural-language user persona",
  "domain": "restaurants | amazon_grocery | amazon_grocery_dense",
  "top_k": 10,
  "city": "optional restaurant city filter"
}
```

## Response Schema

```json
{
  "domain": "restaurants",
  "top_k": 5,
  "count": 5,
  "recommendations": [
    {
      "rank": 1,
      "id": "business_or_product_id",
      "name": "item name",
      "domain": "restaurants",
      "city": "Philadelphia",
      "categories": "Restaurants, African, ...",
      "score": 12.34,
      "reason": "matches spicy, halal; aligns with value; available in Philadelphia",
      "metadata": {
        "stars": "4.5",
        "review_count": "120",
        "price": "2"
      }
    }
  ]
}
```

## Design Note

The offline research system evaluates MPG retrieval and hybrid reranking with NDCG, Recall, Precision, F1, and HitRate. The containerized API is the interactive serving layer. It uses metadata-enriched item profiles and persona-token matching to produce explainable recommendations for new user personas, which directly satisfies the Task B deliverable.
