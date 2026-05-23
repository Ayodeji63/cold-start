import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9&'\-]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "our", "that", "the", "their", "them", "this", "to", "with", "you",
    "your", "want", "wants", "like", "likes", "need", "needs", "prefer",
    "prefers", "looking", "recommend", "recommendation",
}

SIGNAL_TERMS = {
    "spicy": {"spicy", "pepper", "peppery", "hot", "chilli", "chili", "suya", "jerk"},
    "value": {"cheap", "affordable", "budget", "value", "portion", "portions", "large"},
    "service": {"service", "friendly", "staff", "warm", "hospitality"},
    "family": {"family", "group", "kids", "sharing", "platter"},
    "halal": {"halal", "muslim"},
    "west_african": {"nigerian", "african", "ghanaian", "jollof", "suya", "egusi", "fufu", "plantain"},
}


def tokenize(text):
    return [
        tok.lower().strip("-'")
        for tok in TOKEN_RE.findall(text or "")
        if len(tok) > 2 and tok.lower() not in STOPWORDS
    ]


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


class ProfileIndex:
    def __init__(self, dataset, metadata_path, item_label):
        self.dataset = dataset
        self.metadata_path = Path(metadata_path)
        self.item_label = item_label
        self.items = []
        self.df = Counter()
        self.idf = {}
        self._load()

    def _load(self):
        if not self.metadata_path.is_file():
            return

        with self.metadata_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile = " ".join(
                    str(row.get(col, ""))
                    for col in ["name", "categories", "brand", "description", "semantic_profile", "city"]
                )
                tokens = set(tokenize(profile))
                item = {
                    "id": row.get("business_id") or row.get("rest_id") or row.get("item_id"),
                    "name": row.get("name") or row.get("business_id") or row.get("rest_id"),
                    "city": row.get("city", ""),
                    "categories": row.get("categories", ""),
                    "stars": row.get("stars", ""),
                    "review_count": row.get("review_count", ""),
                    "price": row.get("price", ""),
                    "semantic_profile": row.get("semantic_profile", ""),
                    "tokens": tokens,
                    "signals": {
                        "halal": parse_bool(row.get("halal", "")),
                        "west_african": parse_bool(row.get("west_african_similarity", "")),
                        "spicy": parse_bool(row.get("spice_signal", "")),
                        "value": parse_bool(row.get("value_signal", "")),
                        "service": parse_bool(row.get("service_signal", "")),
                        "family": parse_bool(row.get("family_style_signal", "")),
                    },
                }
                if item["id"] and item["tokens"]:
                    self.items.append(item)
                    self.df.update(tokens)

        n_docs = max(len(self.items), 1)
        self.idf = {
            tok: math.log((1 + n_docs) / (1 + count)) + 1.0
            for tok, count in self.df.items()
        }

    def score(self, persona, top_k=10, city_filter=None):
        persona_tokens = set(tokenize(persona))
        persona_signals = {
            signal
            for signal, terms in SIGNAL_TERMS.items()
            if persona_tokens.intersection(terms)
        }

        scored = []
        for item in self.items:
            if city_filter and item["city"].lower() != city_filter.lower():
                continue

            matched = persona_tokens.intersection(item["tokens"])
            lexical = sum(self.idf.get(tok, 1.0) for tok in matched)
            signal_score = sum(
                2.0
                for signal in persona_signals
                if item["signals"].get(signal) or signal in item["tokens"]
            )
            score = lexical + signal_score
            if score <= 0:
                continue

            reasons = self._reasons(item, matched, persona_signals)
            scored.append((score, item, reasons))

        scored.sort(key=lambda row: row[0], reverse=True)
        return [
            {
                "rank": rank,
                "id": item["id"],
                "name": item["name"],
                "domain": self.dataset,
                "city": item["city"],
                "categories": item["categories"],
                "score": round(score, 4),
                "reason": reason,
                "metadata": {
                    "stars": item["stars"],
                    "review_count": item["review_count"],
                    "price": item["price"],
                },
            }
            for rank, (score, item, reason) in enumerate(scored[:top_k], start=1)
        ]

    def _reasons(self, item, matched, persona_signals):
        parts = []
        if matched:
            parts.append("matches " + ", ".join(sorted(matched)[:6]))
        aligned_signals = [
            signal.replace("_", " ")
            for signal in persona_signals
            if item["signals"].get(signal) or signal in item["tokens"]
        ]
        if aligned_signals:
            parts.append("aligns with " + ", ".join(aligned_signals[:4]))
        if item["city"]:
            parts.append(f"available in {item['city']}")
        return "; ".join(parts) if parts else "semantic profile match"


class RecommendationService:
    def __init__(self, root_dir="."):
        root = Path(root_dir)
        self.configs = {
            "restaurants": {
                "metadata_path": root / "data/metadata/naija_yelp_paper_restaurant_detail.csv",
                "item_label": "restaurant",
            },
            "amazon_grocery": {
                "metadata_path": root / "data/metadata/amazonGrocery_restaurant_detail.csv",
                "item_label": "product",
            },
            "amazon_grocery_dense": {
                "metadata_path": root / "data/metadata/amazonGrocery_dense_restaurant_detail.csv",
                "item_label": "product",
            },
        }
        self.indexes = {}

    def datasets(self):
        return {
            name: {
                "loaded": name in self.indexes,
                "items": len(self.indexes[name].items) if name in self.indexes else None,
                "metadata_path": str(config["metadata_path"]),
                "metadata_exists": config["metadata_path"].is_file(),
                "item_label": config["item_label"],
            }
            for name, config in self.configs.items()
        }

    def recommend(self, persona, domain="restaurants", top_k=10, city_filter=None):
        index = self._index(domain)
        top_k = max(1, min(int(top_k), 50))
        return index.score(persona, top_k=top_k, city_filter=city_filter)

    def _index(self, domain):
        if domain not in self.configs:
            raise KeyError(f"Unknown domain '{domain}'. Available domains: {', '.join(self.configs)}")
        if domain not in self.indexes:
            config = self.configs[domain]
            self.indexes[domain] = ProfileIndex(
                domain,
                config["metadata_path"],
                config["item_label"],
            )
        return self.indexes[domain]
