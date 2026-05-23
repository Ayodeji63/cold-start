const persona = document.querySelector("#persona");
const domain = document.querySelector("#domain");
const city = document.querySelector("#city");
const topK = document.querySelector("#topK");
const statusEl = document.querySelector("#status");
const resultsEl = document.querySelector("#results");
const recommendBtn = document.querySelector("#recommendBtn");

const samples = {
  restaurant:
    "A Nigerian student in Philadelphia who likes spicy jollof, halal meat, generous portions, affordable food, and warm service for group dinners.",
  grocery:
    "A health-conscious shopper who likes organic gluten-free snacks, spicy sauces, coffee, tea, and good-value pantry staples.",
};

function setLoading(isLoading) {
  recommendBtn.disabled = isLoading;
  recommendBtn.textContent = isLoading ? "Ranking..." : "Recommend";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderResults(items) {
  if (!items.length) {
    resultsEl.innerHTML = '<div class="empty">No matching recommendations found.</div>';
    return;
  }

  resultsEl.innerHTML = items
    .map((item) => {
      const meta = [];
      if (item.city) meta.push(item.city);
      if (item.categories) meta.push(item.categories);
      if (item.metadata?.stars) meta.push(`${item.metadata.stars} stars`);
      if (item.metadata?.price && item.metadata.price !== "?") meta.push(`price ${item.metadata.price}`);

      return `
        <article class="result">
          <div class="result-top">
            <div class="rank-name">
              <div class="rank">${escapeHtml(item.rank)}</div>
              <div>
                <div class="name">${escapeHtml(item.name)}</div>
                <div class="meta">${escapeHtml(meta.join(" · "))}</div>
              </div>
            </div>
            <div class="score">${escapeHtml(item.score)}</div>
          </div>
          <div class="reason">${escapeHtml(item.reason)}</div>
        </article>
      `;
    })
    .join("");
}

async function recommend() {
  setLoading(true);
  statusEl.textContent = "Ranking candidates";
  resultsEl.innerHTML = "";

  const selectedDomain = domain.value;
  const payload = {
    persona: persona.value,
    domain: selectedDomain,
    top_k: Number(topK.value || 5),
    city: selectedDomain === "restaurants" ? city.value || null : null,
  };

  try {
    const response = await fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "Recommendation failed");
    }
    statusEl.textContent = `${body.count} recommendations`;
    renderResults(body.recommendations);
  } catch (error) {
    statusEl.textContent = "Error";
    resultsEl.innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
  } finally {
    setLoading(false);
  }
}

function syncControls() {
  const isRestaurant = domain.value === "restaurants";
  city.disabled = !isRestaurant;
  if (!isRestaurant) city.value = "";
}

recommendBtn.addEventListener("click", recommend);
domain.addEventListener("change", syncControls);
document.querySelector("#sampleRestaurant").addEventListener("click", () => {
  domain.value = "restaurants";
  city.value = "Philadelphia";
  persona.value = samples.restaurant;
  syncControls();
});
document.querySelector("#sampleGrocery").addEventListener("click", () => {
  domain.value = "amazon_grocery_dense";
  persona.value = samples.grocery;
  syncControls();
});

syncControls();
recommend();
