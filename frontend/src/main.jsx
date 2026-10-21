import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const API = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const NAVIGATION = [
  ["overview", "Overview", "01"],
  ["products", "Product explorer", "02"],
  ["compare", "Compare products", "03"],
  ["reviews", "Review explorer", "04"],
  ["upload", "CSV analysis", "05"],
];
const ASPECTS = [
  "battery", "sound", "camera", "display", "performance", "comfort",
  "build quality", "price/value", "connectivity", "charging",
];

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return payload;
}

function formatCount(value) {
  return Number(value || 0).toLocaleString();
}

function safeSource(url) {
  try {
    const parsed = new URL(url);
    return ["http:", "https:"].includes(parsed.protocol) ? parsed.href : "";
  } catch {
    return "";
  }
}

function downloadCsv(rows, filename) {
  if (!rows?.length) return;
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const quote = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  const content = [
    columns.map(quote).join(","),
    ...rows.map((row) => columns.map((column) => quote(row[column])).join(",")),
  ].join("\r\n");
  const url = URL.createObjectURL(new Blob(["\ufeff", content], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function SentimentBars({ distribution }) {
  if (!distribution) return null;
  return (
    <div className="sentiment-grid">
      {["positive", "negative", "neutral"].map((label) => {
        const result = distribution[label] || { count: 0, percent: 0 };
        return (
          <div className={`sentiment-card ${label}`} key={label}>
            <div className="sentiment-topline">
              <span className="sentiment-label">{label}</span>
              <strong>{Number(result.percent).toFixed(1)}%</strong>
            </div>
            <div className="bar-track" role="img" aria-label={`${label}: ${result.percent}%`}>
              <span style={{ width: `${Math.max(0, Math.min(100, result.percent))}%` }} />
            </div>
            <small>{formatCount(result.count)} reviews</small>
          </div>
        );
      })}
    </div>
  );
}

function Notice({ children, tone = "info" }) {
  return <div className={`notice ${tone}`} role={tone === "error" ? "alert" : "status"}>{children}</div>;
}

function LoadingBlock({ label = "Loading review data…" }) {
  return (
    <div className="loading-block" aria-live="polite">
      <span className="skeleton-line wide" />
      <span className="skeleton-line" />
      <span className="loading-label">{label}</span>
    </div>
  );
}

function ReviewList({ reviews }) {
  if (!reviews?.length) return <Notice>No reviews available for this selection.</Notice>;
  return (
    <div className="review-list">
      {reviews.map((review, index) => {
        const source = safeSource(review.source_url);
        return (
          <article className="review-card" key={review.review_id || `${review.product_id}-${index}`}>
            <div className="review-meta">
              <span className={`tag ${review.predicted_sentiment || "neutral"}`}>
                {review.predicted_sentiment || "not predicted"}
              </span>
              {review.platform && <span>{review.platform}</span>}
              {review.review_date && <span>{review.review_date}</span>}
              {review.product_id && <span>ID {review.product_id}</span>}
            </div>
            <p>{review.review_text}</p>
            {source && <a href={source} target="_blank" rel="noreferrer">Open source review ↗</a>}
          </article>
        );
      })}
    </div>
  );
}

function EvidenceGroups({ result }) {
  if (!result) return null;
  const groups = result.evidence.reduce((acc, item) => {
    (acc[item.aspect] ||= []).push(item);
    return acc;
  }, {});
  const opinionCount = result.evidence.length;
  return (
    <section className="panel evidence-panel">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Aspect-based evidence</span>
          <h2>What reviewers praise and criticise</h2>
        </div>
        <span className="count-chip">{formatCount(result.analyzed_review_count)} reviews analysed</span>
      </div>
      <p className="muted">
        {result.warning || `Paired review-context and target-aspect inference · ${result.model}.`}
        {opinionCount === 0 && " No supported aspect mentions were found."}
      </p>
      {Object.entries(groups).map(([aspect, items]) => {
        const countFor = (label) => new Set(items.filter((item) => item.sentiment === label).map((item) => item.review_id)).size;
        const positiveIds = new Set(items.filter((item) => item.sentiment === "positive").map((item) => item.review_id));
        const negativeIds = new Set(items.filter((item) => item.sentiment === "negative").map((item) => item.review_id));
        const mixed = [...positiveIds].filter((id) => negativeIds.has(id)).length;
        return (
          <details className="aspect-group" open key={aspect}>
            <summary>
              <span>{aspect}</span>
              <small>
                {countFor("positive")} praise · {countFor("negative")} criticism · {mixed} mixed ·{" "}
                {countFor("neutral")} neutral · {countFor("no_opinion")} no opinion · {countFor("uncertain")} uncertain
              </small>
            </summary>
            <div className="evidence-grid">
              {["positive", "negative", "neutral", "no_opinion", "uncertain"].map((label) => {
                const entries = items.filter((item) => item.sentiment === label);
                if (!entries.length) return null;
                return (
                  <div className="evidence-column" key={label}>
                    <h3>{label.replaceAll("_", " ")}</h3>
                    {entries.slice(0, 6).map((item, index) => {
                      const source = safeSource(item.source_url);
                      return (
                        <blockquote key={`${item.review_id}-${index}`}>
                          <p>“{item.review_text}”</p>
                          <footer>
                            Review {item.review_id || "ID unavailable"} · {item.method}
                            {item.confidence != null && ` · ${(item.confidence * 100).toFixed(0)}% confidence`}
                            {source && <> · <a href={source} target="_blank" rel="noreferrer">source</a></>}
                          </footer>
                        </blockquote>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          </details>
        );
      })}
    </section>
  );
}

function Overview({ data, loading, error }) {
  if (loading) return <LoadingBlock label="Loading review analysis…" />;
  if (error) return <Notice tone="error">{error}</Notice>;
  if (!data) return null;
  return (
    <>
      <section className="hero">
        <span className="eyebrow">Consumer technology · review intelligence</span>
        <h1>What do reviewers<br />actually say?</h1>
        <p>Explore product-level sentiment, feature opinions, and the original reviews behind them.</p>
      </section>
      <section className="metric-grid">
        {[
          ["Reviews analyzed", data.review_count],
          ["Product IDs", data.product_id_count],
          ["Named source products", data.named_product_count],
          ["Available sources", data.source_count],
        ].map(([label, value]) => (
          <div className="metric-card" key={label}><span>{label}</span><strong>{formatCount(value)}</strong></div>
        ))}
      </section>
      <section className="panel">
        <div className="section-heading">
          <div><h2>Sentiment</h2></div>
          <span className="count-chip">{formatCount(data.review_count)} reviews</span>
        </div>
        <SentimentBars distribution={data.sentiment} />
      </section>
    </>
  );
}

function ProductExplorer({ products, brands, categories, loadProducts, apiError }) {
  const [query, setQuery] = useState("");
  const [brand, setBrand] = useState("");
  const [category, setCategory] = useState("");
  const [matches, setMatches] = useState([]);
  const [selected, setSelected] = useState("");
  const [detail, setDetail] = useState(null);
  const [aspectResult, setAspectResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [loadingLabel, setLoadingLabel] = useState("Loading product details…");
  const [error, setError] = useState("");

  useEffect(() => {
    setMatches(products);
    if (!selected && products.length) setSelected(products[0].product_id);
  }, [products]);

  useEffect(() => {
    if (!selected) return;
    let active = true;
    setBusy(true);
    setLoadingLabel("Loading product details…");
    setError("");
    setAspectResult(null);
    request(`/api/products/${encodeURIComponent(selected)}`)
      .then((data) => { if (active) setDetail(data); })
      .catch((err) => { if (active) setError(err.message); })
      .finally(() => { if (active) setBusy(false); });
    return () => { active = false; };
  }, [selected]);

  async function search(event) {
    event.preventDefault();
    setError("");
    try {
      const result = await loadProducts(query, brand, category);
      setMatches(result.items);
      setSelected(result.items[0]?.product_id || "");
      if (!result.items.length) setDetail(null);
    } catch (err) {
      setError(err.message);
    }
  }

  async function analyzeAspects() {
    if (!selected) return;
    setBusy(true);
    setLoadingLabel("Analyzing feature opinions…");
    setError("");
    try {
      setAspectResult(await request(`/api/products/${encodeURIComponent(selected)}/aspects`, { method: "POST" }));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <section className="page-intro">
        <span className="eyebrow">Product explorer</span>
        <h1>Find a supported product</h1>
        <p>Source titles are shown where available; otherwise products remain identified by ID.</p>
      </section>
      {apiError && <Notice tone="error">{apiError}</Notice>}
      <form className="search-form" onSubmit={search}>
        <label htmlFor="product-search">Search product names or product IDs</label>
        <div className="search-control">
          <input id="product-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Try a product name or ID" />
          <button className="button primary" type="submit">Search</button>
        </div>
        {(brands.length > 0 || categories.length > 0) && (
          <div className="metadata-filters">
            {brands.length > 0 && <label>Verified brand<select value={brand} onChange={(event) => setBrand(event.target.value)}><option value="">All verified brands</option>{brands.map((value) => <option key={value}>{value}</option>)}</select></label>}
            {categories.length > 0 && <label>Verified category<select value={category} onChange={(event) => setCategory(event.target.value)}><option value="">All verified categories</option>{categories.map((value) => <option key={value}>{value}</option>)}</select></label>}
          </div>
        )}
      </form>
      <p className="muted small-note">No separate verified brand field is available. Search matches source-provided product names and IDs, not brands; brand filters require verified metadata mappings.</p>
      {error && <Notice tone="error">{error}</Notice>}
      {!matches.length && query && <Notice>No reviews available for “{query}”. Search checks source product titles and product IDs.</Notice>}
      {!!matches.length && (
        <div className="product-picker">
          <label htmlFor="product-select">Supported product</label>
          <select id="product-select" value={selected} onChange={(event) => setSelected(event.target.value)}>
            {matches.map((item) => <option key={item.product_id} value={item.product_id}>{item.display_name} · {item.review_count} reviews</option>)}
          </select>
        </div>
      )}
      {busy && <LoadingBlock label={loadingLabel} />}
      {detail && !busy && (
        <>
          <section className="panel product-summary">
            <span className="eyebrow">Supported product</span>
            <h2>{detail.product.display_name}</h2>
            <p className="muted">Product ID {detail.product.product_id} · {formatCount(detail.review_count)} available reviews · Category: {detail.product.category_is_verified ? detail.product.verified_category : "not verified"}</p>
            <p className="muted">Sources: {Object.entries(detail.sources).map(([name, count]) => `${name} (${count})`).join(", ") || "not recorded"} · Dates: {detail.date_start || "unavailable"} to {detail.date_end || "unavailable"}</p>
            <SentimentBars distribution={detail.sentiment} />
            <p className="muted">Overall sentiment is predicted for the whole review; feature opinions are analyzed separately.</p>
          </section>
          <div className="action-row">
            <button className="button primary" type="button" onClick={analyzeAspects} disabled={busy}>Analyze aspect evidence</button>
            <span className="muted">Local CPU inference; first use may download the pretrained aspect model.</span>
          </div>
          {aspectResult && <EvidenceGroups result={aspectResult} />}
          <section className="panel">
            <div className="section-heading"><div><span className="eyebrow">Original evidence</span><h2>Reviews for this product</h2></div></div>
            <ReviewList reviews={detail.reviews.slice(0, 12)} />
            {detail.reviews.length > 12 && <p className="muted">Showing 12 of {formatCount(detail.reviews.length)} reviews. Use Review explorer to filter and export the collection.</p>}
          </section>
        </>
      )}
    </>
  );
}

function Comparison({ catalog }) {
  const categories = [...new Set(catalog.filter((item) => item.category_is_verified).map((item) => item.verified_category))].sort();
  const [category, setCategory] = useState("");
  const [first, setFirst] = useState("");
  const [second, setSecond] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const candidates = catalog.filter((item) => (
    item.has_verified_name
    && (!category || (item.category_is_verified && item.verified_category === category))
  ));

  useEffect(() => {
    setCategory("");
  }, [catalog.length]);
  useEffect(() => {
    setFirst(candidates[0]?.product_id || "");
    setSecond(candidates[1]?.product_id || "");
    setResult(null);
  }, [category, catalog]);

  async function compare(event) {
    event.preventDefault();
    setError("");
    setResult(null);
    try {
      setResult(await request(`/api/compare?product_a=${encodeURIComponent(first)}&product_b=${encodeURIComponent(second)}`));
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <>
      <section className="page-intro"><span className="eyebrow">Product comparison</span><h1>Compare products</h1><p>Compare products with available review data. Verified categories can optionally narrow the choices; review counts are shown for context.</p></section>
      {!candidates.length ? (
        <Notice>No named products with review data are available to compare.</Notice>
      ) : (
        <>
          <form className="compare-form" onSubmit={compare}>
            {!!categories.length && <label>Verified category (optional)<select value={category} onChange={(event) => setCategory(event.target.value)}><option value="">All categories</option>{categories.map((item) => <option key={item}>{item}</option>)}</select></label>}
            {[["First product", first, setFirst], ["Second product", second, setSecond]].map(([label, value, setter]) => (
              <label key={label}>{label}<select value={value} onChange={(event) => setter(event.target.value)}>{candidates.map((item) => <option key={item.product_id} value={item.product_id}>{item.display_name}</option>)}</select></label>
            ))}
            <button className="button primary" type="submit" disabled={!first || !second || first === second}>Compare</button>
          </form>
          {category && candidates.length < 2 && <Notice>At least two named products in this verified category are needed to compare.</Notice>}
          {error && <Notice tone="error">{error}</Notice>}
          {result?.products.map((item) => (
            <section className="panel comparison-card" key={item.product.product_id}>
              <span className="eyebrow">{result.category ? `${result.category} · same verified category` : result.comparison_scope} · {formatCount(item.review_count)} reviews</span>
              <h2>{item.product.display_name}</h2>
              <SentimentBars distribution={item.sentiment} />
            </section>
          ))}
        </>
      )}
    </>
  );
}

function ReviewExplorer({ catalog, overview }) {
  const [filters, setFilters] = useState({
    query: "", product_id: "", sentiment: "", aspect: "", year: "", date_precision: "",
  });
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const years = useMemo(() => {
    const start = Number(String(overview?.date_start || "").slice(0, 4));
    const end = Number(String(overview?.date_end || "").slice(0, 4));
    return start && end ? Array.from({ length: end - start + 1 }, (_, index) => String(start + index)) : [];
  }, [overview]);
  const precisions = Object.keys(overview?.date_precision_counts || {}).filter((value) => !["unavailable", "day"].includes(value) || overview.date_precision_counts[value] > 0);

  function update(key, value) {
    setFilters((current) => ({ ...current, [key]: value }));
  }

  async function search(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    const params = new URLSearchParams(Object.entries(filters).filter(([, value]) => value));
    params.set("limit", "9000");
    try {
      setResult(await request(`/api/reviews?${params.toString()}`));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <section className="page-intro"><span className="eyebrow">Review explorer</span><h1>Inspect the evidence</h1><p>Search original review text and filter by available metadata.</p></section>
      <form className="filter-panel" onSubmit={search}>
        <label>Review text<input value={filters.query} onChange={(event) => update("query", event.target.value)} placeholder="Search words or phrases" /></label>
        <label>Product ID<select value={filters.product_id} onChange={(event) => update("product_id", event.target.value)}><option value="">All products</option>{catalog.map((item) => <option key={item.product_id} value={item.product_id}>{item.display_name}</option>)}</select></label>
        <label>Predicted sentiment<select value={filters.sentiment} onChange={(event) => update("sentiment", event.target.value)}><option value="">All sentiments</option>{["positive", "negative", "neutral"].map((item) => <option key={item}>{item}</option>)}</select></label>
        <label>Mentioned aspect<select value={filters.aspect} onChange={(event) => update("aspect", event.target.value)}><option value="">All aspects</option>{ASPECTS.map((item) => <option key={item}>{item}</option>)}</select></label>
        <label>Year<select value={filters.year} onChange={(event) => update("year", event.target.value)}><option value="">All years</option>{years.map((year) => <option key={year}>{year}</option>)}</select></label>
        <label>Date precision<select value={filters.date_precision} onChange={(event) => update("date_precision", event.target.value)}><option value="">All precisions</option>{precisions.map((precision) => <option key={precision} value={precision}>{precision.includes("year_only") ? "Year-only" : precision === "day" ? "Day-level" : precision}</option>)}</select></label>
        <button className="button primary" type="submit" disabled={busy}>{busy ? "Searching…" : "Apply filters"}</button>
      </form>
      {error && <Notice tone="error">{error}</Notice>}
      {result && (
        <section className="panel">
          <div className="section-heading">
            <div><span className="eyebrow">Filtered evidence</span><h2>{formatCount(result.total)} matching reviews</h2></div>
            <button className="button secondary" type="button" disabled={!result.items.length} onClick={() => downloadCsv(result.items, "filtered_consumer_tech_reviews.csv")}>Export CSV</button>
          </div>
          {result.total > result.items.length && <Notice>Showing the first {formatCount(result.items.length)} results. Refine filters before exporting.</Notice>}
          <ReviewList reviews={result.items} />
        </section>
      )}
    </>
  );
}

function CsvUpload() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(event) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    const body = new FormData();
    body.append("file", file);
    try {
      setResult(await request("/api/upload", { method: "POST", body }));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <section className="page-intro"><span className="eyebrow">Your review collection</span><h1>Analyze a CSV</h1><p>Upload your own review text for local predictions.</p></section>
      <div className="upload-layout">
        <form className="panel upload-form" onSubmit={submit}>
          <label htmlFor="review-file">Review CSV</label>
          <input id="review-file" type="file" accept=".csv,text/csv" onChange={(event) => setFile(event.target.files?.[0] || null)} />
          <p className="muted">Required: <code>review_text</code>. Product IDs and source/date fields are optional. Maximum 10 MB. Pickle/joblib files are not accepted.</p>
          <button className="button primary" type="submit" disabled={!file || busy}>{busy ? "Analyzing…" : "Analyze reviews"}</button>
        </form>
      </div>
      {error && <Notice tone="error">{error}</Notice>}
      {busy && <LoadingBlock label="Predicting sentiment for uploaded reviews…" />}
      {result && (
        <section className="panel">
          <div className="section-heading"><div><span className="eyebrow">Uploaded collection</span><h2>{formatCount(result.review_count)} reviews analyzed</h2></div><button className="button secondary" type="button" onClick={() => downloadCsv(result.items, "uploaded_reviews_with_predictions.csv")}>Export CSV</button></div>
          <p className="muted">{result.notice}</p>
          <SentimentBars distribution={result.sentiment} />
          <ReviewList reviews={result.items.slice(0, 40)} />
          {result.items.length > 40 && <p className="muted">Showing 40 of {formatCount(result.items.length)} reviews.</p>}
        </section>
      )}
    </>
  );
}

function App() {
  const [section, setSection] = useState("overview");
  const [overview, setOverview] = useState(null);
  const [overviewLoading, setOverviewLoading] = useState(true);
  const [overviewError, setOverviewError] = useState("");
  const [catalog, setCatalog] = useState([]);
  const [metadataOptions, setMetadataOptions] = useState({ brands: [], categories: [] });
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [catalogError, setCatalogError] = useState("");

  useEffect(() => {
    request("/api/overview")
      .then(setOverview)
      .catch((error) => setOverviewError(error.message))
      .finally(() => setOverviewLoading(false));
  }, []);

  useEffect(() => {
    if (!["products", "compare", "reviews"].includes(section) || catalog.length) return;
    setCatalogLoading(true);
    request("/api/products?limit=10000")
      .then((data) => {
        setCatalog(data.items);
        setMetadataOptions({
          brands: data.verified_brands,
          categories: data.verified_categories,
        });
      })
      .catch((error) => setCatalogError(error.message))
      .finally(() => setCatalogLoading(false));
  }, [section, catalog.length]);

  async function loadProducts(query = "", brand = "", category = "") {
    const params = new URLSearchParams({ limit: "1000" });
    if (query.trim()) params.set("query", query.trim());
    if (brand) params.set("brand", brand);
    if (category) params.set("category", category);
    return request(`/api/products?${params.toString()}`);
  }

  const currentLabel = NAVIGATION.find(([key]) => key === section)?.[1] || "Overview";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" onClick={(event) => { event.preventDefault(); setSection("overview"); }}>
          <span className="brand-mark">R</span>
          <span><strong>Review intelligence</strong><small>Consumer technology</small></span>
        </a>
        <nav aria-label="Main navigation">
          {NAVIGATION.map(([key, label, number]) => (
             <button className={section === key ? "nav-item active" : "nav-item"} key={key} onClick={() => setSection(key)}>
               <span className="nav-number">{number}</span><span>{label}</span>
             </button>
           ))}
        </nav>
      </aside>
      <main className="main-content">
        {section !== "overview" && (
          <header className="topbar">
            <span className="breadcrumb">Research workspace <span>/</span> {currentLabel}</span>
            <span className="local-badge">Local analysis</span>
          </header>
        )}
        <div className="page-content" key={section}>
          {catalogLoading && ["products", "compare", "reviews"].includes(section)
            ? <LoadingBlock label="Loading product catalogue…" />
            : catalogError && ["products", "compare", "reviews"].includes(section)
              ? <Notice tone="error">{catalogError}</Notice>
              : section === "overview"
                ? <Overview data={overview} loading={overviewLoading} error={overviewError} />
                : section === "products"
                  ? <ProductExplorer products={catalog.slice(0, 100)} brands={metadataOptions.brands} categories={metadataOptions.categories} loadProducts={loadProducts} apiError={catalogError} />
                  : section === "compare"
                    ? <Comparison catalog={catalog} />
                    : section === "reviews"
                      ? <ReviewExplorer catalog={catalog} overview={overview} />
                      : <CsvUpload />}
        </div>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
