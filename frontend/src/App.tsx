import { useCallback, useEffect, useState } from "react";
import { generateClientSide, SAMPLE_LYRICS, useClientApi } from "./lib/clientPipeline";
import type { GenerateResponse, PipelineResult } from "./types";

const VERSION_COLORS: Record<string, string> = {
  v1: "#6b7280",
  v2: "#8b5cf6",
  v3: "#3b82f6",
  v4: "#ec4899",
  v5: "#f59e0b",
};

function ResultCard({ result }: { result: PipelineResult }) {
  const [showPrompt, setShowPrompt] = useState(false);
  const accent = VERSION_COLORS[result.version] ?? "#888";

  return (
    <article className="result-card" style={{ "--accent": accent } as React.CSSProperties}>
      <header className="card-header">
        <span className="version-badge">{result.version.toUpperCase()}</span>
        <div>
          <h3>{result.label}</h3>
          <p className="card-desc">{result.description}</p>
        </div>
      </header>

      <div className="image-frame">
        <img
          src={`data:image/png;base64,${result.image_base64}`}
          alt={`${result.label} album cover`}
          loading="lazy"
        />
        <span className="backend-tag">{result.image_backend}</span>
      </div>

      <div className="meta-row">
        <span className="meta-chip emotion">{result.primary_emotion}</span>
        <span className="meta-chip genre">{result.primary_genre}</span>
        <span className="meta-chip steps">{result.num_inference_steps} steps</span>
      </div>

      <div className="keywords">
        {result.keywords.slice(0, 6).map((kw) => (
          <span key={kw} className="keyword">{kw}</span>
        ))}
      </div>

      <button
        type="button"
        className="prompt-toggle"
        onClick={() => setShowPrompt((s) => !s)}
      >
        {showPrompt ? "Hide prompt" : "Show prompt"}
      </button>

      {showPrompt && (
        <div className="prompt-box">
          <p className="prompt-label">Positive</p>
          <p className="prompt-text">{result.prompt}</p>
          {result.negative_prompt && (
            <>
              <p className="prompt-label">Negative</p>
              <p className="prompt-text negative">{result.negative_prompt}</p>
            </>
          )}
        </div>
      )}
    </article>
  );
}

export default function App() {
  const [lyrics, setLyrics] = useState("");
  const [seed, setSeed] = useState(42);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<GenerateResponse | null>(null);
  const [progress, setProgress] = useState("");

  useEffect(() => {
    if (useClientApi()) {
      setLyrics(SAMPLE_LYRICS);
      return;
    }
    fetch("/api/sample")
      .then((r) => r.json())
      .then((d) => setLyrics(d.lyrics))
      .catch(() => setLyrics(SAMPLE_LYRICS));
  }, []);

  const generate = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResponse(null);
    setProgress("Running NLP pipelines V1–V5…");

    try {
      let data: GenerateResponse;

      if (useClientApi()) {
        data = await generateClientSide(lyrics, seed);
      } else {
        const res = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ lyrics, seed }),
        });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail ?? `Request failed (${res.status})`);
        }
        data = await res.json();
      }

      setResponse(data);
      setProgress("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed");
      setProgress("");
    } finally {
      setLoading(false);
    }
  }, [lyrics, seed]);

  return (
    <div className="app">
      <div className="bg-gradient" aria-hidden />

      <header className="hero">
        <p className="eyebrow">ArtML Pipeline Comparator</p>
        <h1>LyricVision</h1>
        <p className="subtitle">
          Paste lyrics and generate five album covers — one from each NLP-to-image
          pipeline version (V1 through V5).
        </p>
      </header>

      <section className="input-panel">
        <label htmlFor="lyrics">Lyrics</label>
        <textarea
          id="lyrics"
          value={lyrics}
          onChange={(e) => setLyrics(e.target.value)}
          placeholder="Paste song lyrics here…"
          rows={8}
          disabled={loading}
        />

        <div className="controls">
          <label className="seed-control">
            Seed
            <input
              type="number"
              value={seed}
              min={0}
              onChange={(e) => setSeed(Number(e.target.value))}
              disabled={loading}
            />
          </label>

          <button
            type="button"
            className="generate-btn"
            onClick={generate}
            disabled={loading || lyrics.trim().length < 10}
          >
            {loading ? "Generating…" : "Generate 5 Covers"}
          </button>
        </div>

        {loading && (
          <div className="loading-bar">
            <div className="loading-fill" />
            <p>{progress || "This may take a minute — generating 5 images…"}</p>
          </div>
        )}

        {error && <p className="error">{error}</p>}
      </section>

      {response && (
        <section className="results">
          <div className="results-header">
            <h2>Pipeline Comparison</h2>
            <p>
              Seed {response.seed} · Image backend: {response.nlp_backend}
            </p>
          </div>

          <div className="results-grid">
            {response.results.map((r) => (
              <ResultCard key={r.version} result={r} />
            ))}
          </div>
        </section>
      )}

      <footer className="footer">
        <p>
          V1–V4 from <code>improved_main.ipynb</code> · V5 from{" "}
          <code>final_implementation.ipynb</code>
        </p>
        <p className="footer-credit">
          Made by{" "}
          <a
            href="https://github.com/aarushkandukoori"
            target="_blank"
            rel="noopener noreferrer"
          >
            Aarush Kandukoori
          </a>
        </p>
      </footer>
    </div>
  );
}
