"""
UFO RAG Evaluation Pipeline
============================
Evaluates LightRAG chatbot quality across multiple dimensions:
  1. Response Latency       - How fast the system answers (seconds)
  2. ROUGE-L Score          - N-gram overlap with reference answers
  3. Semantic Similarity    - Cosine similarity via embedding model
  4. Faithfulness Score     - LLM-as-judge (0-5) using Ollama
  5. Answer Length          - Completeness proxy (word count)
  6. Mode Comparison        - naive vs local vs global vs hybrid

Output:
  - output/eval_results/eval_results.csv  (raw data)
  - output/eval_results/eval_report.html  (visual report)
"""

import json
import time
import math
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────
LIGHTRAG_BASE_URL = "http://localhost:9621"
OLLAMA_BASE_URL   = "http://localhost:11434"
OLLAMA_LLM_MODEL  = "qwen2.5:7b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

BASE_DIR     = Path(r"D:\Project\UFO_BOT\UFO-BOT")
QUESTIONS_PATH = BASE_DIR / "evaluation" / "eval_questions.json"
OUTPUT_DIR   = BASE_DIR / "output" / "eval_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH  = OUTPUT_DIR / "eval_results.csv"
HTML_PATH = OUTPUT_DIR / "eval_report.html"
LOG_PATH  = OUTPUT_DIR / "eval_pipeline.log"

# Modes to evaluate
EVAL_MODES = ["naive", "local", "global", "hybrid"]

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─── ROUGE-L Implementation (no extra dependencies) ───────────────────────────
def _lcs_length(x: list, y: list) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between hypothesis and reference."""
    h_tokens = hypothesis.lower().split()
    r_tokens = reference.lower().split()
    if not h_tokens or not r_tokens:
        return 0.0
    lcs = _lcs_length(h_tokens, r_tokens)
    precision = lcs / len(h_tokens) if h_tokens else 0.0
    recall    = lcs / len(r_tokens) if r_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ─── API Helpers ──────────────────────────────────────────────────────────────
def query_lightrag(question: str, mode: str, timeout: int = 120) -> tuple[str, float]:
    """Query LightRAG and return (answer, latency_seconds)."""
    url = f"{LIGHTRAG_BASE_URL}/query"
    payload = {"query": question, "mode": mode}
    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        latency = time.time() - start
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("response", data.get("answer", ""))
        return str(answer).strip(), round(latency, 2)
    except Exception as e:
        latency = time.time() - start
        log.warning(f"LightRAG query failed ({mode}): {e}")
        return "", round(latency, 2)


def get_embedding(text: str) -> list[float] | None:
    """Get embedding vector from Ollama."""
    url = f"{OLLAMA_BASE_URL}/api/embed"
    try:
        resp = requests.post(
            url,
            json={"model": OLLAMA_EMBED_MODEL, "input": text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if embeddings:
            return embeddings[0]
    except Exception as e:
        log.warning(f"Embedding failed: {e}")
    return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def llm_faithfulness_score(question: str, answer: str) -> int:
    """
    Ask Ollama to score faithfulness of the answer on a 0-5 scale.
    Returns integer score. Falls back to -1 on error.
    """
    if not answer:
        return 0

    prompt = f"""You are an expert evaluator for a UFO/UAP knowledge base chatbot.
Rate the following answer on a scale of 0-5 based on these criteria:
  5 - Excellent: Accurate, detailed, directly answers the question with specific facts
  4 - Good: Mostly accurate and relevant with minor gaps
  3 - Adequate: Partially answers but lacks key details
  2 - Poor: Vague or tangential, missing most key information
  1 - Very Poor: Mostly irrelevant or incorrect
  0 - No Answer: Empty or completely off-topic

Question: {question}
Answer: {answer}

Respond with ONLY a single digit (0-5). No explanation."""

    url = f"{OLLAMA_BASE_URL}/api/generate"
    try:
        resp = requests.post(
            url,
            json={"model": OLLAMA_LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        # Extract first digit found
        for ch in raw:
            if ch.isdigit():
                score = int(ch)
                return min(5, max(0, score))
    except Exception as e:
        log.warning(f"Faithfulness scoring failed: {e}")
    return -1


# ─── Main Evaluation ──────────────────────────────────────────────────────────
def evaluate():
    log.info("=" * 65)
    log.info("UFO BOT — RAG Evaluation Pipeline")
    log.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 65)

    # Load questions
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    log.info(f"Loaded {len(questions)} evaluation questions")
    log.info(f"Evaluating modes: {EVAL_MODES}")

    results = []

    for q_idx, q in enumerate(questions):
        q_id       = q["id"]
        question   = q["question"]
        reference  = q["reference_answer"]
        category   = q["category"]
        difficulty = q["difficulty"]

        log.info(f"\n{'─'*60}")
        log.info(f"[{q_idx+1}/{len(questions)}] {q_id}: {question[:60]}...")

        # Pre-compute reference embedding once per question
        ref_embedding = get_embedding(reference)
        q_embedding   = get_embedding(question)

        for mode in EVAL_MODES:
            log.info(f"  Mode: [{mode}]")

            # 1. Query LightRAG
            answer, latency = query_lightrag(question, mode)

            if not answer:
                log.warning(f"    → Empty answer, skipping metrics")
                results.append({
                    "question_id": q_id,
                    "category": category,
                    "difficulty": difficulty,
                    "mode": mode,
                    "question": question,
                    "answer": "(no response)",
                    "reference": reference,
                    "latency_sec": latency,
                    "rouge_l": 0.0,
                    "semantic_similarity": 0.0,
                    "faithfulness_score": 0,
                    "answer_word_count": 0,
                })
                continue

            # 2. ROUGE-L
            r_score = rouge_l(answer, reference)

            # 3. Semantic Similarity
            ans_embedding = get_embedding(answer)
            sem_sim = 0.0
            if ans_embedding and ref_embedding:
                sem_sim = cosine_similarity(ans_embedding, ref_embedding)

            # 4. Faithfulness (LLM-as-judge)
            faith_score = llm_faithfulness_score(question, answer)

            # 5. Answer word count
            word_count = len(answer.split())

            log.info(
                f"    → Latency: {latency}s | ROUGE-L: {r_score:.3f} | "
                f"SemSim: {sem_sim:.3f} | Faithfulness: {faith_score}/5 | Words: {word_count}"
            )

            results.append({
                "question_id": q_id,
                "category": category,
                "difficulty": difficulty,
                "mode": mode,
                "question": question,
                "answer": answer,
                "reference": reference,
                "latency_sec": latency,
                "rouge_l": round(r_score, 4),
                "semantic_similarity": round(sem_sim, 4),
                "faithfulness_score": faith_score,
                "answer_word_count": word_count,
            })

    # ── Save CSV ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    log.info(f"\n[DONE] Raw results saved: {CSV_PATH}")

    # ── Generate HTML Report ──────────────────────────────────────────────────
    generate_html_report(df)
    log.info(f"[DONE] HTML report saved: {HTML_PATH}")

    # ── Print Summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("EVALUATION SUMMARY (per mode)")
    log.info("=" * 65)
    summary = df.groupby("mode").agg(
        avg_latency=("latency_sec", "mean"),
        avg_rouge_l=("rouge_l", "mean"),
        avg_semantic_sim=("semantic_similarity", "mean"),
        avg_faithfulness=("faithfulness_score", lambda x: x[x >= 0].mean()),
        avg_word_count=("answer_word_count", "mean"),
    ).round(3)
    print(summary.to_string())


# ─── HTML Report Generator ────────────────────────────────────────────────────
def generate_html_report(df: pd.DataFrame):
    # Aggregate summary by mode
    summary = df.groupby("mode").agg(
        avg_latency=("latency_sec", "mean"),
        avg_rouge_l=("rouge_l", "mean"),
        avg_semantic_sim=("semantic_similarity", "mean"),
        avg_faithfulness=("faithfulness_score", lambda x: x[x >= 0].mean()),
        avg_words=("answer_word_count", "mean"),
        total_questions=("question_id", "count"),
    ).round(3).reset_index()

    # Best mode per metric
    best_mode_rouge    = summary.loc[summary["avg_rouge_l"].idxmax(), "mode"]
    best_mode_sem      = summary.loc[summary["avg_semantic_sim"].idxmax(), "mode"]
    best_mode_faith    = summary.loc[summary["avg_faithfulness"].idxmax(), "mode"]
    fastest_mode       = summary.loc[summary["avg_latency"].idxmin(), "mode"]

    # Build summary table rows
    def badge(mode):
        colors = {"naive": "#6c757d", "local": "#0d6efd", "global": "#198754", "hybrid": "#dc3545"}
        return f'<span style="background:{colors.get(mode,"#333")};color:#fff;padding:2px 10px;border-radius:20px;font-size:0.8em">{mode}</span>'

    summary_rows = ""
    for _, row in summary.iterrows():
        summary_rows += f"""
        <tr>
            <td>{badge(row['mode'])}</td>
            <td>{row['avg_latency']:.2f}s</td>
            <td>{row['avg_rouge_l']:.4f}</td>
            <td>{row['avg_semantic_sim']:.4f}</td>
            <td>{'N/A' if math.isnan(row['avg_faithfulness']) else f"{row['avg_faithfulness']:.2f}/5"}</td>
            <td>{row['avg_words']:.0f}</td>
        </tr>"""

    # Build per-question detail rows
    detail_rows = ""
    for _, row in df.iterrows():
        faith_display = "N/A" if row["faithfulness_score"] < 0 else f"{row['faithfulness_score']}/5"
        answer_short  = (row["answer"][:200] + "...") if len(str(row["answer"])) > 200 else row["answer"]
        detail_rows += f"""
        <tr>
            <td><code>{row['question_id']}</code></td>
            <td>{badge(row['mode'])}</td>
            <td style="font-size:0.85em">{row['question'][:60]}...</td>
            <td>{row['latency_sec']:.2f}s</td>
            <td>{row['rouge_l']:.4f}</td>
            <td>{row['semantic_similarity']:.4f}</td>
            <td>{faith_display}</td>
            <td>{row['answer_word_count']}</td>
        </tr>"""

    # Chart data (for Chart.js)
    modes_js     = json.dumps(list(summary["mode"]))
    rouge_js     = json.dumps(list(summary["avg_rouge_l"]))
    sem_js       = json.dumps(list(summary["avg_semantic_sim"]))
    faith_js     = json.dumps([float(x) if not math.isnan(x) else 0 for x in summary["avg_faithfulness"]])
    latency_js   = json.dumps(list(summary["avg_latency"]))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>UFO BOT — RAG Evaluation Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0a0a14;
      --surface: #12121f;
      --surface2: #1a1a2e;
      --border: #2a2a4a;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #6366f1;
      --green: #10b981;
      --yellow: #f59e0b;
      --red: #ef4444;
      --blue: #3b82f6;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 40px 24px;
    }}

    /* ── Header ── */
    .header {{
      text-align: center;
      margin-bottom: 48px;
    }}
    .header h1 {{
      font-size: 2.4rem;
      font-weight: 700;
      background: linear-gradient(135deg, #818cf8, #6366f1, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }}
    .header p {{
      color: var(--muted);
      font-size: 1rem;
    }}
    .header .timestamp {{
      display: inline-block;
      margin-top: 8px;
      padding: 4px 14px;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 20px;
      font-size: 0.8rem;
      color: var(--muted);
    }}

    /* ── KPI Cards ── */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 40px;
    }}
    .kpi-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px 20px;
      text-align: center;
      position: relative;
      overflow: hidden;
    }}
    .kpi-card::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, var(--accent), #a78bfa);
    }}
    .kpi-value {{
      font-size: 2rem;
      font-weight: 700;
      color: var(--accent);
      margin-bottom: 4px;
    }}
    .kpi-label {{
      font-size: 0.78rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    /* ── Charts ── */
    .charts-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 40px;
    }}
    @media (max-width: 768px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
    .chart-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
    }}
    .chart-card h3 {{
      font-size: 0.9rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 16px;
    }}
    canvas {{ max-height: 260px; }}

    /* ── Tables ── */
    .section {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 28px;
      margin-bottom: 32px;
    }}
    .section h2 {{
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 20px;
      color: var(--text);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
    }}
    th {{
      background: var(--surface2);
      color: var(--muted);
      text-transform: uppercase;
      font-size: 0.72rem;
      letter-spacing: 0.05em;
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    td {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      color: var(--text);
      vertical-align: top;
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: var(--surface2); }}

    /* ── Footer ── */
    .footer {{
      text-align: center;
      color: var(--muted);
      font-size: 0.8rem;
      margin-top: 48px;
    }}
  </style>
</head>
<body>

<div class="header">
  <h1>🛸 UFO BOT — RAG Evaluation Report</h1>
  <p>LightRAG Knowledge Graph Chatbot · UAP/UFO Document Corpus</p>
  <span class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
</div>

<!-- KPI Cards -->
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-value">{len(df['question_id'].unique())}</div>
    <div class="kpi-label">Test Questions</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{len(EVAL_MODES)}</div>
    <div class="kpi-label">Query Modes Tested</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{best_mode_rouge}</div>
    <div class="kpi-label">Best ROUGE-L Mode</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{best_mode_faith}</div>
    <div class="kpi-label">Best Faithfulness Mode</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{fastest_mode}</div>
    <div class="kpi-label">Fastest Mode</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{df['answer_word_count'].mean():.0f}</div>
    <div class="kpi-label">Avg Answer Length (words)</div>
  </div>
</div>

<!-- Charts -->
<div class="charts-grid">
  <div class="chart-card">
    <h3>ROUGE-L Score by Mode</h3>
    <canvas id="chartRouge"></canvas>
  </div>
  <div class="chart-card">
    <h3>Semantic Similarity by Mode</h3>
    <canvas id="chartSem"></canvas>
  </div>
  <div class="chart-card">
    <h3>Faithfulness Score by Mode (LLM Judge)</h3>
    <canvas id="chartFaith"></canvas>
  </div>
  <div class="chart-card">
    <h3>Average Response Latency (seconds)</h3>
    <canvas id="chartLatency"></canvas>
  </div>
</div>

<!-- Summary Table -->
<div class="section">
  <h2>📊 Mode Comparison Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Mode</th>
        <th>Avg Latency</th>
        <th>ROUGE-L</th>
        <th>Semantic Sim.</th>
        <th>Faithfulness</th>
        <th>Avg Words</th>
      </tr>
    </thead>
    <tbody>
      {summary_rows}
    </tbody>
  </table>
</div>

<!-- Detailed Results -->
<div class="section">
  <h2>🔍 Per-Question Results</h2>
  <div style="overflow-x:auto">
  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Mode</th>
        <th>Question</th>
        <th>Latency</th>
        <th>ROUGE-L</th>
        <th>Sem. Sim.</th>
        <th>Faithfulness</th>
        <th>Words</th>
      </tr>
    </thead>
    <tbody>
      {detail_rows}
    </tbody>
  </table>
  </div>
</div>

<div class="footer">
  UFO BOT RAG Evaluation Pipeline · LightRAG + Ollama · RTX 3060
</div>

<script>
const MODE_COLORS = {{
  naive:  '#6c757d',
  local:  '#3b82f6',
  global: '#10b981',
  hybrid: '#f59e0b',
}};
const modes   = {modes_js};
const bgColors = modes.map(m => MODE_COLORS[m] || '#6366f1');

const commonOpts = {{
  responsive: true,
  plugins: {{
    legend: {{ display: false }},
  }},
  scales: {{
    x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1a1a2e' }} }},
    y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1a1a2e' }} }},
  }},
}};

new Chart(document.getElementById('chartRouge'), {{
  type: 'bar',
  data: {{ labels: modes, datasets: [{{ data: {rouge_js}, backgroundColor: bgColors, borderRadius: 8 }}] }},
  options: {{ ...commonOpts, scales: {{ ...commonOpts.scales, y: {{ ...commonOpts.scales.y, min: 0, max: 1 }} }} }},
}});

new Chart(document.getElementById('chartSem'), {{
  type: 'bar',
  data: {{ labels: modes, datasets: [{{ data: {sem_js}, backgroundColor: bgColors, borderRadius: 8 }}] }},
  options: {{ ...commonOpts, scales: {{ ...commonOpts.scales, y: {{ ...commonOpts.scales.y, min: 0, max: 1 }} }} }},
}});

new Chart(document.getElementById('chartFaith'), {{
  type: 'bar',
  data: {{ labels: modes, datasets: [{{ data: {faith_js}, backgroundColor: bgColors, borderRadius: 8 }}] }},
  options: {{ ...commonOpts, scales: {{ ...commonOpts.scales, y: {{ ...commonOpts.scales.y, min: 0, max: 5 }} }} }},
}});

new Chart(document.getElementById('chartLatency'), {{
  type: 'bar',
  data: {{ labels: modes, datasets: [{{ data: {latency_js}, backgroundColor: bgColors, borderRadius: 8 }}] }},
  options: commonOpts,
}});
</script>
</body>
</html>"""

    HTML_PATH.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    evaluate()
