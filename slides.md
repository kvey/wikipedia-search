---
theme: default
title: Wikipedia Research Agent - Eval-Driven Iteration
info: |
  Five-minute self-recording deck for the Wikipedia research agent take-home.
class: deck
fonts:
  sans: Inter
  serif: Fraunces
  mono: IBM Plex Mono
  weights: '300,400,500,600,700'
drawings:
  persist: false
transition: fade
mdc: true
---

<div class="canvas cover">
  <div class="cover-body">
    <span class="eyebrow">Eval-driven iteration</span>
    <h1>Wikipedia Research Agent</h1>
    <p class="subtitle">Measuring <em>grounding</em> — not just answer correctness — so every claim traces back to retrieved evidence.</p>
    <p class="cover-result"><b>Grounding improved on all three models</b> — Opus 0.77 &rarr; 0.90 — from one trace-driven tool fix.</p>
  </div>
  <div class="ledger">
    <div><b>37</b><span>eval cases</span></div>
    <div><b>8</b><span>case categories</span></div>
    <div><b>4</b><span>scored dimensions</span></div>
    <div><b>3</b><span>model sweep</span></div>
  </div>
  <footer class="footrule">Prompt &middot; design &middot; evals &middot; learning &middot; iteration &middot; result</footer>
</div>

<!--
Timing: 25s. Open by saying the target was not a trivia bot; it was a research agent whose answers should be traceable to retrieved Wikipedia evidence.
-->

---
title: Design rationale
---

<div class="canvas">
  <header class="kicker"><span class="num">01</span><span class="eyebrow">Design rationale</span></header>
  <h2>Keep the agent inspectable, then make every run measurable.</h2>
  <div class="two-col">
    <section class="col">
      <h3>Small control loop</h3>
      <p>Built directly on the Anthropic API, no agent framework. The model calls Wikipedia tools and returns a final answer.</p>
    </section>
    <section class="col">
      <h3>Trace everything</h3>
      <p>Each run records model turns, tool calls, retrieved text, and the final answer as JSON traces for later analysis.</p>
    </section>
  </div>
  <div class="flow">
    <span>Question</span><span>Model loop</span><span>Wikipedia tools</span><span>Trace + eval</span>
  </div>
  <p class="callout">Early runs showed Opus could skip search and answer from memory — so the agent now forces at least one tool call.</p>
</div>

<!--
Timing: 40s. Emphasize observability. Traces were the key design choice because they enabled retroactive grading after the metric changed.
-->

---
title: Eval design
---

<div class="canvas">
  <header class="kicker"><span class="num">02</span><span class="eyebrow">Eval design</span></header>
  <h2>A pass/fail trivia test was not enough.</h2>
  <div class="two-col wide-left">
    <section class="col">
      <h3>37 cases &middot; 8 categories</h3>
      <div class="chips"><span>factual</span><span>false premise</span><span>unanswerable</span><span>contrastive</span><span>multi-hop</span><span>computation</span><span>comparison</span><span>disambiguation</span></div>
    </section>
    <section class="col">
      <h3>Four scored dimensions</h3>
      <dl class="dimensions"><dt>answer</dt><dd>fact constraints met</dd><dt>search</dt><dd>appropriate search effort</dd><dt>grounding</dt><dd>claims supported by retrieved text</dd><dt>calibration</dt><dd>hedge or refuse when unsupported</dd></dl>
    </section>
  </div>
  <p class="callout">Overall score is the mean of the four dimensions; the pass threshold is <strong>0.7</strong>.</p>
</div>

<!--
Timing: 40s. Explain that the eval suite evolved to expose why an answer passed, especially whether it was grounded.
-->

---
title: First signal
---

<div class="canvas">
  <header class="kicker"><span class="num">03</span><span class="eyebrow">First signal</span></header>
  <h2>The dashboard made grounding the weak spot visible.</h2>
  <figure class="exhibit fill"><img src="./eval_results/2026-05-22T18-52-16_07b1e15-dirty/dashboard.png" alt="Baseline eval dashboard" /></figure>
  <p class="caption"><b>Run 2, 37 cases —</b> answer scores were high, but grounding lagged for every model: Opus <b>0.773</b>, Sonnet <b>0.842</b>, Haiku <b>0.903</b>.</p>
</div>

<!--
Timing: 40s. This is the central learning. The models were mostly right, but the answers were not always fully supported by retrieved evidence.
-->

---
title: Prompt approach
---

<div class="canvas">
  <header class="kicker"><span class="num">04</span><span class="eyebrow">Prompt approach</span></header>
  <h2>Prompt for evidence-seeking behavior, then let evals police it.</h2>
  <div class="prompt-layout">
    <div class="prompt-card">
      <span>You are a research assistant that answers factual questions using Wikipedia.</span>
      <span class="gap">Guidelines:</span>
      <span>- When a question depends on facts you are not certain of,</span>
      <span class="indent">call the search_wikipedia tool before answering. <mark class="hl">Prefer searching over guessing<b class="hl-n">1</b></mark>.</span>
      <span>- search_wikipedia accepts several queries at once. When the entity is</span>
      <span class="indent">ambiguous or could be phrased multiple ways, <mark class="hl">pass several phrasings<b class="hl-n">2</b></mark> in a single call.</span>
      <span>- If a result's title or snippet looks like it holds the answer but the</span>
      <span class="indent">returned extract doesn't contain the specific fact, <mark class="hl">call get_article<b class="hl-n">3</b></mark>.</span>
      <span>- Base your answer on the retrieved text. If it does not contain the answer,</span>
      <span class="indent"><mark class="hl">say so plainly<b class="hl-n">4</b></mark> rather than inventing details.</span>
      <span>- Answer concisely and directly. Cite the Wikipedia article title(s) you relied on.</span>
    </div>
    <section class="prompt-notes">
      <div><span class="note-num">1</span><b>Force retrieval bias</b><span>Targets the closed-book behavior the first traces exposed.</span></div>
      <div><span class="note-num">2</span><b>Make search robust</b><span>Fan-out phrasing came from missed-query failures in multi-hop cases.</span></div>
      <div><span class="note-num">3</span><b>Escalate to detail</b><span>Addresses thin extracts where snippets looked relevant but lacked support.</span></div>
      <div><span class="note-num">4</span><b>Calibrate unsupported answers</b><span>Maps directly to the calibration and grounding dimensions.</span></div>
    </section>
  </div>
</div>

<!--
Timing: 30s. This is the prompt-engineering answer: the prompt encodes behavior, but the important choice was to make those behaviors measurable rather than relying on instructions alone.
-->

---
title: Ablation
---

<div class="canvas">
  <header class="kicker"><span class="num">05</span><span class="eyebrow">Ablation</span></header>
  <h2>Without tools, answer stayed high while overall collapsed.</h2>
  <div class="compare">
    <section class="metric">
      <h3>With tools</h3>
      <div class="big-number good">0.88</div>
      <p>average overall across models</p>
    </section>
    <div class="versus">vs</div>
    <section class="metric">
      <h3>No tools</h3>
      <div class="big-number warn">0.48</div>
      <p>average overall across models</p>
    </section>
  </div>
  <p class="insight"><b>Key point —</b> answer scores stayed around <strong>0.97</strong> with no tools. Retrieval improved evidence support, not raw recall.</p>
</div>

<!--
Timing: 35s. The no-tool arm proved that many questions were memorized. Search and grounding dimensions are what made the distinction visible.
-->

---
title: Iteration
---

<div class="canvas">
  <header class="kicker"><span class="num">06</span><span class="eyebrow">Iteration</span></header>
  <h2>Trace mining pointed to a tool-layer fix.</h2>
  <div class="two-col">
    <section class="col">
      <h3>Observed gaps</h3>
      <ul><li>One missed query caused slow sequential retries.</li><li>Useful facts often appeared in hit 2 or 3, not the top extract.</li></ul>
    </section>
    <section class="col">
      <h3>Change made</h3>
      <ul><li><code>search_wikipedia(queries[])</code> fans out phrasings.</li><li><code>get_article(title, section?)</code> fetches deeper detail.</li></ul>
    </section>
  </div>
  <div class="flow tight">
    <span>fan out</span><span>merge + dedupe</span><span>extract top 2</span><span>fetch detail</span>
  </div>
  <p class="caption">Search scoring now counts search <em>steps</em>, not every tool call — so a fan-out is one search step.</p>
</div>

<!--
Timing: 40s. Stress that this was model-agnostic. The traces identified mechanical retrieval failures, so the fix lives in the tool surface.
-->

---
title: Targeted result
---

<div class="canvas">
  <header class="kicker"><span class="num">07</span><span class="eyebrow">Targeted result</span></header>
  <h2>Grounding improved across all models.</h2>
  <div class="result-layout">
    <figure class="exhibit"><img src="./eval_results/2026-05-22T20-06-24_073937c-dirty/dashboard.png" alt="Post-change eval dashboard" /></figure>
    <div class="result-stats">
      <div><span class="lbl">Opus grounding</span><b>0.773 <i>→</i> 0.903</b></div>
      <div><span class="lbl">Sonnet grounding</span><b>0.842 <i>→</i> 0.884</b></div>
      <div><span class="lbl">Haiku grounding</span><b>0.903 <i>→</i> 0.953</b></div>
      <div class="hl"><span class="lbl">All models</span><b>37 / 37 pass</b></div>
    </div>
  </div>
</div>

<!--
Timing: 45s. Before and after used the same suite, judge, and threshold. Note the caveat that search score changed meaning after counting search steps.
-->

---
title: Final read
---

<div class="canvas">
  <header class="kicker"><span class="num">08</span><span class="eyebrow">Final read</span></header>
  <h2>The evals made the failure mode specific enough to fix.</h2>
  <div class="two-col">
    <section class="col">
      <h3>Where it succeeds</h3>
      <ul><li>Small raw-API loop stayed inspectable.</li><li>All models reached 37 / 37 passing after the tool fix.</li><li>Ablations separated recall from retrieval.</li></ul>
    </section>
    <section class="col">
      <h3>Where it still fails</h3>
      <figure class="exhibit entity-chart"><img src="./eval_results/entity_coverage_2026-05-22T19-37-10_07b1e15-dirty/entity_coverage_retro.png" alt="Entity coverage retroactive chart" /></figure>
      <p>Grounding checks still miss aliases and number formats, and the suite contains facts frontier models already know.</p>
    </section>
  </div>
  <p class="callout"><b>With more time —</b> link each result to its trace, normalize aliases and numbers, add fresher / less-memorized facts, track cost and latency, and weight dimensions by product risk. <b>Approx. time spent:</b> ~2 hours.</p>
</div>

<!--
Timing: 35s. Close with the thesis: the meaningful improvement was grounding, and the eval design made it actionable. Adjust the time-spent number if your actual total differs.
-->
