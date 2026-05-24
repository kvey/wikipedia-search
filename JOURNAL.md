Journal

* Initial run was just skipping using the tool call because Opus wants to just use its own knowledge
so we force use of at least one tool call
* First implementation was just confirming simple questions
* Added evals for multiple models
* Added counterfactual evals
* Added matplotlib visualization of evals
* Everything was just passing the evals immediately
* Added more evals that are more complicated, such as questions that require multiple queries, or intentionally ambiguous questions
* Added additional eval categories such as search utilization, how grounded the answer is in facts, and refusals
* Manually reviewing some traces
* Added parallelism to our evals
* In checking our query lengths, then checking if specific factual data required in the final outputs given is present, determined the models are often coming up with their answers without grounding in the search - revising our evals to perform ablation for comparison
* Some small set of rate limit failures negatively effected our Sonnet scores - (we didn't jitter our retries)
* Adding an additional eval that checks coverage of entities in final result in the search results, due to saving our traces we're able to do this retroactively
* Trace mining showed two grounding gaps: single queries that miss (the agent re-queries sequentially across turns — "Nobel Prize in Mathematics 2000" never converged and hit max-turns), and thin extracts (only the top hit got a lead extract, so a fact in hit #2/#3 forced an extra turn — e.g. the Nile in the Sahara case)
* In response: `search_wikipedia` now accepts multiple query phrasings and fans out (merge + dedupe by title, best rank wins), extracting the top 2 merged hits; added a `get_article` tool to pull a fuller body / a named section when snippets are too thin
* To keep the `search` dimension honest, it now counts search *steps* (search_wikipedia calls), not total tool calls — a fan-out call is one step and get_article doesn't count, so multi-hop cases still demand genuine chaining (a clean single-fan-out on a 2-hop case scores search=0.5 but still passes on grounding)
* Result (full sweep vs prior baseline): grounding rose for every model — opus 0.77→0.90, sonnet 0.84→0.88, haiku 0.90→0.95; 37/37 pass across all three, no max-turns failures
* Re-running the retroactive entity_coverage check exposed a metric flaw: the literal token "Wikipedia" (from the "Sources: Wikipedia articles …" citation footer) was the #1 flagged "uncovered entity" (111/116 traces) — boilerplate, not hallucination. Fixed by stripping the Sources/Citations footer and stoplisting "Wikipedia" before spaCy NER; mean entity_coverage on the same traces went 0.684→0.802, and the remaining flags are now genuine (number/date formatting, real recall)
* Wrote up the before/after in docs/EVAL_REPORT.md §8
*  
