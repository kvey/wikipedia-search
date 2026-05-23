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
*  
