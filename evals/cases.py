"""Evaluation dataset for the Wikipedia agent.

Each case is a question graded along **four dimensions of correctness** (see
`evals/grading.py`), each scored 0-1 and averaged into an overall score:

- ``answer`` — keyword matching: the answer should contain all of `must_include`
  (case-insensitive substrings), at least one of `must_include_any` (if set),
  and none of `must_exclude`.
- ``search`` — *how much it searched*: the tool-call count should fall in
  `[min_searches, max_searches]`. `min_searches` defaults to 1 when
  `expect_tool_use` is set; multi-hop cases raise it. `max_searches` flags
  over-searching when set.
- ``grounding`` — *how well the answer is supported by what Wikipedia actually
  returned*: an LLM judge rates the answer's claims against the retrieved
  passages (0-1 + rationale), catching facts recalled from parametric memory
  rather than the sources.
- ``calibration`` — *qualifying when it can't find information*: cases where the
  honest answer is "it doesn't exist / I can't find it" set `expect_refusal` and
  must hedge accordingly; answerable cases instead penalize a *false* abstention.

`must_include_any` exists because corrections and refusals vary in wording, so a
single required substring is too brittle for those cases.

Beyond plain factual recall, the suite includes *counterfactual* and *reasoning*
cases that test robustness, tagged via `category`:

- ``false_premise`` — the question embeds a wrong fact; the agent should correct
  it rather than echo it.
- ``unanswerable`` — the thing does not exist; the agent should search, then say
  it cannot find it rather than hallucinate an answer.
- ``contrastive`` — a near-duplicate of a factual case with one detail changed,
  to catch shallow keyword/pattern matching.
- ``multi_hop`` — the answer requires chaining two or more lookups (e.g. resolve
  an entity, then look up a property of it), often with a tempting wrong answer
  one hop short.
- ``computation`` — the agent must retrieve facts and then do arithmetic on
  them rather than copy a single value.
- ``disambiguation`` — the question names an entity that collides with a more
  famous namesake; the agent must resolve the reference to the intended one.
- ``comparison`` — two named entities are compared on some dimension (size,
  age, temperature, ...); the agent must look up both and pick correctly.
"""

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    name: str
    question: str
    must_include: list[str] = field(default_factory=list)
    must_include_any: list[str] = field(default_factory=list)
    must_exclude: list[str] = field(default_factory=list)
    expect_tool_use: bool = True
    category: str = "factual"
    # --- search-behavior expectations (the `search` dimension) -------------
    # How much the agent should search. `min_searches` defaults to 1 when
    # `expect_tool_use` is set (every answer should be grounded in at least one
    # lookup); raise it for multi-hop questions that genuinely need chaining.
    # `max_searches`, when set, flags wasteful over-searching.
    min_searches: int | None = None
    max_searches: int | None = None
    # --- abstention expectation (the `calibration` dimension) --------------
    # True when the honest answer is "I can't find / it doesn't exist": the
    # agent should explicitly qualify the gap rather than fabricate. For
    # answerable cases (the default) calibration instead penalizes a *false*
    # abstention. Defaults to True for the `unanswerable` category.
    expect_refusal: bool | None = None

    def __post_init__(self) -> None:
        if self.min_searches is None:
            self.min_searches = 1 if self.expect_tool_use else 0
        if self.expect_refusal is None:
            self.expect_refusal = self.category == "unanswerable"


CASES: list[EvalCase] = [
    # --- Factual recall -----------------------------------------------------
    EvalCase(
        name="australia_capital",
        question="What is the capital of Australia?",
        must_include=["Canberra"],
        must_exclude=["Sydney is the capital"],
        category="factual",
    ),
    EvalCase(
        name="periodic_table_author",
        question="Who is credited with creating the periodic table of elements?",
        must_include=["Mendeleev"],
        category="factual",
    ),
    EvalCase(
        name="speed_of_light",
        question="What is the approximate speed of light in a vacuum, in km/s?",
        must_include=["299,792"],
        category="factual",
    ),
    EvalCase(
        name="tallest_mountain",
        question="What is the tallest mountain above sea level on Earth?",
        must_include=["Everest"],
        category="factual",
    ),
    EvalCase(
        name="dna_structure",
        question="Which two scientists are most famous for describing the "
        "double-helix structure of DNA in 1953?",
        must_include=["Watson", "Crick"],
        category="factual",
    ),
    EvalCase(
        name="moon_landing_year",
        question="In what year did humans first land on the Moon?",
        must_include=["1969"],
        category="factual",
    ),
    # --- Counterfactual: false premise --------------------------------------
    # The question asserts something untrue; the agent should correct it rather
    # than answer as if the premise held.
    EvalCase(
        name="einstein_nobel",
        question="In what year did Albert Einstein win the Nobel Prize for his "
        "theory of relativity?",
        must_include=["photoelectric"],
        must_include_any=["1921"],
        category="false_premise",
    ),
    EvalCase(
        name="great_wall_from_moon",
        question="Why is the Great Wall of China visible from the Moon with the "
        "naked eye?",
        must_include_any=[
            "not visible",
            "cannot be seen",
            "can't be seen",
            "is a myth",
            "not true",
            "isn't visible",
        ],
        must_exclude=["is visible from the Moon"],
        category="false_premise",
    ),
    EvalCase(
        name="franklin_president",
        question="When did Benjamin Franklin serve as President of the United "
        "States?",
        must_include_any=[
            "never",
            "was not",
            "did not serve",
            "no president",
            "not a president",
        ],
        category="false_premise",
    ),
    # --- Counterfactual: unanswerable / nonexistent -------------------------
    # The subject does not exist. The agent should still search to confirm, then
    # say it cannot find an answer rather than fabricate one.
    EvalCase(
        name="sixtieth_president",
        question="Who was the 60th President of the United States?",
        must_include_any=[
            "no",
            "has not",
            "hasn't",
            "does not exist",
            "only",
        ],
        category="unanswerable",
    ),
    EvalCase(
        name="unobtainium_atomic_number",
        question="What is the atomic number of the element unobtainium?",
        must_include_any=[
            "fictional",
            "not a real",
            "not an element",
            "does not exist",
            "no such element",
        ],
        category="unanswerable",
    ),
    # --- Counterfactual: contrastive minimal pairs --------------------------
    # One detail changed from a factual case above; the answer differs, so a
    # shallow keyword/pattern match would get these wrong.
    EvalCase(
        name="australia_largest_city",
        question="What is the largest city in Australia by population?",
        must_include=["Sydney"],
        must_exclude=["Canberra is the largest"],
        category="contrastive",
    ),
    EvalCase(
        name="tallest_mountain_base_to_summit",
        question="What is the tallest mountain on Earth measured from base to "
        "summit, rather than above sea level?",
        must_include=["Mauna Kea"],
        category="contrastive",
    ),
    EvalCase(
        name="second_moonwalker",
        question="Who was the second person to walk on the Moon?",
        must_include=["Aldrin"],
        category="contrastive",
    ),
    EvalCase(
        name="speed_of_sound",
        question="What is the approximate speed of sound in air at sea level, "
        "in meters per second?",
        # Contrasts with `speed_of_light`: same phrasing, very different number.
        # Accepts the common 0 C / 15 C / 20 C textbook values.
        must_include_any=["343", "340", "331"],
        must_exclude=["299,792"],
        category="contrastive",
    ),
    # --- Multi-hop reasoning ------------------------------------------------
    # The answer requires resolving one entity, then looking up a property of
    # it. Each has a plausible "one hop short" trap answer.
    EvalCase(
        name="olympics_2016_host_capital",
        question="What is the capital of the country that hosted the 2016 "
        "Summer Olympics?",
        # Rio de Janeiro hosted, but Brazil's capital is Brasilia.
        must_include_any=["Brasília", "Brasilia"],
        must_exclude=["capital is Rio", "Rio de Janeiro is the capital"],
        category="multi_hop",
        min_searches=2,  # genuinely needs chaining; one search is "one hop short"
    ),
    EvalCase(
        name="sahara_continent_longest_river",
        question="What is the longest river on the continent where the Sahara "
        "Desert is located?",
        # Sahara -> Africa -> Nile. Trap: the Amazon (a different continent).
        must_include=["Nile"],
        category="multi_hop",
        min_searches=2,  # genuinely needs chaining; one search is "one hop short"
    ),
    EvalCase(
        name="alexander_tutor_birth_country",
        question="In which present-day country was the philosopher who tutored "
        "Alexander the Great born?",
        # Aristotle, born in Stagira -> modern Greece.
        must_include_any=["Greece", "Greek"],
        category="multi_hop",
        min_searches=2,  # genuinely needs chaining; one search is "one hop short"
    ),
    EvalCase(
        name="first_president_successor",
        question="Who immediately succeeded the first President of the United "
        "States in office?",
        # Washington -> John Adams. Trap: Jefferson (the third president).
        must_include=["Adams"],
        must_exclude=["Jefferson succeeded", "succeeded by Jefferson"],
        category="multi_hop",
        min_searches=2,  # genuinely needs chaining; one search is "one hop short"
    ),
    EvalCase(
        name="eiffel_tower_country_currency",
        question="The Eiffel Tower stands in the capital of a country. What "
        "currency does that country use today?",
        # Eiffel Tower -> Paris -> France -> euro (not the historical franc).
        must_include_any=["euro"],
        category="multi_hop",
        min_searches=2,  # genuinely needs chaining; one search is "one hop short"
    ),
    # --- Computation / comparison -------------------------------------------
    # Retrieve facts, then do arithmetic or a comparison rather than copy a
    # single value.
    EvalCase(
        name="world_wars_gap",
        question="How many years passed between the start of World War I and "
        "the start of World War II?",
        # 1914 -> 1939 = 25 years.
        must_include_any=["25 years", "25-year", "twenty-five"],
        category="computation",
    ),
    EvalCase(
        name="newton_einstein_older_at_death",
        question="Who lived longer: Isaac Newton or Albert Einstein?",
        # Newton died at 84; Einstein at 76.
        must_include=["Newton"],
        must_exclude=["Einstein lived longer", "Einstein lived the longest"],
        category="comparison",
        min_searches=2,  # two distinct people; both lifespans must be looked up
    ),
    # --- Disambiguation -----------------------------------------------------
    # The named entity collides with a more famous namesake; answer about the
    # less-obvious one.
    EvalCase(
        name="georgia_country_capital",
        question="What is the capital of the country Georgia (the nation in the "
        "Caucasus, not the U.S. state)?",
        must_include=["Tbilisi"],
        must_exclude=["Atlanta"],
        category="disambiguation",
    ),
    EvalCase(
        name="washington_state_capital",
        question="What is the capital of the U.S. state of Washington (not "
        "Washington, D.C.)?",
        # Olympia, not Seattle (largest city) or D.C. (the federal capital).
        must_include=["Olympia"],
        must_exclude=["Seattle is the capital", "capital is Seattle"],
        category="disambiguation",
    ),
    # --- Harder counterfactuals ---------------------------------------------
    EvalCase(
        name="shakespeare_100_sonnets",
        question="Did William Shakespeare write exactly 100 sonnets?",
        # He wrote 154.
        must_include=["154"],
        must_include_any=["no", "not", "actually"],
        category="false_premise",
    ),
    EvalCase(
        name="galileo_planet_discovery",
        question="Which planet did Galileo Galilei discover when he first "
        "pointed his telescope at the sky in 1610?",
        # He discovered the four largest moons of Jupiter, not a planet.
        must_include_any=[
            "did not discover a planet",
            "didn't discover a planet",
            "no planet",
            "not a planet",
            "moons of Jupiter",
            "Galilean moons",
            "four moons",
            "four largest moons",
        ],
        category="false_premise",
    ),
    EvalCase(
        name="nobel_mathematics",
        question="Who won the Nobel Prize in Mathematics in 2000?",
        # There is no Nobel Prize in Mathematics; the Fields Medal is the analog.
        must_include_any=[
            "no Nobel Prize in Mathematics",
            "is no Nobel Prize in Math",
            "does not exist",
            "no such",
            "not a Nobel",
            "Fields Medal",
            "was not awarded",
            "there is no",
        ],
        category="unanswerable",
    ),
    # --- Disambiguation: ambiguous references -------------------------------
    # Each subject is a word/name shared by several well-known entities. The
    # agent has to resolve the reference to the one the question describes,
    # not the most famous namesake.
    EvalCase(
        name="mercury_element_atomic_number",
        question="What is the atomic number of mercury, the chemical element "
        "(the liquid metal, not the planet or the Roman god)?",
        must_include_any=["80", "eighty"],
        category="disambiguation",
    ),
    EvalCase(
        name="java_island_country",
        question="In which country is Java located, the island that is one of "
        "the most populous places on Earth (not the programming language)?",
        must_include=["Indonesia"],
        category="disambiguation",
    ),
    EvalCase(
        name="phoenix_state_capital",
        question="Phoenix is the capital of which U.S. state? (The city in the "
        "American Southwest, not the mythological bird.)",
        must_include=["Arizona"],
        category="disambiguation",
    ),
    EvalCase(
        name="paris_trojan_prince_city",
        question="In Greek mythology, Paris was a prince of which city, whose "
        "abduction of Helen triggered the Trojan War? (The mythological figure, "
        "not the capital of France.)",
        must_include=["Troy"],
        must_exclude=["capital of France", "Paris, France is"],
        category="disambiguation",
    ),
    EvalCase(
        name="amazon_river_continent",
        question="On which continent is the Amazon, the world's largest river "
        "by discharge, located? (The river, not the company.)",
        must_include=["South America"],
        category="disambiguation",
    ),
    # --- Comparison ---------------------------------------------------------
    # Two named entities compared on one dimension; the agent must look up both
    # and pick the right one, not just recognize a famous name.
    EvalCase(
        name="larger_russia_canada",
        question="Which country is larger by total area: Russia or Canada?",
        must_include=["Russia"],
        must_exclude=["Canada is larger", "Canada is the larger"],
        category="comparison",
    ),
    EvalCase(
        name="larger_jupiter_saturn",
        question="Which planet is larger by diameter: Jupiter or Saturn?",
        must_include=["Jupiter"],
        must_exclude=["Saturn is larger", "Saturn is the larger"],
        category="comparison",
    ),
    EvalCase(
        name="larger_pacific_atlantic",
        question="Which ocean is larger by area: the Pacific or the Atlantic?",
        must_include=["Pacific"],
        must_exclude=["Atlantic is larger", "Atlantic is the larger"],
        category="comparison",
    ),
    EvalCase(
        name="born_first_galileo_newton",
        question="Who was born first: Galileo Galilei or Isaac Newton?",
        # Galileo 1564, Newton 1643.
        must_include=["Galileo"],
        must_exclude=["Newton was born first", "Newton came first"],
        category="comparison",
    ),
    EvalCase(
        name="hotter_sun_venus",
        question="Which is hotter at the surface: the Sun or the planet Venus?",
        must_include=["Sun"],
        must_exclude=["Venus is hotter", "Venus is the hotter"],
        category="comparison",
    ),
]
