RAG_MODES = {
    "mcq": {
        "MAX_CONTEXT_CHUNKS": 4,
        "MAX_TOKENS_PER_CHUNK": 300,
        "MAX_OUTPUT_TOKENS": 800,
        "TOP_K": 4
    },
    "one_line": {
        "MAX_CONTEXT_CHUNKS": 3,
        "MAX_TOKENS_PER_CHUNK": 200,
        "MAX_OUTPUT_TOKENS": 300,
        "TOP_K": 3
    },
    "short": {
        "MAX_CONTEXT_CHUNKS": 5,
        "MAX_TOKENS_PER_CHUNK": 500,
        "MAX_OUTPUT_TOKENS": 3045,
        "TOP_K": 5
    },
    "tashreeh": {
        "MAX_CONTEXT_CHUNKS": 5,
        "MAX_TOKENS_PER_CHUNK": 400,
        "MAX_OUTPUT_TOKENS": 2000,
        "TOP_K": 5
    },
    "khulasa": {
        "MAX_CONTEXT_CHUNKS": 6,
        "MAX_TOKENS_PER_CHUNK": 450,
        "MAX_OUTPUT_TOKENS": 2500,
        "TOP_K": 6
    },
    "markazi": {
        "MAX_CONTEXT_CHUNKS": 5,
        "MAX_TOKENS_PER_CHUNK": 400,
        "MAX_OUTPUT_TOKENS": 600,
        "TOP_K": 5
    },
    "application": {
        "MAX_CONTEXT_CHUNKS": 4,
        "MAX_TOKENS_PER_CHUNK": 350,
        "MAX_OUTPUT_TOKENS": 1200,
        "TOP_K": 4
    },
    "letter": { 
        "MAX_CONTEXT_CHUNKS": 4,
        "MAX_TOKENS_PER_CHUNK": 350,
        "MAX_OUTPUT_TOKENS": 1200,
        "TOP_K": 4
    },
    "story":  { 
        "MAX_CONTEXT_CHUNKS": 4,
        "MAX_TOKENS_PER_CHUNK": 350,
        "MAX_OUTPUT_TOKENS": 1500,
        "TOP_K": 4 
        },
    "dialogue": { 
        "MAX_CONTEXT_CHUNKS": 4,
        "MAX_TOKENS_PER_CHUNK": 350,
        "MAX_OUTPUT_TOKENS": 1500,
        "TOP_K": 4
    }
}

CHUNK_OVERLAP = 100

GENRE_TO_MODE: dict[str, str] = {
    # one-line / objective
    "mcq":                 "mcq",
    "word_meanings":       "mcq",
    "sentence_correction": "one_line",
    "zarbul_imsal":        "one_line",

    # short responses
    "short_question":      "short",
    "general_qa":          "short",
    "comprehension":       "short",
    "translation":         "short",

    # explanatory
    "tashreeh_ghazal":     "tashreeh",
    "tashreeh_nazam":      "tashreeh",
    "nasar_tashreeh":      "tashreeh",
    "poem_explanation":    "tashreeh",

    # summaries / longer structured responses
    "khulasa":             "khulasa",

    # conceptual / moral
    "markazi":       "markazi",

    # formatted writing tasks
    "application":         "application",
    "letter":              "letter",
    "story":               "story",
    "dialogue":            "dialogue",
}

# In config.py — add this

MODEL_ROUTING = {
    # Light/objective — llama-3.1-8b-instant (fast, low tokens)
    "mcq":                 "llama-3.1-8b-instant",
    "word_meanings":       "llama-3.1-8b-instant",
    "sentence_correction": "llama-3.1-8b-instant",
    "zarbul_imsal":        "llama-3.1-8b-instant",
    "translation":         "llama-3.1-8b-instant",

    # Medium — llama-3.3-70b-versatile (good Urdu, balanced)
    "short_question":      "llama-3.3-70b-versatile",
    "general_qa":          "llama-3.3-70b-versatile",
    "comprehension":       "llama-3.3-70b-versatile",
    "markazi_khyal":       "llama-3.3-70b-versatile",

    # Heavy/creative — llama-3.3-70b-versatile (best Urdu quality)
    "tashreeh_ghazal":     "llama-3.3-70b-versatile",
    "tashreeh_nazam":      "llama-3.3-70b-versatile",
    "nasar_tashreeh":      "llama-3.3-70b-versatile",
    "poem_explanation":    "llama-3.3-70b-versatile",
    "khulasa":             "llama-3.3-70b-versatile",
    "application":         "llama-3.3-70b-versatile",
    "letter":              "llama-3.3-70b-versatile",
    "story":               "llama-3.3-70b-versatile",
    "dialogue":            "llama-3.3-70b-versatile",

    # Paper — llama-3.1-8b-instant (single call, minimal prompt)
    "paper":               "llama-3.1-8b-instant",
}