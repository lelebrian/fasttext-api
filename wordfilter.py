#!/usr/bin/env python3
"""
Filter a word list by semantic similarity to “oggetto”, “animale”, etc.

1. Read words from words_list.csv
2. Compute similarity of every word with pivot set
3. Only keep words with best similarity ≥ 0.20 AND similarity < 0.5 with any blacklist word
4. Save two CSVs: kept and discarded
"""

import csv
from pathlib import Path
from gensim.models import KeyedVectors

# ---------- CONFIG ----------------------------------------------------------
MODEL_PATH         = "it/fasttext_100k.kv"
WORD_FILE          = "it/words_list.csv"
THRESHOLD          = 0.20
BLACKLIST_MAX_SIM  = 0.50
OUT_PASS_FILE      = "words_above_0.20.csv"
OUT_FAIL_FILE      = "words_below_0.20.csv"
TARGET_WORDS       = ("oggetto", "animale", "banana", "negozio")
BLACKLIST_WORDS    = ("fratello",)
# ----------------------------------------------------------------------------

def main() -> None:
    print("Loading model …")
    model: KeyedVectors = KeyedVectors.load(MODEL_PATH, mmap="r")
    print("Model loaded with", len(model), "words")

    words = [w.strip() for w in Path(WORD_FILE).read_text(encoding="utf-8").splitlines() if w.strip()]
    print(f"Read {len(words)} candidate words")

    passed, failed = [], []

    for word in words:
        # Compute similarity with target pivots
        sim_targets = [
            model.similarity(word, tgt) if word in model and tgt in model else 0.0
            for tgt in TARGET_WORDS
        ]
        max_target_sim = max(sim_targets)

        # ✅ Compute similarity with blacklist terms
        blacklist_sims = [
            model.similarity(word, blk) if word in model and blk in model else 0.0
            for blk in BLACKLIST_WORDS
        ]
        max_blacklist_sim = max(blacklist_sims, default=0.0)

        #row = (word, *sim_targets, max_target_sim)
        row = [word]

        # ✅ Apply both thresholds
        if max_target_sim >= THRESHOLD and max_blacklist_sim < BLACKLIST_MAX_SIM:
            passed.append(row)
        else:
            failed.append(row)

    header = ["word"] + [f"sim_{t}" for t in TARGET_WORDS] + ["max_sim"]
    for fname, rows in ((OUT_PASS_FILE, passed), (OUT_FAIL_FILE, failed)):
        with open(fname, "w", newline="", encoding="utf-8") as f:
            #csv.writer(f).writerows([header, *rows])
            csv.writer(f).writerows(rows)
        print(f"{len(rows):>6} words → {fname}")

if __name__ == "__main__":
    main()
