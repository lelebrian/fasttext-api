from pathlib import Path
from itertools import combinations
from gensim.models import KeyedVectors

# === CONFIG ===
MODEL_PATH = Path("it/fasttext_100k.kv")  # path to your .kv model
WORDS_FILE = Path("C:\\Documenti_Lele\\Games\\+AdobeScripts\\resources\\word_list.txt")
OUTPUT_FILE = Path("C:\\Documenti_Lele\\Games\\+AdobeScripts\\resources\\word_pairs_by_similarity.txt")

# === Load Model ===
print("🔄 Loading model...")
model = KeyedVectors.load(str(MODEL_PATH))
print(f"✅ Model loaded with {len(model.index_to_key)} words.")

# === Load words ===
print("📥 Reading words list...")
with WORDS_FILE.open("r", encoding="utf-8") as f:
    words = [line.strip() for line in f if line.strip() in model]

print(f"✅ Found {len(words)} words in model.")

# === Compute pairwise similarities ===
print("🔍 Calculating similarities...")
similarities = []
for w1, w2 in combinations(words, 2):
    sim = model.similarity(w1, w2)
    similarities.append((sim, w1, w2))

# === Sort and save ===
print("💾 Writing ranked pairs...")
similarities.sort(reverse=True)
with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for sim, w1, w2 in similarities:
        f.write(f"{w1}, {w2}: {sim:.4f}\n")

print(f"✅ Done! Saved to: {OUTPUT_FILE}")
