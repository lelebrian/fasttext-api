import spacy
from gensim.models import KeyedVectors
import csv

# === CONFIGURATION ===
vec_file = "C:\\Users\\emanu\\Desktop\\it_vec\\cc.it.300.vec"
output_csv = "it/fasttext_nouns_data.csv"
output_words_path = "it/fasttext_nouns_only_words.csv"
max_words = 50000

# === LOAD ORIGINAL MODEL ===
print("🔄 Loading .vec file...")
original_model = KeyedVectors.load_word2vec_format(vec_file, limit=max_words)
print(f"✅ Loaded {len(original_model.index_to_key)} total words.")

# === CREATE FILTERED MODEL with only lowercase words ===
print("📦 Filtering lowercase-only words...")
valid_words = [w for w in original_model.index_to_key if w.islower()]
filtered_vectors = [original_model[w] for w in valid_words]

model = KeyedVectors(vector_size=original_model.vector_size)
model.add_vectors(valid_words, filtered_vectors)
print(f"✅ Kept {len(valid_words)} lowercase words.")

# === LOAD spaCy ===
print("🔎 Loading spaCy Italian model...")
nlp = spacy.load("it_core_news_sm")

abstract_suffixes = [
    "tà", "zione", "zioni", "enza", "ezza",
    "ismo", "ità", "tudine"
]

def is_probably_abstract(word):
    return any(
        word.endswith(suffix)
        for suffix in abstract_suffixes
    )

# === PROCESS NOUNS ===
rows = []
seen_words = set()

for word in model.index_to_key:
    doc = nlp(word)
    if not doc:
        continue
    token = doc[0]
    if token.pos_ == "NOUN" and token.text == token.lemma_ and not is_probably_abstract(word):
        try:
            nearest_word, similarity = model.most_similar([word], topn=1)[0]
            precedes = "YES" if nearest_word in seen_words else "NO"
            rows.append([
                token.text,
                token.lemma_,
                token.pos_,
                token.tag_,
                token.morph,
                nearest_word,
                similarity,
                precedes
            ])
            seen_words.add(word)
        except KeyError:
            continue

# === SAVE TO CSV ===
print(f"💾 Saving {len(rows)} rows to CSV...")
with open(output_csv, mode='w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["text", "lemma", "pos", "tag", "morph", "nearest", "similarity", "precedes"])
    writer.writerows(rows)

print(f"✅ Done! File saved to: {output_csv}")


# === SAVE list of words with precedes == "NO" ===

print("📝 Salvo solo le parole con precedes == NO...")

with open(output_words_path, mode='w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["text"])
    for row in rows:
        if row[-1] == "NO":
            writer.writerow([row[0]])

print(f"✅ Lista parole salvata in: {output_words_path}")

