from gensim.models import KeyedVectors
import spacy

# === CONFIGURATION ===
vec_file = "C:\\Users\\emanu\\Desktop\\it_vec"
output_file = "it/fasttext_100k_dedup.kv"
max_words = 80000

print("🔄 Loading Italian language model...")
nlp = spacy.load("it_core_news_sm")

def get_lemma(word):
    doc = nlp(word)
    return doc[0].lemma_

print("🔄 Loading .vec file...")
raw_model = KeyedVectors.load_word2vec_format(vec_file, limit=max_words)
print(f"✅ Loaded {len(raw_model.index_to_key)} words.")

# === DEDUPLICATION BY LEMMA ===
print("🧹 Deduplicating by lemma...")
seen_lemmas = {}
filtered_words = []

for word in raw_model.index_to_key:
    lemma = get_lemma(word.lower())
    if lemma not in seen_lemmas:
        seen_lemmas[lemma] = word
        filtered_words.append(word)

print(f"✅ Reduced to {len(filtered_words)} unique lemmas.")

# === FILTERED MODEL CREATION ===
model = KeyedVectors(vector_size=raw_model.vector_size)
model.add_vectors(
    filtered_words,
    [raw_model[word] for word in filtered_words]
)
d
print("💾 Saving to .kv format...")
model.save(output_file)
print(f"✅ Saved deduplicated model to: {output_file}")
