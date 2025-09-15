from gensim.models import KeyedVectors

# === CONFIGURATION ===
vec_file = "C:\\Users\\emanu\\Desktop\\es_vec\\cc.es.300.vec"
output_file = "es/fasttext_100k.kv"
max_words = 70000

# === LOAD MODEL ===
print("🔄 Loading .vec file...")
original_model = KeyedVectors.load_word2vec_format(vec_file, limit=max_words)
print(f"📦 Loaded {len(original_model.index_to_key)} words.")

# === FILTER: only fully lowercase words ===
valid_words = [w for w in original_model.index_to_key if w.islower()]
filtered_vectors = [original_model[w] for w in valid_words]

# === BUILD NEW MODEL ===
model = KeyedVectors(vector_size=original_model.vector_size)
model.add_vectors(valid_words, filtered_vectors)
print(f"✅ Filtered: {len(valid_words)} lowercase words kept.")

# === SAVE ===
print("💾 Saving to .kv format...")
model.save(output_file)
print(f"✅ Saved KeyedVectors model to: {output_file}")
