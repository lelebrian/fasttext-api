from gensim.models import KeyedVectors

# === CONFIGURATION ===
vec_file = "it/cc.it.300.vec"        # path to your .vec file
output_file = "it/fasttext_100k.kv"     # desired output .kv file
max_words = 80000                   # number of words to keep

# === CONVERSION ===
print("🔄 Loading .vec file...")
model = KeyedVectors.load_word2vec_format(vec_file, limit=max_words)

print(f"✅ Loaded {len(model.index_to_key)} words.")
print("💾 Saving to .kv format...")

model.save(output_file)

print(f"✅ Saved KeyedVectors model to: {output_file}")
