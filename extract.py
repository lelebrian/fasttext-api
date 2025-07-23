from gensim.models import KeyedVectors
import csv

# === CONFIGURATION ===
vec_file = "C:\\Users\\emanu\\Desktop\\it_vec\\cc.it.300.vec"
output_csv = "it/fasttext_100k_words.csv"   # output CSV file path
max_words = 100000

# === LOAD MODEL ===
print("🔄 Loading .vec file...")
original_model = KeyedVectors.load_word2vec_format(vec_file, limit=max_words)
print(f"✅ Loaded {len(original_model.index_to_key)} words.")

valid_words = [w for w in original_model.index_to_key if w.islower()]
filtered_vectors = [original_model[w] for w in valid_words]

# === BUILD NEW MODEL ===
model = KeyedVectors(vector_size=original_model.vector_size)
model.add_vectors(valid_words, filtered_vectors)
print(f"✅ Filtered: {len(valid_words)} lowercase words kept.")

# === SAVE WORDS TO CSV ===
print("💾 Saving words to CSV...")
with open(output_csv, mode='w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    for word in model.index_to_key:
        writer.writerow([word])

print(f"✅ Words saved to: {output_csv}")
