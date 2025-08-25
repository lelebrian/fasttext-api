from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from gensim.models import KeyedVectors
import psutil
import os
import numpy as np
import Levenshtein
import json
import logging

app = Flask(__name__)
CORS(app)

# configure once at app startup (if not already)
logging.basicConfig(level=logging.INFO)

print("🚀 AVVIO CORRETTO DEL CODICE MODIFICATO")

# Carica il modello all'avvio
print("Caricamento modello FastText...")


from gensim.models import KeyedVectors

def print_ram_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size
    mem_mb = mem_bytes / 1024**2
    print(f"🔍 Current RAM usage: {mem_mb:.2f} MB")

# Example: after loading the model
print_ram_usage()


def load_with_progress(filename, max_words):
    with open(filename, 'r', encoding='utf-8') as f:
        total_words, dim = map(int, f.readline().split())
        print(f"Totale parole: {total_words}, dimensione: {dim}")

        model = KeyedVectors(vector_size=dim)
        words = []
        vectors = []

        for i, line in enumerate(f):
            if i >= max_words:
                break
            parts = line.rstrip().split(" ")
            word = parts[0]
            # vec = list(map(float, parts[1:]))
            vec = np.array(list(map(float, parts[1:])), dtype=np.float32)
            words.append(word)
            vectors.append(vec)

            if i % 100000 == 0:
                print(f"Caricate {i}/{max_words} parole...")

        model.add_vectors(words, vectors)

        del words
        del vectors

        return model

# Loading kv
model = KeyedVectors.load("it/fasttext_100k.kv", mmap='r')  # local

# Loading vec
num_words = 100000
#model = load_with_progress("it\cc.it.300.vec", num_words) # NUMERO PAROLE

print_ram_usage()

print("Modello caricato!")

@app.route("/similarity")
def similarity():
    w1 = request.args.get("word1")
    w2 = request.args.get("word2")

    if not w1 or not w2:
        return jsonify({"error": "Parametri 'word1' e 'word2' obbligatori"}), 400

    if w1 not in model or w2 not in model:
        return jsonify({"error": "Una delle parole non è nel vocabolario"}), 404

    score = model.similarity(w1, w2)
    return jsonify({
        "word1": w1,
        "word2": w2,
        "similarity": round(float(score), 4)
    })


@app.route("/hint")
def hint():
    #Argomenti
    w1 = request.args.get("word1")
    w2 = request.args.get("word2")
    tentative = int(request.args.get("tentative", 0))
    lev_threshold = 2
    lev_threshold_similarity = 0.7

    # Parametri
    strategy = request.args.get("strategy", "corrected_rank_sum")
    limit = request.args.get("topn", default=1000, type=int)
    record_best = request.args.get("best", default=1000, type=int)
    aiutino = 0.03  # per strategia "min_score"
    weight_rank1 = 1.0 + 0.3 * tentative  # per strategia "corrected_rank_sum"

    # Argomento: blacklist
    blacklist_param = request.args.get("blacklist", "")
    blacklist = set(word.strip().lower() for word in blacklist_param.split(",") if word.strip())

    # Check parole non nulle e in vocabolario
    if not w1 or not w2:
        return jsonify({"error": "Parametri 'word1' e 'word2' obbligatori"}), 400

    if w1 not in model or w2 not in model:
        return jsonify({"error": "Una delle parole non è nel vocabolario"}), 404

    max_iterations = 20   # you can tweak this
    iteration = 1
    candidate_words = set()
    top_w2 = model.most_similar(w2, topn=limit)
    rank_w2 = {word: (i + 1, float(score)) for i, (word, score) in enumerate(top_w2)}

    while iteration <= max_iterations and not candidate_words:
        try:
            if strategy == "converge":
                top_w1 = model.most_similar(w1, topn=record_best * iteration)
            else:
                top_w1 = model.most_similar(w1, topn=limit * iteration)
        except KeyError:
            return jsonify({"error": "Errore nel calcolo delle similarità"}), 500

        # Calcola il rank e la similarità per ogni parola
        rank_w1 = {word: (i + 1, float(score)) for i, (word, score) in enumerate(top_w1)}

        # Candidati: presenti in entrambe le liste e non blacklistati
        candidate_words = set(rank_w1.keys()).intersection(rank_w2.keys())
        candidate_words = {w for w in candidate_words if w.lower() not in blacklist}

        # Filtra per Levenshtein
        candidate_words = {
            w for w in candidate_words
            if Levenshtein.distance(w.lower(), w1.lower()) > lev_threshold
            and Levenshtein.distance(w.lower(), w2.lower()) > lev_threshold
            and all(Levenshtein.distance(w.lower(), b) > lev_threshold for b in blacklist)
        }

        if candidate_words:
            break  # ✅ trovato almeno un candidato, uscita
        else:
            logging.info("Iteration %s: no candidates found, expanding search...", iteration)
            iteration += 1

    if not candidate_words:
        return jsonify({"error": "Nessuna parola trovata dopo %s iterazioni" % max_iterations}), 404
        best = None

    for word in candidate_words:

        score1 = rank_w1[word][1]
        score2 = rank_w2[word][1]
        rank1 = rank_w1[word][0]
        rank2 = rank_w2[word][0]

        if strategy == "converge":
            score = rank2
            is_better = best is None or score < best["score"]
        elif strategy == "rank_sum":
            score = rank1 + rank2
            is_better = best is None or score < best["score"]
        elif strategy == "corrected_rank_sum":
            score = rank1 * weight_rank1 + rank2
            is_better = best is None or score < best["score"]
        elif strategy == "corrected_min_score":
            adjusted_score2 = score2 * (1 + aiutino * tentative)
            score = min(score1, adjusted_score2)
            is_better = best is None or score > best["score"]
        else:
            return jsonify({"error": "Strategia non valida"}), 400

        if is_better:
            best = {
                "word": word,
                "score": round(score, 4),
                "strategy": strategy,
                "score_with_word1": round(score1, 4),
                "score_with_word2": round(score2, 4),
                "rank_in_word1": rank1,
                "rank_in_word2": rank2
            }

    print_ram_usage()

    if best:
        return jsonify({
            "word": best["word"],
            "rank_sum": int(best["rank_in_word1"] + best["rank_in_word2"]),
            "rank_1": int(best["rank_in_word1"]),
            "rank_2": int(best["rank_in_word2"])
        })

    else:
        return jsonify({"error": "Nessuna parola trovata"}), 404


@app.route("/hint_test")
def hint_test():
    #Argomenti
    w1 = request.args.get("word1")
    w2 = request.args.get("word2")
    tentative = int(request.args.get("tentative", 0))
    lev_threshold = 2
    lev_threshold_similarity = 0.7

    # Parametri
    strategy = request.args.get("strategy", "corrected_rank_sum")
    limit = request.args.get("topn", default=1000, type=int)
    record_best = request.args.get("best", default=1000, type=int)
    aiutino = 0.03  # per strategia "min_score"
    weight_rank1 = 1.0 + 0.3 * tentative  # per strategia "corrected_rank_sum"

    # Argomento: blacklist
    blacklist_param = request.args.get("blacklist", "")
    blacklist = set(word.strip().lower() for word in blacklist_param.split(",") if word.strip())

    # Check parole non nulle e in vocabolario
    if not w1 or not w2:
        return jsonify({"error": "Parametri 'word1' e 'word2' obbligatori"}), 400

    if w1 not in model or w2 not in model:
        return jsonify({"error": "Una delle parole non è nel vocabolario"}), 404

    # Recupera le x-mila parole più simili a soluzione e a hint
    try:
        if strategy == "converge":
            top_w1 = model.most_similar(w1, topn=record_best)
        else:
            top_w1 = model.most_similar(w1, topn=limit * 2)
        top_w2 = model.most_similar(w2, topn=limit * 2)
    except KeyError:
        return jsonify({"error": "Errore nel calcolo delle similarità"}), 500

    # Calcola il rank e la similarità per ogni parola
    rank_w1 = {word: (i + 1, float(score)) for i, (word, score) in enumerate(top_w1)}
    rank_w2 = {word: (i + 1, float(score)) for i, (word, score) in enumerate(top_w2)}

    logging.info("Number of words in rank_w1: %s", len(rank_w1))   # 👈 this will log to console/file
    logging.info("Number of words in rank_w2: %s", len(rank_w2))   # 👈 this will log to console/file

    # Prende come parole candidate solo quelle che sono sia in top_w1 e in top_w2
    candidate_words = set(rank_w1.keys()).intersection(rank_w2.keys())

    logging.info("Candidate words after first intersection: %s", len(candidate_words))

    candidate_words = {w for w in candidate_words if w.lower() not in blacklist}

    logging.info("Candidate words after blacklist: %s", len(candidate_words))

    # Prende solo le parole con Lev distance > soglia
    candidate_words = {
        w for w in candidate_words
        if w.lower() not in blacklist and
        Levenshtein.distance(w.lower(), w1.lower()) > lev_threshold and
        Levenshtein.distance(w.lower(), w2.lower()) > lev_threshold and
        all(Levenshtein.distance(w.lower(), b) > lev_threshold for b in blacklist)
    }

    logging.info("Candidate words after lev distance: %s", len(candidate_words))

    best = None

    track = []

    logging.info("Going through the candidate_words: %s", len(candidate_words))   # 👈 this will log to console/file

    for word in candidate_words:

        score1 = rank_w1[word][1]
        score2 = rank_w2[word][1]
        rank1 = rank_w1[word][0]
        rank2 = rank_w2[word][0]

        if strategy == "converge":
            score = rank2
            is_better = best is None or score < best["score"]
        elif strategy == "rank_sum":
            score = rank1 + rank2
            is_better = best is None or score < best["score"]
        elif strategy == "corrected_rank_sum":
            score = rank1 * weight_rank1 + rank2
            is_better = best is None or score < best["score"]
        elif strategy == "corrected_min_score":
            adjusted_score2 = score2 * (1 + aiutino * tentative)
            score = min(score1, adjusted_score2)
            is_better = best is None or score > best["score"]
        else:
            return jsonify({"error": "Strategia non valida"}), 400
        
        track_word = {
                "word": word,
                "score": round(score, 4),
                "strategy": strategy,
                "score_with_word1": round(score1, 4),
                "score_with_word2": round(score2, 4),
                "rank_in_word1": rank1,
                "rank_in_word2": rank2
            }
        
        track.append(track_word)

        ### Logs to console
        logging.info("Track word: %s", json.dumps(track_word, ensure_ascii=False))

        if is_better:
            best = {
                "word": word,
                "score": round(score, 4),
                "strategy": strategy,
                "score_with_word1": round(score1, 4),
                "score_with_word2": round(score2, 4),
                "rank_in_word1": rank1,
                "rank_in_word2": rank2
            }

    print_ram_usage()

    if best:
        return jsonify({
            "word": best["word"],
            "rank_sum": int(best["rank_in_word1"] + best["rank_in_word2"]),
            "rank_1": int(best["rank_in_word1"]),
            "rank_2": int(best["rank_in_word2"]),
            "track": track   # <--- include here
        })

    else:
        return jsonify({"error": "Nessuna parola trovata"}), 404


@app.route("/test")
def testnew():
    #Argomenti
    w1 = request.args.get("word1")
    w2 = request.args.get("word2")
    tentative = int(request.args.get("tentative", 0))
    lev_threshold = 2
    lev_threshold_similarity = 0.7

    # Parametri
    strategy = request.args.get("strategy", "corrected_rank_sum")
    record_best = request.args.get("best", default=1000, type=int)
    limit = request.args.get("topn", default=1000, type=int)
    limit2 = 10000
    aiutino = 0.03  # per strategia "min_score"
    weight_rank1 = 1.0 + 0.3 * tentative  # per strategia "corrected_rank_sum"

    # Argomento: blacklist
    blacklist_param = request.args.get("blacklist", "")
    blacklist = set(word.strip().lower() for word in blacklist_param.split(",") if word.strip())

    # Argomento: check
    check = request.args.get("check", None)

    # Check parole non nulle e in vocabolario
    if not w1 or not w2:
        return jsonify({"error": "Parametri 'word1' e 'word2' obbligatori"}), 400
    if w1 not in model or w2 not in model:
        return jsonify({"error": "Una delle parole non è nel vocabolario"}), 404

    # Recupera le x-mila parole più simili a soluzione e a hint
    try:
        top_w1 = model.most_similar(w1, topn=record_best)
        top_w2 = model.most_similar(w2, topn=limit2)

        top_w1 = [
            w for w in top_w1
            if Levenshtein.distance(w[0].lower(), w1.lower()) > lev_threshold
        ]
    except KeyError:
        return jsonify({"error": "Errore nel calcolo delle similarità"}), 500

    # Calcola il rank e la similarità per ogni parola
    rank_w1 = {word: (i + 1, float(score)) for i, (word, score) in enumerate(top_w1)}
    rank_w2 = {word: (i + 1, float(score)) for i, (word, score) in enumerate(top_w2)}
    
    # Prende come parole candidate solo quelle che sono sia in top_w1 e in top_w2
    candidate_words = rank_w1.keys()
    candidate_words = {w for w in candidate_words if w.lower() not in blacklist}

    # Prende solo le parole con Lev distance > soglia
    candidate_words = {
        w for w in candidate_words
        if w.lower() not in blacklist and
        Levenshtein.distance(w.lower(), w1.lower()) > lev_threshold and
        Levenshtein.distance(w.lower(), w2.lower()) > lev_threshold and
        all(Levenshtein.distance(w.lower(), b) > lev_threshold for b in blacklist)
    }

    def compute_word_rank_and_score(model, word, target):
        try:
            similarity = model.similarity(word, target)
        except KeyError:
            return None
        all_similarities = []
        for other in model.index_to_key:
            if other == word:
                continue
            try:
                score = model.similarity(word, other)
                all_similarities.append((other, score))
            except KeyError:
                continue
        all_similarities.sort(key=lambda x: x[1], reverse=True)
        for rank, (w, _) in enumerate(all_similarities, start=1):
            if w == target:
                return {"rank": rank, "score": round(similarity, 4)}
        return None

    def best_words_by_max_min_score(n=5):
        results = []
        for word in candidate_words:
            r1 = rank_w1[word][0]
            r2 = rank_w2[word][0] if word in rank_w2 else limit2
            score1 = round(rank_w1[word][1], 4)
            score2 = round(rank_w2[word][1], 4) if word in rank_w2 else 0
            score = min(score1 * (1 + aiutino * tentative), score2)
            results.append((score, word, r1, r2, score1, score2))

        results.sort(reverse=True)

        return [
            {
                "word": word,
                "min_score": round(score, 4),
                "score_with_word1": score1,
                "score_with_word2": score2,
                "rank_in_word1": r1,
                "rank_in_word2": r2
            }
            for score, word, r1, r2, score1, score2 in results[:n]
        ]

    def best_words_by_corrected_rank_sum(n=5):
        weight_rank1 = 1.0 + 0.3 * tentative
        results = []
        for word in candidate_words:
            r1 = rank_w1[word][0]
            r2 = rank_w2[word][0] if word in rank_w2 else limit2
            score1 = round(rank_w1[word][1], 4)
            score2 = round(rank_w2[word][1], 4) if word in rank_w2 else 0
            score = r1 * weight_rank1 + r2
            results.append((score, word, r1, r2, score1, score2))
        results.sort()
        return [
            {
                "word": word,
                "rank_in_word1": r1,
                "rank_in_word2": r2,
                "score_with_word1": score1,
                "score_with_word2": score2,
                "weight_on_rank1": round(weight_rank1, 2),
                "weighted_rank_sum": round(score, 1)
            }
            for score, word, r1, r2, score1, score2 in results[:n]
        ]
    
    annotated_top_w1_formatted = []

    top_words_by_rank = {item["word"] for item in best_words_by_corrected_rank_sum()}
    top_words_by_score = {item["word"] for item in best_words_by_max_min_score()}

    for i, (word, score) in enumerate(top_w1, start=1):
        rank1 = rank_w1[word][0]
        rank2 = rank_w2[word][0] if word in rank_w2 else limit2
        score1 = round(rank_w1[word][1], 3)
        score2 = round(rank_w2[word][1], 3) if word in rank_w2 else 0
        score_by_rank = int(round(rank1 * weight_rank1 + rank2, 0))
        score_min_by_semantic = round(min(score1 * (1 + aiutino * tentative), score2), 3)
        #score_sum_by_semantic = round((score1 * (1 + aiutino * tentative)) + score2, 3)

        formatted = " " + f'{word}'.ljust(15) + ": " + str({
            "r1": f'{rank1}'.rjust(5),
            "r2": f'{rank2}'.rjust(5),
            "rank": f'{score_by_rank}'.rjust(5),
            "s1": f'{score1}'.rjust(5),
            "s2": f'{score2}'.rjust(5),
            #"scoresum": f'{score_sum_b"y_semantic}'.rjust(5),
            "scoremin": f'{score_min_by_semantic}'.rjust(5),
        }).replace("'", "")

        if word in top_words_by_rank:
            formatted += " *R*"
        else:
            formatted += "    "

        if word in top_words_by_score:
            formatted += " *S*"
        else:
            formatted += "    "

        formatted += "    "

        annotated_top_w1_formatted.append(formatted)

    return Response(json.dumps({
        "best_5_by_corrected_rank_sum": best_words_by_corrected_rank_sum(),
        "best_5_combined": best_words_by_max_min_score(),
        "top_w1": annotated_top_w1_formatted
    }, ensure_ascii=False), mimetype='application/json')


@app.route("/lev")
def levenshtein_similar_words():
    word = request.args.get("word")
    if not word:
        return jsonify({"error": "Parametro 'word' mancante"}), 400

    try:
        top_words = model.most_similar(word, topn=1000)
    except KeyError:
        return jsonify({"error": f"La parola '{word}' non è nel vocabolario"}), 404

    result = []
    for rank, (vocab_word, sim_score) in enumerate(top_words[:1000], start=1):
        lev_distance = Levenshtein.distance(word.lower(), vocab_word.lower())
        lev_ratio = Levenshtein.ratio(word.lower(), vocab_word.lower())
        formatted = (
            f'"{vocab_word} ({rank})", '
            f'"levenshtein_distance": {lev_distance}, '
            f'"levenshtein_ratio": {round(lev_ratio, 4)}, '
            f'"semantic_similarity": {round(sim_score, 4)}'
        )
        result.append(formatted)

    return jsonify({
        "input": word,
        "top_1000_by_semantic_similarity": result
    })



@app.route("/check")
def check():
    w1 = request.args.get("word1")
    w2 = request.args.get("word2")
    check = request.args.get("check")

    if not w1 or not w2 or not check:
        return jsonify({"error": "Parametri 'word1', 'word2' e 'check' obbligatori"}), 400

    if w1 not in model or w2 not in model or check not in model:
        return jsonify({"error": "Una delle parole non è nel vocabolario"}), 404

    def get_rank_and_score(base_word, target_word):
        try:
            target_score = float(model.similarity(base_word, target_word))  # converti a float puro
        except KeyError:
            return None

        similarities = []
        for word in model.index_to_key:
            if word == base_word:
                continue
            try:
                score = float(model.similarity(base_word, word))  # converti anche qui
                similarities.append((word, score))
            except KeyError:
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        for rank, (word, _) in enumerate(similarities, start=1):
            if word == target_word:
                return {"word": target_word, "rank": rank, "score": round(target_score, 4)}
        return None

    info_w1 = get_rank_and_score(w1, check)
    info_w2 = get_rank_and_score(w2, check)

    return jsonify({
        "with_word1": info_w1,
        "with_word2": info_w2
    })


@app.route("/ram")
def ram():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    mem_mb = mem_bytes / 1024**2
    return jsonify({"ram_usage_mb": round(mem_mb, 2)})



if __name__ == "__main__":
    app.run(debug=True)
