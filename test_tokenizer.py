import os
import sys
import tempfile
import time

from tokenizer import Tokenizer

results = []


def check(name, condition, details=""):
    '''Affiche le résultat d'un test et le garde pour le bilan final'''
    results.append(condition)
    print(f"[{'OK' if condition else 'ÉCHEC'}] {name} {details}")


def roundTrip(tokenizer, name, text):
    '''Encode puis décode le texte, vérifie qu'on retrouve l'original'''
    start = time.perf_counter()
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    duration = time.perf_counter() - start
    ratio = len(text.encode("utf-8")) / len(tokens) if tokens else 0
    check(name, decoded == text, f"({len(tokens)} tokens, ratio {ratio:.2f}, {duration:.2f} s)")
    return tokens


if __name__ == "__main__":
    with open("data/miserables_train.txt", encoding="utf-8") as f:
        trainingText = f.read()
    with open("data/miserables_test.txt", encoding="utf-8") as f:
        testingText = f.read()

    # petit entraînement pour que les tests restent rapides
    vocabularySize = 500
    tokenizer = Tokenizer(vocabularySize)
    start = time.perf_counter()
    tokenizer.trainText(trainingText[:300_000])
    print(f"Entraînement de test : vocabulaire {vocabularySize} sur 300 Ko en {time.perf_counter() - start:.2f} s")
    print()

    check("taille du vocabulaire", tokenizer.vocabularySize == vocabularySize, f"({tokenizer.vocabularySize})")

    roundTrip(tokenizer, "extrait d'entraînement", trainingText[100_000:110_000])
    tokens = roundTrip(tokenizer, "tome V complet (jamais vu)", testingText)
    check("ids dans le vocabulaire", all(0 <= t < tokenizer.vocabularySize for t in tokens))
    roundTrip(tokenizer, "caractères inconnus", "Est-ce que ça marche avec 🚀 et ñ ?")
    roundTrip(tokenizer, "texte vide", "")
    roundTrip(tokenizer, "un seul caractère", "é")
    roundTrip(tokenizer, "espaces et retours à la ligne", "  \n\n\t  mot  \n")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tokenizer.json")
        tokenizer.save(path)
        loaded = Tokenizer()
        loaded.load(path)
    sample = testingText[:50_000]
    check(
        "sauvegarde puis chargement",
        loaded.vocabularySize == tokenizer.vocabularySize and loaded.encode(sample) == tokenizer.encode(sample),
    )

    # texte trop petit pour atteindre le vocabulaire demandé : train doit s'arrêter tôt
    small = Tokenizer(10_000)
    small.trainText("ab ab ab")
    check(
        "arrêt quand il n'y a plus de paire",
        small.vocabularySize < 10_000 and small.decode(small.encode("ab ab ab")) == "ab ab ab",
        f"(vocabulaire {small.vocabularySize})",
    )

    print()
    failures = results.count(False)
    if failures:
        print(f"{failures} test(s) en échec sur {len(results)}")
        sys.exit(1)
    print(f"Les {len(results)} tests passent")
