import time
from collections import Counter

from tokenizer import Tokenizer, merging


def getBestPair(tokens):
    return Counter(zip(tokens, tokens[1:])).most_common(1)[0][0]


def trainOld(text, vocabulary_size):
    '''Ancienne version de l'entraînement, sur le texte entier sans découpe en mots'''
    tokens = list(text.encode("utf-8"))  # On récupère les tokens
    encodingDico = {}
    for num in range(256, vocabulary_size):  # BPE
        bestPair = getBestPair(tokens)
        encodingDico[num] = bestPair
        tokens = merging(bestPair, tokens, num)
    return encodingDico


def speedTest(text, vocabularySize):
    '''Compare le temps d'entraînement de trainOld et du Tokenizer'''
    print(f"=== Test de vitesse ({len(text.encode('utf-8'))} bytes, vocabulaire {vocabularySize}) ===")
    start = time.perf_counter()
    trainOld(text, vocabularySize)
    print(f"trainOld : {time.perf_counter() - start:.2f} s")
    start = time.perf_counter()
    Tokenizer(vocabularySize).trainText(text)
    print(f"Tokenizer : {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    with open("data/miserables_train.txt", encoding="utf-8") as f:
        trainingText = f.read()
    speedTest(trainingText[:500_000], 400)
