from collections import Counter

def getBestPair(tokens):
    return Counter(zip(tokens, tokens[1:])).most_common(1)[0][0]


def merging(best_pair, tokens, newToken):
    resultingTokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
            resultingTokens.append(newToken)
            i += 1
        else:
            resultingTokens.append(tokens[i])
        i += 1
    return resultingTokens


def decode(encodedTokens, encodingDico):
    decodedTokens = []
    for num in encodedTokens:
        recGetDec(decodedTokens, num, encodingDico)
    return decodedTokens


def recGetDec(decodedTokens, num, encodingDico):
    pair = encodingDico.get(num)
    if num >= 256:
        recGetDec(decodedTokens, pair[0], encodingDico)
        recGetDec(decodedTokens, pair[1], encodingDico)
    else:
        decodedTokens.append(num)
    return


def encode(tokens, encodingDico, vocabulary_size):
    intermediateEncoding = list(tokens)
    for num in range(256, vocabulary_size):
        intermediateEncoding = merging(encodingDico[num], intermediateEncoding, num)
    return intermediateEncoding


def train(text, vocabulary_size):
    tokens = list(text.encode("utf-8"))
    encodingDico = {}
    for num in range(256, vocabulary_size):
        bestPair = getBestPair(tokens)
        encodingDico[num] = bestPair
        tokens = merging(bestPair, tokens, num)
    return encodingDico


def testText(text, encodingDico, vocabulary_size):
    rawTokens = list(text.encode("utf-8"))
    encodedTokens = encode(rawTokens, encodingDico, vocabulary_size)
    decodedText = bytes(decode(encodedTokens, encodingDico)).decode("utf-8")
    print("Original :", text)
    print("Decode   :", decodedText)
    print("Identique :", decodedText == text)
    print("Taille :", len(rawTokens), "bytes ->", len(encodedTokens), "tokens",
          f"(ratio {len(rawTokens) / len(encodedTokens):.2f})")
    print()


if __name__ == "__main__":
    vocabulary_size = 300

    trainingText = (
        "Le tokenizer apprend les paires de bytes les plus fréquentes dans le texte. "
        "À chaque étape, il fusionne la paire la plus fréquente en un nouveau token, "
        "puis il recommence sur le texte compressé. Plus le texte est long, plus les "
        "fusions deviennent intéressantes : les mots courants comme « le », « la », "
        "« les », « des », « est » ou « que » finissent par devenir un seul token. "
        "Les caractères accentués comme é, è, à ou ç prennent deux bytes en UTF-8, "
        "donc le tokenizer apprend vite à les regrouper. Les emojis comme 😀 en prennent "
        "quatre. Le but est de réduire la longueur des séquences tout en gardant un "
        "vocabulaire de taille raisonnable pour le réseau de neurones qui viendra après."
    )
    encodingDico = train(trainingText, vocabulary_size)

    print("Merges appris :")
    for num, pair in encodingDico.items():
        merged = bytes(decode([num], encodingDico)).decode("utf-8", errors="replace")
        print(num, pair, repr(merged))
    print()

    print("=== Texte d'entrainement ===")
    testText(trainingText, encodingDico, vocabulary_size)

    print("=== Nouveau texte ===")
    otherText = (
        "Est-ce que le tokenizer marche aussi sur un texte qu'il n'a jamais vu ? "
        "Les mots fréquents devraient être compressés, et les caractères inconnus "
        "comme 🚀 ou ñ restent de simples bytes."
    )
    testText(otherText, encodingDico, vocabulary_size)

