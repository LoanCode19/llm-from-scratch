import re
import time
from collections import Counter

WORD_PATTERN = re.compile(r" ?\w+| ?[^\w\s]+|\s+")


def getBestPair(tokens):
    return Counter(zip(tokens, tokens[1:])).most_common(1)[0][0]


def getBestPairWords(wordsTokens, wordsFreq):
    '''Récupère la paire la plus commune parmis tous les mots 
    pondérée par la fréquence des mots la contenant'''
    pairs = Counter()
    for tokens, freq in zip(wordsTokens, wordsFreq):
        for pair in zip(tokens, tokens[1:]):
            pairs[pair] += freq
    if not pairs:
        return None
    return pairs.most_common(1)[0][0]


def merging(best_pair, tokens, newToken):
    '''Remplacement de l'ancienne paire de bytes la plus fréquente dans notre
    liste par le nouveau token : newToken'''
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
    '''Fonction de décodage d'un texte tokenizé'''
    decodedTokens = []
    for num in encodedTokens:
        recGetDec(decodedTokens, num, encodingDico)
    return decodedTokens


def recGetDec(decodedTokens, num, encodingDico):
    ''' Obtenir le decodage de manière récursive sur un token (on décode le token de gauche puis de droite)'''
    pair = encodingDico.get(num)
    if num >= 256:
        recGetDec(decodedTokens, pair[0], encodingDico)
        recGetDec(decodedTokens, pair[1], encodingDico)
    else:
        decodedTokens.append(num)
    return


def encodeOld(tokens, encodingDico, vocabulary_size):
    '''version naïve de l'encodage, token par token'''
    intermediateEncoding = list(tokens)
    for num in range(256, vocabulary_size): #encodage par ordre d'"importance"
        intermediateEncoding = merging(encodingDico[num], intermediateEncoding, num)
    return intermediateEncoding


def encodeWord(word, inverseEncodingDico):
    '''Encodage des tokens mot par mots'''
    wordTokens = list(word.encode("utf-8"))#conversion du mot en ses bytes associés 
    while len(wordTokens) >= 2: #pas de BPE si pas de paire
        bestPair = min(
            zip(wordTokens, wordTokens[1:]),
            key=lambda pair: inverseEncodingDico.get(pair, float("inf")),
        ) #On prend la paire qui possède un encodage avec l'indice le plus petit 
        if bestPair not in inverseEncodingDico:#si tout est encodé dans le mot
            return wordTokens
        else:
            wordTokens = merging(bestPair, wordTokens, inverseEncodingDico[bestPair])
    return wordTokens


def encode(text, encodingDico):
    '''Encode tout le texte selon le dico d'encodage donné (encodage par mot)'''
    words = WORD_PATTERN.findall(text) #division du texte en mots
    encodedTokens = []
    encodeCache = {}
    inverseEncodingDico = {v: k for k, v in encodingDico.items()}
    for word in words:
        if encodeCache.get(word, 0) != 0:
            encodedTokens += encodeCache[word]
        else:
            wordTokens = encodeWord(word, inverseEncodingDico)
            encodeCache[word] = wordTokens
            encodedTokens += wordTokens
    return encodedTokens


def splitWords(text):
    return Counter(WORD_PATTERN.findall(text))


def train(text, vocabulary_size):
    wordsCount = splitWords(text)
    wordsTokens = [list(word.encode("utf-8")) for word in wordsCount]
    wordsFreq = list(wordsCount.values())
    encodingDico = {}
    for num in range(256, vocabulary_size):
        bestPair = getBestPairWords(wordsTokens, wordsFreq)
        if bestPair is None:
            break
        encodingDico[num] = bestPair
        wordsTokens = [merging(bestPair, tokens, num) for tokens in wordsTokens]
    return encodingDico


def trainOld(text, vocabulary_size):
    tokens = list(text.encode("utf-8"))#On récupère les tokens 
    encodingDico = {}
    for num in range(256, vocabulary_size): #BPE
        bestPair = getBestPair(tokens)
        encodingDico[num] = bestPair
        tokens = merging(bestPair, tokens, num)
    return encodingDico


def testText(text, encodingDico):
    rawTokens = list(text.encode("utf-8"))
    encodedTokens = encode(text, encodingDico)
    decodedText = bytes(decode(encodedTokens, encodingDico)).decode("utf-8")
    print("Original :", text[:200])
    print("Decode   :", decodedText[:200])
    print("Identique :", decodedText == text)
    print(
        "Taille :",
        len(rawTokens),
        "bytes ->",
        len(encodedTokens),
        "tokens",
        f"(ratio {len(rawTokens) / len(encodedTokens):.2f})",
    )
    print()


def speedTest(text, vocabulary_size):
    print(
        f"=== Test de vitesse ({len(text.encode('utf-8'))} bytes, vocabulaire {vocabulary_size}) ==="
    )
    for trainFunction in (trainOld, train):
        start = time.perf_counter()
        trainFunction(text, vocabulary_size)
        print(f"{trainFunction.__name__} : {time.perf_counter() - start:.2f} s")
    print()


def encodeSpeedTest(text, encodingDico, vocabulary_size):
    print(f"=== Test de vitesse de l'encodage ({len(text.encode('utf-8'))} bytes) ===")
    start = time.perf_counter()
    newTokens = encode(text, encodingDico)
    print(f"encode : {time.perf_counter() - start:.2f} s, {len(newTokens)} tokens")

if __name__ == "__main__":
    vocabulary_size = 1000

    with open("data/miserables_train.txt", encoding="utf-8") as f:
        trainingText = f.read()
    with open("data/miserables_test.txt", encoding="utf-8") as f:
        testingText = f.read()

    speedTest(trainingText[:500_000], 400)

    start = time.perf_counter()
    encodingDico = train(trainingText, vocabulary_size)
    print(
        f"Entraînement complet : {time.perf_counter() - start:.2f} s, {len(encodingDico)} merges"
    )
    print()

    print("20 derniers merges appris :")
    for num, pair in list(encodingDico.items())[-20:]:
        merged = bytes(decode([num], encodingDico)).decode("utf-8", errors="replace")
        print(num, pair, repr(merged))
    print()

    print("=== Extrait du texte d'entraînement (tomes I à IV) ===")
    testText(trainingText[100_000:110_000], encodingDico)

    print("=== Extrait jamais vu (tome V) ===")
    testText(testingText[100_000:110_000], encodingDico)

    encodeSpeedTest(testingText[:50_000], encodingDico, vocabulary_size)

    print("=== Caractères inconnus ===")
