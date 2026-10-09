import json
import os
import re
from collections import Counter

# " ?\w+['’]?" : un mot avec son espace devant, et l'apostrophe collée à la fin (" qu'", " l'")
# " ?[^\w\s]+" : de la ponctuation avec son espace devant
# "\s*\n+"     : des retours à la ligne, avec les espaces qui les précèdent
# "\s+(?!\S)"  : des espaces, sauf le dernier s'il est suivi d'un mot (il reste collé au mot)
# "\s+"        : les espaces restants (en fin de texte)
WORD_PATTERN = re.compile(r" ?\w+['’]?| ?[^\w\s]+|\s*\n+|\s+(?!\S)|\s+")


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


class Tokenizer:
    '''Tokenizer BPE sur les bytes : 256 tokens de base (un par byte)
    plus un token par merge appris'''

    def __init__(self, vocabularySize=1500):
        self.vocabularySize = vocabularySize
        self.encodingDico = {}  # id -> paire, dans l'ordre de création des merges
        self.inverseEncodingDico = {}  # paire -> id
        self.encodeCache = {}  # mot -> tokens

    def train(self, path):
        '''Entraîne le tokenizer sur le fichier texte donné'''
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.trainText(text)

    def trainText(self, text):
        '''BPE sur les mots uniques du texte, pondérés par leur fréquence'''
        wordsCount = Counter(WORD_PATTERN.findall(text))
        wordsTokens = [list(word.encode("utf-8")) for word in wordsCount]
        wordsFreq = list(wordsCount.values())
        encodingDico = {}
        for num in range(256, self.vocabularySize):
            bestPair = getBestPairWords(wordsTokens, wordsFreq)
            if bestPair is None:
                break
            encodingDico[num] = bestPair
            wordsTokens = [merging(bestPair, tokens, num) for tokens in wordsTokens]
        self.setMerges(encodingDico)

    def setMerges(self, encodingDico):
        '''Remplace les merges et met à jour tout ce qui en dépend'''
        self.encodingDico = encodingDico
        self.inverseEncodingDico = {v: k for k, v in encodingDico.items()}
        self.encodeCache = {}  # l'ancien cache ne correspond plus aux nouveaux merges
        self.vocabularySize = 256 + len(encodingDico)

    def encodeWord(self, word):
        '''Encodage des tokens mot par mots'''
        wordTokens = list(word.encode("utf-8"))  # conversion du mot en ses bytes associés
        while len(wordTokens) >= 2:  # pas de BPE si pas de paire
            bestPair = min(
                zip(wordTokens, wordTokens[1:]),
                key=lambda pair: self.inverseEncodingDico.get(pair, float("inf")),
            )  # On prend la paire qui possède un encodage avec l'indice le plus petit
            if bestPair not in self.inverseEncodingDico:  # si tout est encodé dans le mot
                return wordTokens
            wordTokens = merging(bestPair, wordTokens, self.inverseEncodingDico[bestPair])
        return wordTokens

    def encode(self, text):
        '''Encode tout le texte (encodage par mot)'''
        encodedTokens = []
        for word in WORD_PATTERN.findall(text):  # division du texte en mots
            if word not in self.encodeCache:
                self.encodeCache[word] = self.encodeWord(word)
            encodedTokens += self.encodeCache[word]
        return encodedTokens

    def decode(self, encodedTokens):
        '''Fonction de décodage d'un texte tokenizé'''
        decodedBytes = []
        for num in encodedTokens:
            self.recGetDec(decodedBytes, num)
        # errors="replace" : une suite de tokens peut couper un caractère UTF-8 en deux
        return bytes(decodedBytes).decode("utf-8", errors="replace")

    def recGetDec(self, decodedBytes, num):
        '''Obtenir le decodage de manière récursive sur un token
        (on décode le token de gauche puis de droite)'''
        if num >= 256:
            pair = self.encodingDico[num]
            self.recGetDec(decodedBytes, pair[0])
            self.recGetDec(decodedBytes, pair[1])
        else:
            decodedBytes.append(num)

    def save(self, path):
        '''Sauvegarde les merges en JSON, sous forme de [id, gauche, droite]'''
        merges = [[num, pair[0], pair[1]] for num, pair in self.encodingDico.items()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": merges}, f)

    def load(self, path):
        '''Recharge des merges sauvegardés avec save'''
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # JSON n'a pas de tuples : on reconstruit les paires
        self.setMerges({num: (left, right) for num, left, right in data["merges"]})


def trainTokenizer(vocabularySize, dataFile):
    '''Entraîne un tokenizer sur data/dataFile et le sauvegarde dans tokenizer_<taille>.json'''
    tokenizer = Tokenizer(vocabularySize)
    tokenizer.train(os.path.join("data", dataFile))
    tokenizer.save(f"tokenizer_{vocabularySize}.json")
    return tokenizer


if __name__ == "__main__":
    trainTokenizer(1500, "miserables_train.txt")
