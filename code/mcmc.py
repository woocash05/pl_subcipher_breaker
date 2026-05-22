import numpy as np
import pandas as pd
import random
import math
from bigram_polski import charToInd, Mlog, alfabet
import config
from ag import newKey, decode, fitness, encription, keyAccuracy, init_cipher_bigrams

CONST = 500
def run_mcmc(callback=None, stop_event=None):
    originalKey = list(config.alfabet_without_space)
    random.shuffle(originalKey)

    encriptionKey = list(config.alfabet_without_space)
    random.shuffle(encriptionKey)
    keyEncriptionCopy = "".join(encriptionKey.copy())

    if not callback:
        print(f"klucz szyfrujący: ", keyEncriptionCopy)

    encriptionText = encription(config.inputText, encriptionKey)
    if not callback:
        print("Tekst zaszyfrowany: ", encriptionText)

    init_cipher_bigrams(encriptionText)
    iterations = config.mcmc_iterations

    currentKey = originalKey.copy()
    currentFitness = fitness(encriptionText, currentKey)

    bestKey = currentKey.copy()
    bestFitness = currentFitness

    if not callback:
        print("Rozpoczynam łamanie szyfru...")

    for i in range(iterations):
        if stop_event and stop_event.is_set():
            break
        
        proposedKey = newKey(currentKey, config.newKey_amount)
        proposedFitness = fitness(encriptionText, proposedKey)
        delta = proposedFitness - currentFitness
        if delta > 0:
            currentFitness, currentKey = proposedFitness, proposedKey
            if currentFitness > bestFitness:
                bestFitness = currentFitness
                bestKey = currentKey.copy()
        else:
            if delta > -CONST:                  
                prob = math.exp(delta)          
                if random.random() < prob:      
                    currentFitness, currentKey = proposedFitness, proposedKey

        if callback:
            if i % 100 == 0 or i == iterations - 1:
                accuracy = keyAccuracy(currentKey, encriptionKey)
                decoded_text_sample = decode(encriptionText, currentKey)
                callback(keyEncriptionCopy, "".join(currentKey), currentFitness, accuracy, i, decoded_text_sample)
        else:
            if i % 200 == 0:
                obecny_tekst = decode(encriptionText, currentKey)
                print(f"Iteracja {i:05d} | Fit: {currentFitness:.2f} | {obecny_tekst[:60]}...")

    if not callback:
        print("\n=== GOTOWE ===")
        print("Najlepszy wynik:")
        print(decode(encriptionText, bestKey))
        print(f"Klucz: ", "".join(bestKey))
        print(f"Wskaźnik poprawności klucza: ", keyAccuracy(bestKey, encriptionKey) * 100, "%")

if __name__ == "__main__":
    run_mcmc()
