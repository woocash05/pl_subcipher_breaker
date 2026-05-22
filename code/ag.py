import numpy as np
import random
from config import mutation_rate, populationSize, generations, nSelection, nElite, alfabet_without_space, historical_best, stagnationCounter, prevBest, massMutate, maxAttempts, maxDiversity, inputText, Mlog, charToInd, mcmc_steps_per_mutation
import math

def newKey(oldVersionKey, amount=1):
    new = oldVersionKey.copy()

    # podmiana N parindeksow
    for _ in range(amount):
        i,j = random.sample(range(len(new)), 2)
        new[i], new[j] = new[j], new[i]

    return new

def decode(inputText, key):
    # slownik dekodujacy
    decodingMap = {key[i]: alfabet_without_space[i] for i in range (len(alfabet_without_space))}

    # dla danego znaku podmien go na odpowiednik ze slownika
    encodedText=[]
    for char in inputText:
        if char == " ":
            encodedText.append(" ")
        else:
            encodedText.append(decodingMap.get(char, char))

    return "".join(encodedText)

cipher_bigrams = None

def init_cipher_bigrams(encriptionText):
    global cipher_bigrams
    cipher_bigrams = np.zeros((33, 33))
    text = encriptionText + " "
    for i in range(len(text) - 1):
        c1 = text[i]
        c2 = text[i+1]
        ind1 = charToInd.get(c1)
        ind2 = charToInd.get(c2)
        if ind1 is not None and ind2 is not None:
            cipher_bigrams[ind1, ind2] += 1

def fast_fitness(key):
    P = np.zeros(33, dtype=int)
    P[0] = 0
    for i in range(len(alfabet_without_space)):
        enc_char = key[i]
        dec_char = alfabet_without_space[i]
        P[charToInd[enc_char]] = charToInd[dec_char]
    
    return np.sum(cipher_bigrams * Mlog[P[:, np.newaxis], P])

def fitness(inputText, key):
    # inputText argument is kept for compatibility but not used since we use global cipher_bigrams
    return fast_fitness(key)

def mcmc_optimize(key, iterations):
    current_key = key.copy()
    current_fitness = fast_fitness(current_key)
    
    best_key = current_key.copy()
    best_fitness = current_fitness
    
    for _ in range(iterations):
        proposed_key = current_key.copy()
        # swap 2
        i, j = random.sample(range(len(proposed_key)), 2)
        proposed_key[i], proposed_key[j] = proposed_key[j], proposed_key[i]
        
        proposed_fitness = fast_fitness(proposed_key)
        delta = proposed_fitness - current_fitness
        
        if delta > 0:
            current_key = proposed_key
            current_fitness = proposed_fitness
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_key = current_key.copy()
        else:
            if delta > -500:
                prob = math.exp(delta)
                if random.random() < prob:
                    current_key = proposed_key
                    current_fitness = proposed_fitness
                    
    return best_key

def encription(inputText, encriptionKey):
    shuffleMap = {alfabet_without_space[i]: encriptionKey[i] for i in range(len(alfabet_without_space))}
    encodedText = []
    for char in inputText:
        if char == " ":
            encodedText.append(" ")
        else:
            encodedText.append(shuffleMap.get(char,char))

    return "".join(encodedText)

def keyAccuracy(bestKey, encriptionKey):
    counter = 0
    for i in range(len(alfabet_without_space)):
        if bestKey[i] == encriptionKey[i]:
            counter += 1

    ratio = counter / len(alfabet_without_space)
    
    return ratio

# population

def initialPopulation(populationSize):
    population = []
    for _ in range(populationSize):
        solution = list(alfabet_without_space)
        random.shuffle(solution)
        population.append(solution)

    return population

def populationBest(population, populationSize, encriptionText):
   bestIndividual = population[0]
   bestIndividualFitness = float('-inf')
   for individual in population:
       individualFitness = fitness(encriptionText, individual)
       if individualFitness > bestIndividualFitness:
           bestIndividualFitness, bestIndividual = individualFitness, individual
   
   return bestIndividual, bestIndividualFitness

#crossing

def swapMechanismRec(child, dictionary, c1, c2):
    diff = set(alfabet_without_space) - set(child)
    if not diff:
        return child

    for i in range(len(alfabet_without_space)):
        if (i < c1 or i >= c2) and child[i] in child[c1:c2]:
            child[i] = dictionary[child[i]]
            return swapMechanismRec(child, dictionary, c1, c2)

def checkDuplicates(child, dictionary, c1, c2):
    checkingSet = set(child)
    diff = set(alfabet_without_space) - checkingSet

    if not diff:
        return child

    return swapMechanismRec(child, dictionary, c1, c2)
    
def generateCutPoints():
     while True:
          c1 = random.randrange(len(alfabet_without_space))      # pierwszy punkt cięcia
          c2 = random.randrange(len(alfabet_without_space))      # drugi punkt cięcia

          if c1 != c2 and c1 < c2:
              break
     return c1,c2

# partially-mapped
def pmx_crossover(child1, child2):
    c1,c2 = generateCutPoints()
    
    crossoverSlicedMap = {child1[i]: child2[i] for i in range(c1,c2)}                        # slownik dla child2    (wycięty)
    crossoverSlicedMapReverse = {v: k for k,v in crossoverSlicedMap.items()}                 # slownik dla child1    (dodatkowo odwrócony)

    child1[c1:c2], child2[c1:c2] = child2[c1:c2], child1[c1:c2]
    child1 = checkDuplicates(child1, crossoverSlicedMapReverse, c1, c2)
    child2 = checkDuplicates(child2, crossoverSlicedMap, c1, c2)

    return child1, child2

# cyclic 
def cx_crossover(child1, child2):
    ch1_cp = child1.copy()
    ch2_cp = child2.copy()

    n = random.randrange(len(alfabet_without_space))
    
    firstLetter = child1[n]
    
    child1[n], child2[n] = child2[n], child1[n]                                           # pierwsza podmiana liter
    
    currentLetter = ch2_cp[n]
    
   
    while(currentLetter != firstLetter):
        position = ch1_cp.index(currentLetter)
        currentLetter = ch2_cp[position]
        child1[position], child2[position] = child2[position], child1[position] 

    return child1, child2

# ordered
def ox_crossover(child1, child2):
    c1,c2 = generateCutPoints()

    ch1_cp = child1.copy()
    ch2_cp = child2.copy()

    child1[c1:c2], child2[c1:c2] = child2[c1:c2], child1[c1:c2]                             # podmiana środków

    tmp1 = ch1_cp[c2:] + ch1_cp[:c2]
    res1 = [gen for gen in tmp1 if gen not in child1[c1:c2]]
    
    numAtEnd = len(child1) - c2

    child1[c2:] = res1[:numAtEnd]
    child1[:c1] = res1[numAtEnd : numAtEnd + c1]

    tmp2 = ch2_cp[c2:] + ch2_cp[:c2]
    res2 = [gen for gen in tmp2 if gen not in child2[c1:c2]]
   
    child2[c2:] = res2[:numAtEnd]
    child2[:c1] = res2[numAtEnd : numAtEnd + c1]

    return child1, child2

def getColoredKey(bestKey, targetKey):
    GREEN = '\033[92m'
    RED= '\033[91m'
    RESET = '\033[0m'
    
    ColoredOutput =""

    for bChar, tChar in zip(bestKey, targetKey):
        if bChar == tChar:
            ColoredOutput += f"{GREEN}{bChar}{RESET}"
        else:
            ColoredOutput += f"{RED}{bChar}{RESET}"

    return ColoredOutput

def generateSmartPopulation(populationSize):
    population=[]
    
    first = list(alfabet_without_space)
    random.shuffle(first)
    population.append(first)

    attempts = 0
    while len(population) < populationSize and attempts <= maxAttempts:
        attempts += 1

        candidate = list(alfabet_without_space)
        random.shuffle(candidate)

        isDiverse = True

        for existing in population:
            if positionalSimilarity(candidate, existing) >= maxDiversity:
                isDiverse = False
                break

            if orderPreservation(candidate, existing) >= maxDiversity:
                isDiverse = False
                break

        if isDiverse:
            population.append(candidate)
            attempts = 0

    if len(population) < populationSize:
        print(f"Sito zapchane. Dobijam {populationSize - len(population)} kluczy losowo.")
        while len(population) < populationSize:
            fallback = list(alfabet_without_space)
            random.shuffle(fallback)
            population.append(fallback)

    return population

def positionalSimilarity(key1, key2):
    matches = sum(1 for a, b in zip(key1, key2) if a == b)

    return matches/len(alfabet_without_space)

def orderPreservation(child, parent):
    parent_pairs = set("".join(parent[i:i+2]) for i in range(len(parent)-1))
    child_pairs = set("".join(child[i:i+2]) for i in range(len(child)-1))
    preserved = parent_pairs.intersection(child_pairs)

    return len(preserved) / len(parent_pairs)