import random
import numpy as np
import matplotlib.pyplot as plt
import config
from ag import encription, fitness, generateSmartPopulation, initialPopulation, pmx_crossover, cx_crossover, ox_crossover, newKey, populationBest, keyAccuracy, decode, getColoredKey, init_cipher_bigrams, mcmc_optimize

def run_ag(callback=None, stop_event=None, use_hybrid=True):
    encriptionKey = list(config.alfabet_without_space)
    random.shuffle(encriptionKey)
    keyEncriptionCopy = "".join(encriptionKey.copy())

    encriptionText = encription(config.inputText, encriptionKey)
    init_cipher_bigrams(encriptionText)
    trueFitness = fitness(encriptionText, encriptionKey)

    population = generateSmartPopulation(config.populationSize)

    fitnessHistory = []
    mutatePoints = []
    avgHistory = []
    stdHistory = []
    medHistory = []
    
    stagnationCounter = 0
    prevBest = float('-inf')
    localCooldownCounter = config.cooldownCounter

    for gen in range(config.generations):
        if stop_event and stop_event.is_set():
            break

        if localCooldownCounter > 0: 
            localCooldownCounter -= 1

        results = []
        for individual in population:
            result = fitness(encriptionText, individual)
            results.append(result)

        current_best_fitness = max(results)
        avgHistory.append(np.mean(results))
        medHistory.append(np.median(results))
        stdHistory.append(np.std(results))
        fitnessHistory.append(current_best_fitness)

        stagnationLimit = int(config.generations * 0.1) if gen < int(config.generations * 0.6) else int(config.generations * 0.05)

        if current_best_fitness > prevBest:
            prevBest = current_best_fitness
            stagnationCounter = 0
        else:
            stagnationCounter += 1

        worstCase = min(results)
        shiftedResults = [(res - worstCase) + 1 for res in results]
        totalFitnessShifted = sum(shiftedResults)
        probability = [res / totalFitnessShifted for res in shiftedResults]

        parents = random.choices(population, weights=probability, k=config.nSelection)

        children = []
        pairs = zip(parents[::2], parents[1::2])
        for first, second in pairs:
            child1, child2 = first.copy(), second.copy()
            progress = gen / config.generations

            if progress < 0.30:
                methods, probs = ["PMX", "OX", "CX"], [1.0, 0.0, 0.0]
            elif progress < 0.70:
                methods, probs = ["PMX", "OX", "CX"], [1.0, 0.0, 0.0]
            else:
                methods, probs = ["PMX", "OX", "CX"], [0.1, 0.7, 0.2]
            
            method = random.choices(methods, weights=probs, k=1)[0]
            if method == "PMX": child1, child2 = pmx_crossover(child1, child2)
            elif method == "CX": child1, child2 = cx_crossover(child1, child2)
            elif method == "OX": child1, child2 = ox_crossover(child1, child2)
            
            children.append(child1)
            children.append(child2)

        if use_hybrid:
            for i in range(len(children)):
                children[i] = mcmc_optimize(children[i], config.mcmc_steps_per_mutation)

        if localCooldownCounter == 0:
            if stagnationCounter > stagnationLimit:
                mutatePoints.append(gen)
                for _ in range(config.massMutate):
                    idx = random.randrange(len(children))
                    children[idx] = newKey(children[idx], config.newKey_amount * 2)
                stagnationCounter = 0
                localCooldownCounter = config.cooldownLimit

            elif np.median(results) >= current_best_fitness * 0.99:
                mutatePoints.append(gen)
                for i in range(len(children)):
                    if random.random() < 0.3:
                        children[i] = newKey(children[i], config.newKey_amount)
                localCooldownCounter = config.cooldownLimit // 2

        for i in range(len(children)):
            if random.random() < config.mutation_rate:
                children[i] = newKey(children[i], config.newKey_amount)

        ind_res = sorted(list(zip(population, results)), key=lambda x: x[1], reverse=True)
        for i in range(config.nElite):
            children.append(ind_res[i][0])

        population = children + random.choices(population, weights=probability, k=(config.populationSize - len(children)))

        bestIndividual, bestIndividualFitness = populationBest(population, config.populationSize, encriptionText)
        accuracy = keyAccuracy(bestIndividual, encriptionKey)
        if accuracy > config.historical_best:
            config.historical_best = accuracy

        if callback:
            decoded_text_sample = decode(encriptionText, bestIndividual)
            callback(keyEncriptionCopy, "".join(bestIndividual), bestIndividualFitness, accuracy, gen, decoded_text_sample)
        elif gen % 10 == 0 or gen == config.generations - 1:
            bestKeyString = getColoredKey(bestIndividual, encriptionKey)
            sampleText = decode(encriptionText[:80], bestIndividual)
            print(f"Gen {gen} | Fitness: {bestIndividualFitness:.4f} | Accuracy: {accuracy * 100:.1f}%")

    # --- POPRAWKA: Oddzielamy obsługę wykresu ---
    if callback:
        plot_data = {
            'generations_range': list(range(len(fitnessHistory))),
            'fitnessHistory': fitnessHistory,
            'avgHistory': avgHistory,
            'stdHistory': stdHistory,
            'medHistory': medHistory,
            'mutatePoints': mutatePoints,
            'trueFitness': trueFitness
        }
        callback(plot_data=plot_data)
    else:
        plotfitness(range(len(fitnessHistory)), fitnessHistory, avgHistory, stdHistory, medHistory, mutatePoints, trueFitness)

def plotfitness(generations_range, fitnessHistory, avgHistory, stdHistory, medHistory, mutatePoints, trueFitness):
    plt.figure(figsize=(10,6))
    plt.fill_between(generations_range, np.array(avgHistory) - np.array(stdHistory), np.array(avgHistory) + np.array(stdHistory), color='gray', alpha=0.2, label='Odchylenie std')
    plt.plot(generations_range, fitnessHistory, color='green', linewidth=2, label='best fitness')
    plt.plot(generations_range, medHistory, color='blue', linewidth=1, label='mediana z populacji')
    plt.title('Proces uczenia Algorytmu Genetycznego', fontsize=14)
    plt.xlabel('Pokolenie (Iteration)', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.axhline(y=trueFitness, color='r', linestyle='--', label='Original fitness')
    plt.vlines(mutatePoints, plt.ylim()[0], plt.ylim()[1], colors='orange', linestyles='--', alpha=0.4, label='Mutacja wymuszona')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_ag()