# Kryptoanaliza Szyfru Podstawieniowego: Algorytm Memetyczny

## Opis projektu
Narzędzie służące do łamania szyfrów podstawieniowych w języku polskim. Wykorzystuje zaawansowany Algorytm Genetyczny (AG) z mechanizmami zapobiegającymi wczesnej stagnacji populacji (zwiększona mutacja) oraz implementację metody Markov Chain Monte Carlo (MCMC) z kryterium Metropolisa. Połączenie obu metod pozwoliło na stworzenie Algorytmu Memetycznego, gdzie GA odpowiada za globalną eksplorację, a MCMC za lokalne szlifowanie rozwiązań. Dodatkowo, projekt stanowi platformę badawczą umożliwiającą szczegółową analizę porównawczą skuteczności i dynamiki działania trzech podejść: czystego Algorytmu Genetycznego (GA), metody MCMC oraz ich hybrydowego połączenia.

## Główne funkcjonalności

### Analiza Bigramów
Prawdopodobieństwa występowania par liter zostały obliczone na podstawie Pana Tadeusza i zapisane w postaci logarytmicznej w macierzy (plik mlog_matrix.npy). Stanowią one podstawę do oceny fitness generowanych kluczy. Plik macierz_bigramow_prob.xlsx pozwala na ręczną weryfikację wyników.

### Algorytm Genetyczny
Testuje trzy różne typy krzyżowania, aby zachować zróżnicowanie puli genetycznej: PMX, CX oraz OX. Zawiera wykres wartości fitness, ułatwiający bieżącą obserwację działania mechanizmów mutacji i poszukiwań.

### Wizualizacja Postępu
Wyświetla kolorowany wygenerowany klucz zestawiony z kluczem używanym do szyfrowania tekstu, co znacząco pomaga ocenić zgodność podczas testów.

### Algorytm MCMC
Przeprowadza losowy spacer badając przestrzeń kluczy (zmiany metodą transpozycji), przyjmując i odrzucając nowe stany wedle reguł akceptacji.

## Wymagania
Aby używać programu, upewnij się, że masz zainstalowane biblioteki numpy, pandas oraz matplotlib.



# Cryptanalysis of Substitution Cipher: Memetic Algorithm

## Project Description
A tool designed to break substitution ciphers in the Polish language. It utilizes an advanced Genetic Algorithm (GA) featuring mechanisms to prevent early population stagnation (mass mutation) and an implementation of the Markov Chain Monte Carlo (MCMC) method with the Metropolis criterion. The synergy of these methods resulted in a Memetic Algorithm, where GA handles global exploration and MCMC provides local optimization. Furthermore, the project serves as a research platform for a detailed comparative analysis of the effectiveness and performance dynamics of three distinct approaches: pure Genetic Algorithm (GA), the MCMC method, and their hybrid combination.

## Key Features

### Bigram Analysis
The probabilities of letter pairs were calculated based on the text of Pan Tadeusz and saved in a logarithmic matrix (file mlog_matrix.npy). These probabilities serve as the foundation for evaluating the fitness of generated keys. The macierz_bigramow_prob.xlsx file allows for manual verification of the results.

### Genetic Algorithm
Tests three different types of crossover to maintain genetic pool diversity: PMX, CX, and OX. Includes a fitness value chart, facilitating real-time observation of mutation and search mechanisms.

### Progress Visualization
Displays a color-coded generated key compared with the key used to encrypt the text, which significantly helps in assessing accuracy during testing.

### MCMC Algorithm
Conducts a random walk exploring the key space (changes using the transposition method), accepting and rejecting new states according to acceptance rules.

## Requirements
To use the program, ensure you have the numpy, pandas, and matplotlib libraries installed.
