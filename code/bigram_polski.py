import collections
import numpy as np
import re
import pandas as pd
import pickle

with open('Pan_Tadeusz.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.lower()
    text = re.sub(r'[^aąbcćdeęfghijklłmnńoóprsśtuwyzźż]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

alfabet = " aąbcćdeęfghijklłmnńoóprsśtuwyzźż"
charToInd = {char: i for i, char in enumerate(alfabet)}

with open('char_map.pkl', 'wb') as f:
    pickle.dump(charToInd, f)

n_alf=len(alfabet)
M = np.zeros((n_alf,n_alf))

for i in range(len(text) - 1):
    current_char = text[i]
    next_char = text[i+1]

    ind1 = charToInd[current_char]
    ind2 = charToInd[next_char]

    M[ind1][ind2] += 1

M += 1                      #+1 do kazdego pola macierzy
sum_row = M.sum(axis=1)     #suma kazdego wiersza w liscie (tablicy jednowymiarowej)

M_prob = M / sum_row[:, np.newaxis]      # prawdopodobienstwo wystapienia po danym znaku
Mlog = np.log(M_prob)                    # skala logarytmiczna (wszystkie wspolczynniki ujemne)

np.save('Mlog_matrix.npy', Mlog)


# etykiety = [c if c != ' ' else '[SPACJA]' for c in alfabet]

# df_prob = pd.DataFrame(M_prob, index = etykiety, columns = etykiety)
# df_log = pd.DataFrame(Mlog, index = etykiety, columns = etykiety)

# df_prob.to_excel("macierz_bigramow_prob.xlsx")
# df_log.to_excel("macierz_bigramow_log.xlsx")
# print("Macierz została wyeksportowana do pliku macierz_bigramow_prob.xlsx")
# print("Macierz została wyeksportowana do pliku macierz_bigramow_log.xlsx")
