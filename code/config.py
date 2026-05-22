import pickle
import numpy as np

with open('char_map.pkl', 'rb') as f:
    charToInd = pickle.load(f)

Mlog = np.load('Mlog_matrix.npy')

mutation_rate = 0.25
newKey_amount = 1
populationSize = 200
generations = 2000
nSelection = 100
nElite = 5
alfabet_without_space = "aąbcćdeęfghijklłmnńoóprsśtuwyzźż"
historical_best = 0
stagnationCounter = 0
prevBest = -99999
massMutate = int(populationSize * 0.15)
maxAttempts = 5000
maxDiversity = 0.1875
cooldownCounter = 0
cooldownLimit = 20
mcmc_steps_per_mutation = 50

mcmc_iterations = 20000000

full_inputText = """mroźny powiew wiatru smagał twarze wędrowców brnących przez głęboki śnieg w oddali majaczyły groźne szczyty gór przeklętych spowite gęstą mgłą drużyna liczyła na krótki odpoczynek przy ognisku lecz nagle ciszę przerwał potworny ryk niosący się echem po wąwozie stary czarodziej uniósł kostur który rozbłysnął błękitnym światłem ukazując sylwetki nadciągających bestii wojownik dobył ciężkiego miecza a łucznik naciągnął cięciwę mierząc w mrok każdy wiedział że ta noc będzie sprawdzianem ich męstwa i hartu ducha walka o przeżycie właśnie się rozpoczęła pod czarnym niebem Swoje prawdziwe oblicze się to przecież umiłowanym pisarzem i w jego włazów, oznaczone rubinowymi w drugiej bryk. Była to zakazane rzeczy: poczynając obracać się na jego zegarka starych cywilnych pilot widział, poprzez szklane na aktualności, zastałoby rurze, żeby jej pograłby na automatach! Jeśliby a po wtóre nie nosił okularów. Poszedł tylko oskorupiał mu dłonie; zezwalał na korzystanie pół korony. A gdyby poszedł włazów, oznaczone rubinowymi kurz oskorupiał mu uciążliwej wichury? Dwukoronówka. Srebrna, dźwięcząca, powiedzieć, że wyraźnie czuł jej je odczytywać z aparatu mikromózgi wielkości pestki czekał na dalszy zrzucali ją na spadochronie! Oczywiście Zapewne w miejscu, gdzie rude od prochu, jego przywykłe do mroku oczy jakim sunął, nieważki pod Olej ścina się metalowy mózg do góry nogami. Kiedy coraz słabsze, zachowały nodze, zadał pytanie poprzez szklane wskazują okrwawionym palcem mordercę? Się niebieskimi płomyczkami oznaczone rubinowymi lampkami, swojemu ręce! Wiśni, które można pozostawiając po sobie i wszędzie wszystko, co okaże Ośla Łączka wymienił odpowiedź. Wyglądał, refleksji nad zmianą komisji z dowodem w ręku. Jej krągłość i widział, pograłby na automatach. Jeśliby automat którego się zżymał, bo nie upatrywał materialne podłoże deptało coraz słabsze, zachowały ale, po pierwsze, musiałby nie było jasne, w jaki drogę pocisku względem tła Ujścia rezerwowych włazów, wielkiego żalu do Smigi W niebieskawym półmroku fortuny, kiedy wyrwał życia w zwłokach powracających. Witam bardzo serdecznie na szczęśliwe przygody zapraszam na seans wsystkich Polaków! Kocham sztuczną inteligencję! drużyna liczyła na krótki odpoczynek przy ognisku lecz nagle ciszę przerwał Niełatwą, bo nie na tym kończy się, jak nogą Zręcznie wierzgnąć, z uśmiechem witać lada kogo; Bo taka grzeczność modna, zda mi się kupiecka, Ale nie staropolska, ani też szlachecka. Grzeczność wszystkim należy, lecz każdemu inna; Bo nie jest bez grzeczności i miłość ǳiecinna, I wzgląd męża dla żony przy luǳiach, i pana Dla sług swoich, a w każdej jest pewna odmiana. Trzeba się długo uczyć, ażeby nie zbłąǳić Historia, Obyczaje"""

# Domyślnie 20 słów
inputText = " ".join(full_inputText.split()[:20])