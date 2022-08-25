import numpy as np

giocatori = [
    'Davide',
    'Joe',
    'Lorenzo',
    'Teo',
    'Nicola',
    'Marco'
]

regola = ['Draft', 'Random']
for i in range(0, 5):
    np.random.shuffle(giocatori)
    np.random.shuffle(regola)
    print("Scelta: " + regola[0])
    print("Team 1")
    print(giocatori[0])
    print(giocatori[1])
    print(giocatori[2])
    print("\nTeam 2")
    print(giocatori[3])
    print(giocatori[4])
    print(giocatori[5])
    print("\n\n")
