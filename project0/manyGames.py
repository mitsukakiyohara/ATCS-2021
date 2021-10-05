games = ["secret hitler", "monopoly", "life", "chess"]
print("i like", games[0])
new_game = ''

while new_game != 'quit': 
    new_game = input("what game do you like? ")
    if new_game != 'quit': 
        games.append(new_game)

print(games)