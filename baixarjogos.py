import requests
import json


username = "heigonsoldera"

archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"

archives_response = requests.get(archives_url)

if archives_response.status_code == 200:
    archives_data = archives_response.json()

    archive_urls = archives_data["archives"]
    
    all_games = []
    
    for url in archive_urls:        
        response = requests.get(url)
        
        if response.status_code == 200:            
            data = response.json()
            
            if "games" in data:                
                games = data["games"]
                
                all_games.extend(games)
    
    with open(f"{username}_all_games.json", "w") as f:
        json.dump(all_games, f)

    print(f"Todos os jogos de {username} foram salvos em {username}_all_games.json")
else:
    print("Falha ao obter os jogos. Verifique o nome de usuário e tente novamente.")
