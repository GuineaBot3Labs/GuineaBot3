import berserk
import requests
import traceback
import time
import random

TOKEN = 'YOUR-ACCESS-KEY'

tokensession = berserk.TokenSession(TOKEN)
client = berserk.Client(session=tokensession)
session = requests.Session()
session.headers.update({"Authorization": f"Bearer lip_D3yKCVwoB6tr9JWTrr0W"})
session_counter = 0  # Initialize a session counter

class Game:
        
    def get_game(self, username, challenge_id):
        global session_counter
        print(f"Accepted challenge from {username} (Session: {session_counter})")


session = requests.Session()
session.headers.update({"Authorization": f"Bearer {TOKEN}"})

# Initialize the timestamp for the last challenge

def get_game():
    try:
        response = session.get("https://lichess.org/api/account/playing")
        if response.status_code != 200:
            print(f"Error getting games: {response.text}")
            time.sleep(10)
            return None

        games = response.json()['nowPlaying']
        for game in games:
        
            return games
        time.sleep(5)
        return None

    except Exception:
        traceback.print_exc()
        
def handle_challenge(challenge):
    username = challenge['challenger']['id']
    challenge_id = challenge['id']
    time_control = challenge['timeControl']
    variant = challenge['variant']['key']
    games = get_game()
    
    if games != None:
        print(f"Declined challenge from {username} due to being too busy at the moment.")
        try:
            client.challenges.decline(challenge_id, reason=berserk.Reason.LATER)
        except Exception as e:
            pass
        return

    if variant != 'standard':
        print(f"Declined challenge from {username} due to variant {variant}")
        try:
            client.challenges.decline(challenge_id, reason=berserk.Reason.VARIANT)
        except Exception as e:
            pass
        return

    if time_control['type'] == 'clock' and time_control['limit'] < 18 * 60:
        print(f"Declined challenge from {username} due to time control below 18 minutes")
        try:
            client.challenges.decline(challenge_id, reason=berserk.Reason.TIMECONTROL)
        except Exception as e:
            pass
        return

    if username != "vladK17": # built in firewall that will block all requests from this specific user due to suspicious activity (bot spamming)
        try:
            client.challenges.accept(challenge_id)
        except Exception as e:
            pass
        global session_counter
        game = Game()
        game.get_game(username, challenge_id)
        session_counter += 1

    else:
        try:
            client.challenges.decline(challenge_id, reason=berserk.Reason.GENERIC)
        except Exception as e:
            pass
        print(f"Declined challenge from {username}")

# Stream incoming challenges and handle them
for event in client.board.stream_incoming_events():
    if event['type'] == 'challenge':
        handle_challenge(event['challenge'])
        last_challenge_time = time.time()


