import requests
import time
import random
import traceback
import json
import berserk


time_difference = time.time()
TOKEN = 'YOUR-API-KEY'
NAME = 'YOUR-USERNAME' # Note: Use your actual Username here
tokensession = berserk.TokenSession(TOKEN)
client = berserk.Client(session=tokensession)
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

# Check for 2-minute interval outside the loop
while True:
    if time_difference - time.time() < 90:
        response = session.get('https://lichess.org/api/bot/online')
        online_bots = [json.loads(line) for line in response.text.splitlines() if NAME not in line]

        try:
            games = get_game()  # Get the ongoing games
            if games == None:
                bot_to_play = random.choice(online_bots)
                time_control_minutes = random.randint(6, 40)  # Randomly choose a time control between 6 to 40 minutes
                gamevar = ['standard', 'horde', 'chess960']
                gamevar = random.choice(list(gamevar))
                print(f"minutes: {time_control_minutes}")
                time_control = time_control_minutes * 60  # Convert to seconds


                try:
                    print(f"Challenging {bot_to_play['id']} with time {time_control_minutes}, variant {gamevar} for increment of minutes")
                    # client.challenges.create(bot_to_play['id'], variant=gamevar, clock_limit=time_control, clock_increment=0, rated=True)
                    client.challenges.create('WorstFish', variant='standard', clock_limit=12*60, clock_increment=0, rated=False)

                except Exception:
                    time.sleep(90)

                last_challenge_time = time.time()  # Reset the timestamp for the last challenge

            else:
                print("there is a game going on currently, sleeping...")
                time.sleep(90)
        except Exception:
            traceback.print_exc()
            continue
    else:
        print(f"Have not reached time of the timer... {last_challenge_time}")
        time.sleep(40)
