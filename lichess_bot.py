import time
import random
import json
import logging

# Set up logging
logging.basicConfig(filename='requests.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

class LichessBot:
    def __init__(self, session, client):
        self.session = session
        self.client = client
        self.session_counter = 0
        self.last_challenge_time = time.time()

    class Game:
        def get_game(self, username, challenge_id, session_counter):
            logging.info(f"Accepted challenge from {username} (Session: {session_counter})")

    def get_game(self):
        try:
            response = self.session.get("https://lichess.org/api/account/playing")
            if response.status_code != 200:
                logging.error(f"Error getting games: {response.text}")
                time.sleep(10)
                return None

            games = response.json().get('nowPlaying', [])
            if games:
                return games
            time.sleep(5)
            return None

        except Exception:
            logging.error("Exception occurred", exc_info=True)

    def handle_challenge(self, challenge):
        username = challenge['challenger']['id']
        challenge_id = challenge['id']
        time_control = challenge['timeControl']
        variant = challenge['variant']['key']
        games = self.get_game()
        
        if games is not None:
            logging.info(f"Declined challenge from {username} due to being too busy at the moment.")
            try:
                self.client.challenges.decline(challenge_id)
            except Exception as e:
                pass
            return

        if variant != 'standard':
            logging.info(f"Declined challenge from {username} due to variant {variant}")
            try:
                self.client.challenges.decline(challenge_id)
            except Exception as e:
                pass
            return

        if time_control['type'] == 'clock' and time_control['limit'] < 13 * 60:
            logging.info(f"Declined challenge from {username} due to time control below 13 minutes")
            try:
                self.client.challenges.decline(challenge_id)
            except Exception as e:
                pass
            return

        if username != "vladK17":
            try:
                self.client.challenges.accept(challenge_id)
            except Exception as e:
                pass
                return
            game = self.Game()
            game.get_game(username, challenge_id, self.session_counter)
            self.session_counter += 1
        else:
            try:
                self.client.challenges.decline(challenge_id)
            except Exception as e:
                pass
            logging.info(f"Declined challenge from {username}")

    def manage_games(self):
        time_difference = time.time() - self.last_challenge_time
        if time_difference >= 600:  # 10 minutes
            try:
                response = self.session.get('https://lichess.org/api/bot/online')
                online_bots = [json.loads(line) for line in response.text.splitlines() if 'GuineaBot3' not in line]
                games = self.get_game()
                if games is None:
                    bot_to_play = random.choice(online_bots)
                    time_control_minutes = random.randint(13, 50)
                    logging.info(f"minutes: {time_control_minutes}")
                    time_control = time_control_minutes * 60
                    randvariant = 'standard'

                    try:
                        logging.info(f"Challenging {bot_to_play['id']} with time {time_control_minutes} for increment of minutes and variant {randvariant}")
                        if random.randint(1, 8) == 1:
                            rated = False
                        else:
                            rated = False  # Currently, rated is always false due to debugging and beta-testing.

                        if random.randint(1, 4) in [1, 2, 3]:
                            level = random.randint(1, 8)
                            logging.info(f"Challenging stockfish with time {time_control_minutes} for increment of minutes and level of {level}")
                            self.client.challenges.create_ai(level=level, clock_limit=time_control_minutes, clock_increment=0)
                        else:
                            self.client.challenges.create(bot_to_play['id'], variant=randvariant, clock_limit=time_control, clock_increment=0, rated=rated)
                    except Exception:
                        time.sleep(90)

                    self.last_challenge_time = time.time()  # Reset the timestamp for the last challenge.
                else:
                    logging.info("There is a game going on currently, sleeping...")
                    time.sleep(90)
            except Exception:
                logging.error("Exception occurred in manage_games", exc_info=True)

    def handle_incoming_challenges(self):
        for event in self.client.board.stream_incoming_events():
            if event['type'] == 'challenge':
                self.handle_challenge(event['challenge'])
                self.last_challenge_time = time.time()

    def start(self):
        while True:
            self.handle_incoming_challenges()
            self.manage_games()
            time.sleep(1)
