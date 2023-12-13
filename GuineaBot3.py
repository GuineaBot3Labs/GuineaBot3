import berserk
import requests
import chess
import chess.pgn
import io
from tqdm import tqdm
from art import print_acsii_art
import chess.engine
import copy
import platform
from timeout_decorator import timeout
from collections import OrderedDict
import torch
import TOM as T
import traceback
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import traceback
import numpy as np
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import matplotlib
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import gc
import pickle
import time
import sys
from functools import wraps
import threading
import random
from collections import deque

try:

    class BreakLoopException(Exception):
        pass
        
    class Attention(nn.Module):
        def __init__(self, in_channels):
            super(Attention, self).__init__()
            self.attention1 = nn.MultiheadAttention(embed_dim=in_channels, num_heads=12)
            self.attention2 = nn.MultiheadAttention(embed_dim=in_channels, num_heads=12)


        def forward(self, x):
            # x shape: [batch, channels, height, width]
        
            x = x.view(x.size(0), x.size(1), -1).permute(2, 0, 1)

            # Apply attention
            x, _ = self.attention1(x, x, x)
            x, _ = self.attention2(x, x, x)



            # Combine attention outputs (you can also concatenate or use other methods)
            attn_output = x

            # Reshape back
            attn_output = attn_output.view(-1, 96, 1, 1)


            
            return attn_output



    class ChessNet(nn.Module):
        def __init__(self, num_convs=3, num_fcs=7):
            super(ChessNet, self).__init__()

            num_output_actions = 4672  # Rough estimation of unique moves in chess
            self.attention = Attention(96)  # Assuming the input dimension is 4096

            # Convolutional layers and their corresponding Batch Normalization layers
            self.convs = nn.ModuleList()
            self.conv_ins = nn.ModuleList()
            in_channels = 14
            out_channels = 24
            for _ in range(num_convs):
                self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                self.conv_ins.append(nn.InstanceNorm2d(out_channels))
                in_channels = out_channels
                out_channels *= 2

            # Compute the output size of the conv layers by doing a forward pass with a dummy tensor
            x = torch.zeros(1, 14, 8, 8)  # Dummy input (batch_size=1, channels=12, height=8, width=8)
            for conv in self.convs:
                x = F.relu(conv(x))
                x = F.max_pool2d(x, kernel_size=2, stride=2)
                
            fc_input_size = x.numel()  # Total number of elements in 'x'

            # Fully connected layers and their corresponding Batch Normalization layers
            self.fcs = nn.ModuleList()
            self.fc_lns = nn.ModuleList()
            out_features = 4096
            for i in range(num_fcs):
                if i == 0:
                    self.fcs.append(nn.Linear(4096, out_features))  # Assuming fc_input_size is 4096
                else:
                    self.fcs.append(nn.Linear(out_features, out_features))
                self.fc_lns.append(nn.LayerNorm(out_features))

            # Output layer
            self.output_layer = nn.Linear(out_features, num_output_actions)  # Assuming num_output_actions is 4672

        


        def forward(self, x):
            # Pass input through each convolutional layer
            for conv, bn in zip(self.convs, self.conv_ins):
                x = F.relu(bn(conv(x)))
                x = F.max_pool2d(x, kernel_size=2, stride=2)
            # Calculate the input size for the first fully connected layer
            fc_input_size = x.numel() // x.size(0)  # We divide by batch size to get the size for a single sample

            # Update the first fully connected layer's input size
            self.fcs[0] = nn.Linear(fc_input_size, 4096).to(x.device)
            

            x = self.attention(x)
            
            # Flatten output from convolutional layers
            x = x.view(x.size(0), -1)

            # Pass flattened output through fully connected layers
            for fc, bn in zip(self.fcs, self.fc_lns):
                x = F.relu(bn(fc(x)))
            # Output layer
            x = self.output_layer(x)


            
            return x
            
        def mutate(self, device):
        
            self.to(device)
            min_fcs = 13  # Minimum number of fully connected layers
            max_fcs = 26  # Maximum number of fully connected layers
            min_convs = 3  # Minimum number of convolutional layers
            max_convs = 6  # Maximum number of convolutional layers
            noise = 0.5
    
            # Decide to add or remove a fully connected layer
            if random.random() < 0.1:
                if len(self.fcs) < max_fcs and random.choice([True, False]):
                    # Add a new fully connected layer with random weights
                    self.fcs.append(nn.Linear(4096, 4096))
                elif len(self.fcs) > min_fcs:
                    # Remove a fully connected layer
                    del self.fcs[-1]
    
            # Decide to add or remove a convolution layer
            if random.random() < 0.1:
                if len(self.convs) < max_convs and random.choice([True, False]):
                    # Add a new convolution layer with random weights
                    self.convs.append(nn.Conv2d(14, 24, kernel_size=3, padding=1))
                elif len(self.convs) > min_convs:
                    # Remove a convolution layer
                    del self.convs[-1]
    
            # Mutate existing weights
            with torch.no_grad():
                for param in self.parameters():
                    if random.random() < 0.1:
                        param += torch.randn_like(param) * noise

            self.to(device)
            


    class TargetChessNet(nn.Module):
        def __init__(self, num_convs=3, num_fcs=7):
            super(TargetChessNet, self).__init__()

            num_output_actions = 4672  # Rough estimation of unique moves in chess
            self.attention = Attention(96)  # Assuming the input dimension is 4096

            # Convolutional layers and their corresponding Batch Normalization layers
            self.convs = nn.ModuleList()
            self.conv_ins = nn.ModuleList()
            in_channels = 14
            out_channels = 24
            for _ in range(num_convs):
                self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                self.conv_ins.append(nn.InstanceNorm2d(out_channels))
                in_channels = out_channels
                out_channels *= 2

            # Compute the output size of the conv layers by doing a forward pass with a dummy tensor
            x = torch.zeros(1, 14, 8, 8)  # Dummy input (batch_size=1, channels=12, height=8, width=8)
            for conv in self.convs:
                x = F.relu(conv(x))
                x = F.max_pool2d(x, kernel_size=2, stride=2)
            fc_input_size = x.numel()  # Total number of elements in 'x'

            # Fully connected layers and their corresponding Batch Normalization layers
            self.fcs = nn.ModuleList()
            self.fc_lns = nn.ModuleList()
            out_features = 4096
            for i in range(num_fcs):
                if i == 0:
                    self.fcs.append(nn.Linear(4096, out_features))  # Assuming fc_input_size is 4096
                else:
                    self.fcs.append(nn.Linear(out_features, out_features))
                self.fc_lns.append(nn.LayerNorm(out_features))

            # Output layer
            self.output_layer = nn.Linear(out_features, num_output_actions)  # Assuming num_output_actions is 4672

        


        def forward(self, x):
            # Pass input through each convolutional layer
            for conv, bn in zip(self.convs, self.conv_ins):
                x = F.relu(bn(conv(x)))
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        
            # Calculate the input size for the first fully connected layer
            fc_input_size = x.numel() // x.size(0)  # We divide by batch size to get the size for a single sample

            # Update the first fully connected layer's input size
            self.fcs[0] = nn.Linear(fc_input_size, 4096).to(x.device)

            x = self.attention(x)
            # Flatten output from convolutional layers
            x = x.view(x.size(0), -1)

            # Pass flattened output through fully connected layers
            for fc, bn in zip(self.fcs, self.fc_lns):
                x = F.relu(bn(fc(x)))

            # Output layer
            x = self.output_layer(x)

            return x

        def mutate(self, device):
            self.to(device)
            min_fcs = 10  # Minimum number of fully connected layers
            max_fcs = 26  # Maximum number of fully connected layers
            min_convs = 3  # Minimum number of convolutional layers
            max_convs = 6  # Maximum number of convolutional layers
            noise = 0.5

            # Decide to add or remove a fully connected layer
            if random.random() < 0.1:
                if len(self.fcs) < max_fcs and random.choice([True, False]):
                    # Assuming all fully connected layers have 4096 units
                    fc_input_size = 4096
                    # Add a new fully connected layer with random weights
                    self.fcs.append(nn.Linear(fc_input_size, 4096))
                elif len(self.fcs) > min_fcs:
                    # Remove a fully connected layer
                    del self.fcs[-1]

            # Decide to add or remove a convolution layer
            if random.random() < 0.1:
                if len(self.convs) < max_convs and random.choice([True, False]):
                    # Get the number of output channels from the last Conv2D layer
                    last_out_channels = self.convs[-1].out_channels
                    # Add a new convolution layer with random weights
                    self.convs.append(nn.Conv2d(last_out_channels, last_out_channels * 2, kernel_size=3, padding=1))
                elif len(self.convs) > min_convs:
                    # Remove a convolution layer
                    del self.convs[-1]

            # Mutate existing weights
            with torch.no_grad():
                for param in self.parameters():
                    if random.random() < 0.1:
                        param += torch.randn_like(param) * noise

            self.to(device)

    class DQNAgent:
        def __init__(self, alpha=0.3, gamma=0.985, epsilon=0.5, epsilon_min=0.001, epsilon_decay=0.995, pgn=True, vebrose=False, device='cpu'):
            self.alpha = alpha
            self.gamma = gamma
            self.pgn = pgn
            self.vebrose = vebrose
            # Create the device variable, will work with 'cuda:0', 'cuda:1', and 'cpu'
            self.device = torch.device(device)

            # Create the online model and the target model
            self.model = ChessNet()
            self.target_model = TargetChessNet()

            # Move models to device
            self.model = self.model.to(self.device)
            self.target_model = self.target_model.to(self.device)

            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.memory = []
            self.move_rewards = []
            self.short_term_memory = []  # Initialize short_term_memory
            self.optimizer = optim.Adam(self.model.parameters(), lr=alpha, weight_decay=0.01)
            self.loss_fn = nn.MSELoss()
            self.session = requests.Session()
            self.token = 'YOUR-API-TOKEN'
            self.name = 'YOUR-USERNAME'
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            self.client = berserk.Client(berserk.TokenSession(self.token))

            self.game_id = None
            self.game_over = False
            self.game_over_count = 0
            self.opponent_username = None
            self.color = None
            self.error = None
            self.is_draw = False
            self.is_stalemate = False
            self.lastfen = None
            self.backupfen = None
            self.plot = False
            self.opponent_move = None
            self.repeat_count = 0
            self.Last_Move = None
            self.my_color = None
            self.best_reward = 0
            self.batch_size = 270
            self.states = []
            self.next_states = []
            self.actions = []
            self.current_move = False


            print(f"Using device: {device}!")
            # self.model = torch.nn.parallel.DistributedDataParallel(self.target_model)
            # self.target_model = torch.nn.parallel.DistributedDataParallel(self.target_model)


            # self.model.to(self.device)
            # self.target_model.to(self.device)
            time.sleep(1)

        def load_model_weights_both(self, model_path):
            model_weights_before = {name: param.clone() for name, param in self.model.named_parameters()}
            checkpoint = torch.load(model_path)
            self.model.train()
            self.target_model.train()
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.eval()
            self.target_model.eval()
 
            # Compare the weights
            for name, param in self.model.named_parameters():
                diff = model_weights_before[name] - param
                if torch.any(diff):
                    print(f"\nModel weights for {name} are not the same!")
                else:
                    print(f"\nModel weights for {name} are unchanged.")
            self.replay_from_file(board)
            print("Done!")
        
        @timeout(5)
        def stream_game(self, board):
            if self.game_over == False:
                moves = self.client.games.stream_game_moves(self.game_id)
                for line in moves:
                    if self.vebrose:
                        print(line)

                    self.backupfen = line.get('fen') # Get the backup FEN in case of invalid move error
                    possible = [line.get(key) for key in ['lm', 'lastMove', 'LastMove', 'lastmove']]
                    if self.backupfen == 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' and self.color == None:
                        self.repeat_count += 1
                        board.turn = self.color



                    if self.repeat_count > 60:
                        
                        time.sleep(5)
                        
                    lm = next((m for m in possible if m is not None), None)  # Get the first non-None move
                    
                    # Process both 'lm' and 'LastMove' if they exist
                    for move in [lm]:
                        
                        
                        game_status = line.get('status', {}).get('name')  # Get the game status
                        if self.vebrose:

                            print("DEBUG: game status: ", game_status)
                        
                            print("checking checkpoint 1...")
                         
                            print("checking checkpoint 2...")
                        if game_status == 'draw':
                            if self.vebrose:
                             
                                print("Game is over. Pausing for 5 seconds.")
                            self.is_draw = True
                        if self.vebrose:
                        
                            print("checking checkpoint 3...")
                        if game_status == 'stalemate':
                            board.set_fen(self.backupfen)
                            if self.vebrose:

                                print("Game is over. Pausing for 5 seconds.")
                            self.is_stalemate = True
                            self.is_draw = True
                        if self.vebrose:
                        
                            print("checking checkpoint 4...")
                        if game_status == 'aborted':
                            board.set_fen(self.backupfen)
                            if self.vebrose:

                                print("Game is over. Pausing for 5 seconds.")
                            self.game_over = True

                        if game_status == 'outoftime':
                            if self.vebrose:

                                print("Game is over. Pausing for 5 seconds.")
                            self.game_over = True
                            self.is_draw = True
                            time.sleep(5)          

                        if game_status == 'resign':
                            if self.vebrose:

                                print("Game is over. Pausing for 5 seconds.")
                            self.game_over = True
                            self.is_draw = True
                            time.sleep(5)                  
                        if self.vebrose:
      
                            print("checking checkpoint 5...")
                        self.repeat_count += 1
                        if self.vebrose:

                            print(f"DEBUG: Repeat Count: {self.repeat_count}")
                        
                        if board.turn == self.color:
                            if self.vebrose:

                                print("Accidentally ran self.stream_game() though my turn...")
                            self.current_move = True
                            return
                            
                        if self.repeat_count > 3:
                            if game_status == 'mate':
                                self.game_over = True
                            else:
                                pass
                        if self.repeat_count > 60:
                            time.sleep(5)
                            self.error = True
                            self.game_over = True

                        else:
                            if self.vebrose:

                                print("checking checkpoint 6...")

                                print(f"DEBUG Opponent Move: {move}")
                                print(f"DEBUG Last Move From Opponent: {self.Last_Move}")
                            self.Last_Move = move
                            if self.vebrose:

                                print("check complete!")
                            
                            self.opponent_move = move
                            
                            return move

        def make_move(self, move):
            try:
                self.client.bots.make_move(self.game_id, move)
            except Exception:
                pass
        def clearscreen(self):
            x = platform.platform()
            if x.startswith("Linux"):
                os.system('clear')
            elif x.startswith("Windows"):
                os.system('cls')
            else:
                os.system('clear')
                
        def disable_function(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                print(f"The function {func.__name__} is disabled.")
                return None  # or some default value
            return wrapper

        def get_game(self, board):
            while True:
                try:
                    response = self.session.get("https://lichess.org/api/account/playing")
                    if response.status_code != 200:
                        print(f"Error getting games: {response.text}")
                        time.sleep(10)
                        continue

                    games = response.json()['nowPlaying']
                    # while not games:
                        # print("No games found, awaiting challenges (if any), or challenging a bot...")
                        # self.handle_incoming_challenges()
                        # time.sleep(600)
                        # self.challenge_bot(board)
                        # continue

                    for game in games:
                        if game['isMyTurn']:
                            self.game_id = game['gameId']
                            self.opponent_username = game['opponent']['username']
                            self.color = chess.WHITE if game['color'] == 'white' else chess.BLACK
                            self.my_color = "White" if self.color == chess.WHITE else "Black"

                            return

                    print("No games where it's my turn, sleeping...")
                    time.sleep(20)
                except Exception:
                    pass
        
        def simulate_self_play(self, num_games):
                self.memory_white = []
                self.memory_black = []
                self.short_term_memory_white = []
                self.short_term_memory_black = []
                for game_num in tqdm(range(num_games), desc="Doing self play"):
                    board2 = chess.Board()
                    board2.set_fen(chess.STARTING_FEN)
                    board2.turn = chess.WHITE
                    self.color = chess.WHITE
                    move_num = 0
                    while not board2.is_checkmate() and not board2.is_game_over() and not board2.is_stalemate() and not board2.is_insufficient_material() and not board2.is_seventyfive_moves() and not board2.is_variant_draw(): 
                        state = self.board_to_state(board2)
                        move_num += 1
                        move = self.choose_action(state, list(board2.legal_moves), board2, True, move_num)
                        # print(f"Best move determined by GuineaBot3: {move}")
                        board2.push(move)
                        self.color = board2.turn
                        self.clearscreen()
                        self.print_board(board2)
                                
                    if len(self.memory_white) >= self.batch_size:
                        if self.vebrose:
                            print("Now commencing training stage 2 (may take a while, read a book or watch tv or something, I really don't care.)")
                        self.replay(batch_size, board, True, chess.WHITE)
        
                        self.memory_white = []
                        self.short_term_memory_white = []
                        # Clear the GPU cache
                        gc.collect()
                        if self.device.startswith('cuda'):
                            torch.cuda.empty_cache()
                    elif len(self.memory_black) >= self.batch_size:
                        if self.vebrose:
                            print("Now commencing training stage 2 (may take a while, read a book or watch tv or something, I really don't care.)")
                        self.replay(batch_size, board, True, chess.BLACK)
        
                        self.memory_black = []
                        self.short_term_memory_black = []
                        # Clear the GPU cache
                        gc.collect()
                        if self.device.startswith('cuda'):
                            torch.cuda.empty_cache()
                    else:
                        self.short_term_memory_white = []
                        self.short_term_memory_black = []
                
        def replay_pgn_and_learn(self, file_path):
            try:
                board1 = chess.Board()
                game_count = 0
                while True:
                    try:
                        num_games = input("How many games do you want GuineaBot3 to selfplay?: ")
                        num_games = int(num_games)
                        break
                    except ValueError:
                        print("Invalid input. Please enter an number.")
                self.simulate_self_play(num_games)
                with open(file_path, 'r') as pgn_file:
                    while True:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break  # Exit if no more games

                        game_count += 1
                        print(f"Processing game {game_count}", end='\r')

                        board1.reset()
                        moves = list(game.mainline_moves())
                        for move in tqdm(moves, desc=f"Processing moves of game: {game_count}"):
                            state = self.board_to_state(board1)
                            board1.push(move)
                            original_piece_type = board1.piece_at(move.from_square).piece_type if board1.piece_at(move.from_square) else None
                            reward = self.get_reward(board1, board1.turn, move, original_piece_type, True)  # Adapt reward function if needed
                            done = board1.is_game_over()
                            next_state = self.board_to_state(board1)
                            self.update_model(state, move, reward)
                            self.remember(state, move, reward, next_state, done, True, board.turn)
                        if len(self.memory_white) >= self.batch_size:
                            print("Now commencing replay (may take a while)")
                            self.replay(self.batch_size, board1, True, chess.WHITE)

                            self.memory_white = []
                            self.short_term_memory_white = []
                            # Clear the GPU cache
                            gc.collect()
                            if self.device.startswith('cuda'):
                                torch.cuda.empty_cache()
                        elif len(self.memory_black) >= self.batch_size:
                            print("Now commencing replay (may take a while)")
                            self.replay(self.batch_size, board1, True, chess.BLACK)

                            self.memory_black = []
                            self.short_term_memory_black = []
                            # Clear the GPU cache
                            gc.collect()
                            if self.device.startswith('cuda'):
                                torch.cuda.empty_cache()
                        else:
                            self.short_term_memory_white = []
                            self.short_term_memory_black = []

                print("\nAll games processed. Total games:", game_count)
                self.simulate_self_play(100)
            except KeyboardInterrupt:
                pass


        def replay_from_file(self, board):
            try:
                with open("trainingdata.bin", "rb") as f:
                    data = pickle.load(f)
                    self.memory = data['memory']
                    self.short_term_memory = data['short_term_memory']

            except FileNotFoundError:
                print("File not found. Skipping replay from file.")


        def load_model_weights(self, model_path):
            # Load the original state dictionary
            original_state_dict = torch.load(model_path)

            # Create a new state dictionary without the 'module.' prefix
            new_state_dict = OrderedDict()
            for k, v in original_state_dict.items():
                name = k
                while name[:7] == 'module.':  # remove 'module.' prefix until it's gone
                    name = name[7:]
                new_state_dict[name] = v

            # Load the state dictionary into the model
            self.model.load_state_dict(new_state_dict, strict=False)
            self.target_model.load_state_dict(new_state_dict, strict=False)
            self.model.eval()
            self.target_model.eval()


            

        def move_to_index(self, board, move):
            from_square = move.from_square
            piece = board.piece_at(from_square)
            if piece is None:
                return None
            moved_piece = piece.piece_type
            self.color = int(piece.color)
            index = from_square * 12 + moved_piece - 1 + self.color * 6
            return index

        def print_board(self, board):
            symbols = {
                'r': '♖', 'n': '♘', 'b': '♗', 'q': '♕',
                'k': '♔', 'p': '♙', 'R': '♜', 'N': '♞',
                'B': '♝', 'Q': '♛', 'K': '♚', 'P': '♟', '.': '.'}

            str_board = str(board)
            for key, value in symbols.items():
                str_board = str_board.replace(key, value)
            print(str_board)

        def index_to_move(self, board, index):
            from_square = index // 12
            piece_type = index % 12 + 1
            if piece_type > 6:
                piece_type -= 6
                self.color = chess.WHITE
            else:
                self.color = chess.BLACK
            piece = board.piece_at(from_square)
            if piece is not None and piece.piece_type == piece_type and piece.color == self.color:
                legal_moves = list(board.legal_moves)
                for move in legal_moves:
                    if move.from_square == from_square:
                        return move

        def remember(self, state, action, reward, next_state, done, selfplay=False, turn=None):
            if selfplay:
                if turn == chess.WHITE:
                    self.short_term_memory_white.append((state, action, reward, next_state, done))
                else:
                    self.short_term_memory_black.append((state, action, reward, next_state, done))
            else:
                self.short_term_memory.append((state, action, reward, next_state, done))
            if selfplay:
                if turn == chess.WHITE:
                    if np.random.rand() < 0.4 and not self.batch_size <= len(self.memory_white):  # 40% chance to remember the move in long term
                        self.memory_white.append((state, action, reward, next_state, done))
                else:
                    if np.random.rand() < 0.4 and not self.batch_size <= len(self.memory_black):  # 40% chance to remember the move in long term
                        self.memory_black.append((state, action, reward, next_state, done))
            else:
                if np.random.rand() < 0.4 and not self.batch_size <= len(self.memory):  # 40% chance to remember the move in long term
                    self.memory.append((state, action, reward, next_state, done))
                
        def update_model(self, state, action, reward):

            action_index = self.move_to_index(board, action)
            target_f = self.model(state).detach().clone()
            target_f = target_f.to(self.device)  # Move target_f to device
            target_f[0, action_index] = reward

            loss = self.loss_fn(target_f, self.model(state))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            target_f2 = self.target_model(state).detach().clone().to(self.device)
            target_f2[0, action_index] = reward
            loss2 = self.loss_fn(target_f2, self.target_model(state).to(self.device))
            self.optimizer.zero_grad()
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(self.target_model.parameters(), max_norm=1.0)
            self.optimizer.step()


        def model_learn(self, state, opponent_move, reward):

            ### Update the main model ###
            opponent_move_index = self.move_to_index(board, opponent_move)
            target_f2 = self.model(state).detach().clone()
            target_f2 = target_f2.to(self.device)  # Move target_f to device
    
            # Update the Q-value for the opponent's move with the negative reward
            target_f2[0, opponent_move_index] = -reward
        
            loss = self.loss_fn(target_f2, self.model(state))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            ### Update the target model ###
            target_f2 = self.target_model(state).detach().clone().to(self.device)
            target_f2[0, opponent_move_index] = reward
            loss2 = self.loss_fn(target_f2, self.target_model(state).to(self.device))
            self.optimizer.zero_grad()
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(self.target_model.parameters(), max_norm=1.0)
            self.optimizer.step()


        def choose_action(self, state, legal_moves, board, selfplay=False, move_num=0):
                try:
                        while legal_moves:
                                if torch.rand(1) <= self.epsilon:
                                        if self.vebrose:
                                            print("DEBUG: EXPLORATION MOVE")
                                        random_move = self.choose_actionrandom(state, legal_moves, board, selfplay)
                                        return random_move

                                else:
                                        if self.vebrose:
                                            print("DEBUG: NOT EXPLORATION MOVE")
                                        self.model.eval()
                                        self.target_model.eval()
                                        state = state.to(self.device)  # Move the state to GPU

                                        q_values = self.model(state).detach().view(-1)
                                        target_q_values = self.target_model(state).detach().view(-1)
                                        combined_q_values = (q_values + target_q_values) / 2

                                        legal_move_indices = [self.move_to_index(board, move) for move in legal_moves]
                                        legal_q_values = torch.tensor([combined_q_values[i] for i in legal_move_indices if i is not None and i < len(combined_q_values)])
                                        self.best_reward = 0

                                        move_index = legal_move_indices[torch.argmax(legal_q_values).item()]
                                        move = self.index_to_move(board, move_index)
                                        if self.vebrose:
                                            print(f"{i}st Move calculated from bot: {move}\n")
                                        self.model.train()
                                        done = board.is_game_over()
                                        original_piece_type = board.piece_at(move.from_square).piece_type if board.piece_at(move.from_square) else None
                                        board.push(move)
                                        next_state = self.board_to_state(board)
                                        next_state.to(self.device)
                                        reward = self.get_reward(board, self.color, move, original_piece_type, selfplay, move_num)
                                        if np.isnan(reward) or np.isinf(reward):
                                            reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
                                        self.move_rewards.append(reward)
                                        best_move = move
                                        if self.vebrose:
                                            print(f"DEBUG: Rewards: {self.move_rewards}")
                                        if self.plot:
                                            self.line.set_ydata(self.move_rewards)
                                            self.line.set_xdata(range(len(self.move_rewards)))
                                            plt.xlim(0, len(self.move_rewards))
                                            plt.ylim(min(self.move_rewards), max(self.move_rewards))
                                            plt.draw()
                                            plt.pause(0.001)
                                        board.pop()
                                        self.update_model(state, move, reward)
                                        self.remember(state, move, reward, next_state, done, selfplay, board.turn)
                                        if not selfplay:
                                            board.push(best_move)
                                            
                                        return best_move
                except KeyboardInterrupt:
                        board.turn == self.color
                        manual_move = input("Enter your move in UCI format: ")
                        best_move = chess.Move.from_uci(manual_move)
                        self.model.train()
                        state = self.board_to_state(board)
                        done = board.is_game_over()
                        original_piece_type = board.piece_at(best_move.from_square).piece_type if board.piece_at(best_move.from_square) else None
                        self.remember(state, best_move, reward, next_state, done, selfplay, board.turn)
                        if not selfplay:
                            board.push(best_move)
                        reward = self.get_reward(board, self.color, best_move, original_piece_type, selfplay)
                        next_state = self.board_to_state(board)
                        next_state.to(self.device)
                        self.update_model(state, best_move, reward)

                        if self.vebrose:
                            print(f"DEBUG: Rewards: {reward}")
                        return best_move

        def choose_actionrandom(self, state, legal_moves, board, selfplay=False):
                while legal_moves:
                        self.model.eval()
                        self.target_model.eval()
                        state = self.board_to_state(board)
                        state = state.to(self.device)  # Move the state to GPU


                        for i in range(11):

                            move = random.choice(list(board.legal_moves))
                            if self.vebrose:
                                print(f"{i}st move from bot: {move}\n")
                            self.model.train()
                            done = board.is_game_over()
                            original_piece_type = board.piece_at(move.from_square).piece_type if board.piece_at(move.from_square) else None
                            board.push(move)
                            reward = self.get_reward(board, self.color, move, original_piece_type, selfplay)
                            if np.isnan(reward) or np.isinf(reward):
                                reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
                            self.move_rewards.append(reward)
                            next_state = self.board_to_state(board)
                            next_state.to(self.device)
                            best_move = move
                            best_state = next_state
                            self.best_reward = reward


                            board.pop()
                        
                        if best_move is not None:
                            if self.vebrose:
                                print(f"DEBUG: Rewards: {self.move_rewards}")
                            if self.plot:
                                self.line.set_ydata(self.move_rewards)
                                self.line.set_xdata(range(len(self.move_rewards)))
                                plt.xlim(0, len(self.move_rewards))
                                plt.ylim(min(self.move_rewards), max(self.move_rewards))
                                plt.draw()
                                plt.pause(0.001)
                            self.update_model(state, best_move, reward)
                            self.remember(state, move, reward, best_state, done, selfplay, board.turn)
                            if not selfplay:
                                board.push(best_move)
                            return best_move
                        else:
                            move = best_move
                            if self.vebrose:
                                print(f"DEBUG: Rewards: {self.move_rewards}")
                            if self.plot:
                                self.line.set_ydata(self.move_rewards)
                                self.line.set_xdata(range(len(self.move_rewards)))
                                plt.xlim(0, len(self.move_rewards))
                                plt.ylim(min(self.move_rewards), max(self.move_rewards))
                                plt.draw()
                                plt.pause(0.001)
                            self.update_model(state, best_move, reward)
                            self.remember(state, move, reward, best_state, done, selfplay, board.turn)
                            board.push(best_move)
                            return best_move


        def replay(self, batch_size, board, selfplay=False, color=None):
            if self.vebrose:
                print("DEBUG: Starting replay function...")
                if selfplay:
                    print("DEBUG: Running self play version")
            if not selfplay:
                # Sample from long-term memory (self.memory)
                long_term_batch = random.sample(self.memory, batch_size // 2)

                # Sample from short-term memory (self.short_term_memory)
                short_term_batch = random.sample(self.short_term_memory, min(len(self.short_term_memory), batch_size // 2))

                # Combine both batches
                batch = long_term_batch + short_term_batch

                for state, action, reward, next_state, done in tqdm(batch, desc="Processing batch"):
                    state = torch.tensor(state, dtype=torch.float)
                    next_state = torch.tensor(next_state, dtype=torch.float)
                    reward = torch.tensor(reward, dtype=torch.float)

                    if done:
                        target = reward
                    else:
                        # Double Q-Network update
                        with torch.no_grad():
                            next_state_q_values = self.target_model(next_state).view(-1)
        
                        next_state_legal_move_indices = [self.move_to_index(board, move) for move in board.legal_moves]
        
                        next_state_legal_q_values = torch.tensor([next_state_q_values[i] for i in next_state_legal_move_indices if i is not None and i < len(next_state_q_values)])
                        if next_state_legal_q_values.nelement() == 0:
                            if self.vebrose:
                                print("DEBUG: next_state_legal_q_values is empty")
                            continue
                        else:
                            best_next_action_index = next_state_legal_move_indices[torch.argmax(next_state_legal_q_values).item()]
                            target = reward + self.gamma * self.model(next_state).view(-1)[best_next_action_index]
            
                            # Convert move to a unique index
                            action_index = self.move_to_index(board, action)
        
                            target_f = self.model(state).detach().clone()
                            target_f[0, action_index] = target
        
        
                            loss = self.loss_fn(target_f, self.model(state))
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
            else:
                if color == chess.WHITE:
                    # Sample from long-term memory (self.memory_white)
                    print(len(self.memory_white))
                    long_term_batch = random.sample(self.memory_white, batch_size // 2)

                    # Sample from short-term memory (self.short_term_memory_white)
                    short_term_batch = random.sample(self.short_term_memory_white, min(len(self.short_term_memory_white), batch_size // 2))

                    # Combine both batches
                    batch = long_term_batch + short_term_batch

                    for state, action, reward, next_state, done in tqdm(batch, desc="Processing white batch"):
                        state = torch.tensor(state, dtype=torch.float)
                        next_state = torch.tensor(next_state, dtype=torch.float)
                        reward = torch.tensor(reward, dtype=torch.float)

                        if done:
                            target = reward
                        else:
                            # Double Q-Network update
                            with torch.no_grad():
                                next_state_q_values = self.target_model(next_state).view(-1)
        
                            next_state_legal_move_indices = [self.move_to_index(board, move) for move in board.legal_moves]
        
                            next_state_legal_q_values = torch.tensor([next_state_q_values[i] for i in next_state_legal_move_indices if i is not None and i < len(next_state_q_values)])
                            if next_state_legal_q_values.nelement() == 0:
                                if self.vebrose:
                                    print("DEBUG: next_state_legal_q_values is empty")
                                continue
                            else:
                                best_next_action_index = next_state_legal_move_indices[torch.argmax(next_state_legal_q_values).item()]
                                target = reward + self.gamma * self.model(next_state).view(-1)[best_next_action_index]
        
                                # Convert move to a unique index
                                action_index = self.move_to_index(board, action)
        
                                target_f = self.model(state).detach().clone()
                                target_f[0, action_index] = target
        
        
                                loss = self.loss_fn(target_f, self.model(state))
                                self.optimizer.zero_grad()
                                loss.backward()
                                self.optimizer.step()
                else:
                    # Sample from long-term memory (self.memory_black)
                    print(len(self.memory_black))
                    long_term_batch = random.sample(self.memory_black, batch_size // 2)

                    # Sample from short-term memory (self.short_term_memory)
                    short_term_batch = random.sample(self.short_term_memory_black, min(len(self.short_term_memory_black), batch_size // 2))

                    # Combine both batches
                    batch = long_term_batch + short_term_batch

                    for state, action, reward, next_state, done in tqdm(batch, desc="Processing black batch"):
                        state = torch.tensor(state, dtype=torch.float)
                        next_state = torch.tensor(next_state, dtype=torch.float)
                        reward = torch.tensor(reward, dtype=torch.float)
    
                        if done:
                            target = reward
                        else:
                            # Double Q-Network update
                            with torch.no_grad():
                                next_state_q_values = self.target_model(next_state).view(-1)
            
                            next_state_legal_move_indices = [self.move_to_index(board, move) for move in board.legal_moves]
        
                            next_state_legal_q_values = torch.tensor([next_state_q_values[i] for i in next_state_legal_move_indices if i is not None and i < len(next_state_q_values)])
                            if next_state_legal_q_values.nelement() == 0:
                                if self.vebrose:
                                    print("DEBUG: next_state_legal_q_values is empty")
                                continue
                            else:
                                best_next_action_index = next_state_legal_move_indices[torch.argmax(next_state_legal_q_values).item()]
                                target = reward + self.gamma * self.model(next_state).view(-1)[best_next_action_index]
        
                                # Convert move to a unique index
                                action_index = self.move_to_index(board, action)
        
                                target_f = self.model(state).detach().clone()
                                target_f[0, action_index] = target
        
        
                                loss = self.loss_fn(target_f, self.model(state))
                                self.optimizer.zero_grad()
                                loss.backward()
                                self.optimizer.step()
        
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            if self.vebrose:
                print("DONE!")






        def train(self, episodes, batch_size, board):
            try:
                print_acsii_art()
                print("GuineaBot3 v4.1.9 (COMPACT EDITION, WHEEK WHEEK!!!), copyrighted (©) 2022 april 23")
                episode = 0
                counter = 0
                self.losses = 0
                self.draws = 0
                self.wins = 0
                game = 1
                if self.plot:
                    matplotlib.use('Qt5Agg')
                    plt.ion()
                    plt.figure(figsize=(5,5))
                    plt.title('Rewards')
                    plt.xlabel('Move')
                    plt.ylabel('Reward')
                    self.line, = plt.plot(self.move_rewards)
                try:
                    # Load the model weights if a model file exists
                    print("loading model weights...", end='', flush=True)

                    for i in range(100):
                        for cursor in '|/-\\':
                           sys.stdout.write(cursor)
                           sys.stdout.flush()
                           time.sleep(0.01)
                           sys.stdout.write('\b')


                    self.load_model_weights_both("agent1_model.pt")
                    print('\nloaded model weights!')
                    time.sleep(1)
                except Exception:
                    traceback.print_exc()
                    x = input("\nfailed to load weights, do you want to continue?: ")
                    if x == "n" or x == "no":
                        print("Bye!")
                        exit(0)
                    elif x == "y" or x == "yes":
                        pass
                    else:
                        print("enter a valid value, y or n")
                        exit(0)

                if self.pgn == True:
                    i = input("\nDo You want to train model on pgns? (y|n): ")
                    if i == "y":
                        q = input("\nwhat is the filepath?: ")
                        print("\nTraining, do not turn off server...")
                        try:
                            self.replay_pgn_and_learn(q)
                        except Exception as e:
                            print(f"failed, just going to run untrained: \n")
                            traceback.print_exc()
                    else:
                        print("Skipping...\n")
                while episodes > episode:

                    if len(self.move_rewards) >= 1000:	
                        self.move_rewards.clear()
                    self.game_over = False
                    self.is_stalemate = False
                    self.is_draw = False
                    self.repeat_count = 0
                    self.get_game(board)
                    board.turn = chess.WHITE
                    counter = 0
                    try:
                        self.client.bots.post_message(self.game_id, f"Hi! I am ", self.name, ", powered by GuineaBOTv4! I am a Learning model, please give feedback of my games, so my developer can improve me!", spectator=True)
                    except Exception:
                        pass
                    try:
                        self.client.bots.post_message(self.game_id, f"Hi! I am ", self.name, ", powered by GuineaBOTv4! I am a Learning model, please give feedback of my games, so my developer can improve me!", spectator=False)
                    except Exception:
                        pass
                    moves = 0
                    torch.autograd.set_detect_anomaly(True)

                    


                    while not board.is_checkmate() and not board.is_game_over() and not board.is_stalemate() and not board.is_insufficient_material() and not board.is_seventyfive_moves() and not board.is_variant_draw() and not self.game_over == True:
                        if board.turn == self.color:
                            self.repeat_count = 0
                            if len(self.move_rewards) >= 200:
                                self.move_rewards = []
                            legal_moves = list(board.legal_moves)
                            if not legal_moves:
                                break
                            state = self.board_to_state(board)
                            action = self.choose_action(state, list(board.legal_moves), board)
                            moves += 1
                            self.make_move(action)                                            
                            self.clearscreen()
                            self.print_board(board)
                            if len(self.memory) >= batch_size:
                                print("WARNING: No more memory, training stage 2 is suspended until the end of the game")
                            else:
                                print("Current game: ", str(game))
                                print("Move count: ", str(moves))
                                print("Amount of wins: ", str(self.wins))
                                print("Amount of draws: ", str(self.draws))
                                print("Amount of losses: ", str(self.losses))
                                print("DEBUG: Memory size (long_term): ", len(self.memory))
                                print("DEBUG: Memory left: ", batch_size - len(self.memory))
                                print("DEBUG: Batch_Size: ", str(self.batch_size))
                                print("DEBUG: Color: ", self.color, " (", self.my_color, ")")
                                print(f"DEBUG: Epsilon: {self.epsilon}")
                        else:  # It's not the agent's turn
                            try:
                                start_time = time.time()
                                while True:
                                    try:
                                        move = self.stream_game(board)

                                        if move is not None:
                                            self.opponent_move = move
                                            self.get_opponent_move(board, counter)
                                            break

                      
                                    except Exception as e:
                                        if self.vebrose:
                                            print(f"something happened, checking: {e}")
                                            print(f"Caught an exception of type: {type(e)}")
                                        if self.error == True:
                                            try:
                                                self.client.bots.post_message(self.game_id, "I ran into a error, I am truly sorry...", spectator=False)
                                                self.client.bots.resign_game(self.game_id)
                                            except Exception:
                                                self.Last_Move = None
                                                self.lastfen = None
                                                try:
                                                    self.client.bots.resign_game(self.game_id)
                                                except Exception:
                                                    self.game_id = None
                                                    self.game_over = True
                                                    break
                                        elif self.is_draw == True:
                                            try:
                                                self.client.bots.post_message(self.game_id, "Tie!!", spectator=False)
                                                self.Last_Move = None
                                                self.lastfen = None
                                                self.game_id = None
                                                self.game_over = True
                                                self.repeat_count = 0
                                                break
                                            except Exception as e:
                                                self.Last_Move = None
                                                self.lastfen = None
                                                self.game_id = None
                                                self.game_over = True
                                                self.repeat_count = 0

                               
                                        elif isinstance(e, berserk.exceptions.ResponseError) or '429 Client Error: Too Many Requests for url:' in str(e):
                                            time.sleep(6)
                                        else:
                                            self.repeat_count = 0
                                            time.sleep(6)
                                            
                                    

                                    # Check if time without move exceeds 10 minutes
                                    if time.time() - start_time > 600:
                                        print("No move for 10 minutes, starting a new game...")
                                        try:
                                            self.client.bots.resign_game(self.game_id)
                                        except Exception as e:
                                            if batch_size <= len(self.memory):
                                                print("Now commencing training stage 2 (may take a while, read a book or watch tv or something, I really don't care.)")
                                                self.replay(batch_size, board)

                                                print("Saving/Updating model weights")
                                                # Save the model weights after each episode
                                                torch.save({
                                                'model_state_dict': self.model.state_dict(),
                                                'target_model_state_dict': self.target_model.state_dict(),
                                                'optimizer_state_dict': self.optimizer.state_dict(),
                                                # You can include more states here if needed
                                                }, "agent1_model.pt")
                                                self.memory = []
                                                # Clear the GPU cache
                                                gc.collect()
                                                if self.device.startswith('cuda'):
                                                    torch.cuda.empty_cache()
                        
                                            self.short_term_memory = []
                                            self.game_id = None
                                            self.Last_Move = None
                                            self.lastfen = None
                                            self.game_over = True

                                        break
                            except Exception:
                                pass    
 
                    if board.is_checkmate() and board.turn != self.color:
                        self.clearscreen()
                        self.print_board(board)
                        print(f"GuineaBOT Won!")
                        game += 1
                        self.wins += 1
                        moves = 0
                        self.Last_Move = None
                        self.lastfen = None
                        try:
                            self.client.bots.post_message(self.game_id, "Good try!", spectator = False)
                        except Exception:
                            pass
                        episode += 1
                        board.set_board_fen(chess.STARTING_BOARD_FEN)
                        if batch_size <= len(self.memory):
                            if self.vebrose:
                                print("Now commencing training stage 2 (may take a while, read a book or watch tv or something, I really don't care.)")
                            self.replay(batch_size, board)
                            if self.vebrose:

                                print("Saving/Updating model weights")
                            # Save the model weights after each episode
                            torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'target_model_state_dict': self.target_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            # You can include more states here if needed
                            }, "agent1_model.pt")
                            self.memory = []
                            gc.collect()
                            if self.device.startswith('cuda'):
                                torch.cuda.empty_cache()
                        self.short_term_memory = []
                        self.game_over = True

                    if board.is_checkmate() and board.turn == self.color:
                        self.clearscreen()
                        self.print_board(board)
                        print(f"GuineaBOT Lost!")
                        try:
                            self.client.bots.post_message(self.game_id, "Congradulations! You have successfully beaten me! Good job!", spectator = False)
                        except Exception:
                            pass
                        game += 1
                        self.losses += 1
                        moves = 0
                        self.Last_Move = None
                        self.lastfen = None
                        episode += 1
                        board.set_board_fen(chess.STARTING_BOARD_FEN)
                        if batch_size <= len(self.memory):
                            if self.vebrose:

                                print("Now commencing training stage 2 (may take a while, read a book or watch tv or something, I really don't care.)")
                            self.replay(batch_size, board)
                            if self.vebrose:

                                print("Saving/Updating model weights")
                            # Save the model weights after each episode
                            torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'target_model_state_dict': self.target_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            # You can include more states here if needed
                            }, "agent1_model.pt")
                            self.memory = []
                            # Clear the GPU cache
                            gc.collect()
                            if self.device.startswith('cuda'):
                                torch.cuda.empty_cache()
                        
                        self.short_term_memory = []
                        self.game_over = True

                    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.can_claim_threefold_repetition() or board.is_variant_draw():
                        print("Draw!")
                        board.set_board_fen(chess.STARTING_BOARD_FEN)
                        game += 1
                        moves = 0
                        self.Last_Move = None
                        self.lastfen = None
                        episode += 1
                        try:
                            self.client.bots.post_message(self.game_id, "Tie!", spectator = False)
                        except Exception:
                            pass
                        self.draws += 1

                        if batch_size <= len(self.memory):
                            if self.vebrose:

                                print("Now commencing training stage 2 (may take a while, read a book or watch tv or something, I really don't care.)")
                            self.replay(batch_size, board)
                            if self.vebrose:

                                print("Saving/Updating model weights")
                            # Save the model weights after each episode
                            torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'target_model_state_dict': self.target_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            # You can include more states here if needed
                            }, "agent1_model.pt")
                            self.memory = []
                            gc.collect()
                            if self.device.startswith('cuda'):
                                torch.cuda.empty_cache()
                        
                        
                        self.short_term_memory = []
                        self.game_over = True
                    
                    if self.game_over == True:
                        board.set_board_fen(chess.STARTING_BOARD_FEN)
                        moves = 0
                        self.game_over = False
        
        



                        
            except Exception:
                traceback.print_exc()

        def board_to_state(self, board):
            state = torch.zeros(1, 14, 8, 8)  # Extend to 14 channels
            piece_types = list(range(1, 7)) * 2  # 1-6 for BLACK, 1-6 for WHITE
            colors = [chess.BLACK] * 6 + [chess.WHITE] * 6
            piece_values = {
                chess.PAWN: 3,
                chess.KNIGHT: 6,
                chess.BISHOP: 4,
                chess.ROOK: 7,
                chess.QUEEN: 8,
                chess.KING: 5
            }

            for i, (piece_type, color) in enumerate(zip(piece_types, colors)):
                for square in board.pieces(piece_type, color):
                    rank, file = divmod(square, 8)
                    state[0, i, 7 - rank, file] = piece_values[piece_type]  # Use piece_values to set the identifier

            # Add squares that are under attack or dominated
            for color in [chess.WHITE, chess.BLACK]:
                for square in chess.SQUARES:
                    if board.is_attacked_by(color, square):
                        rank, file = divmod(square, 8)
                        state[0, 13, 7 - rank, file] += 1  # Increment if square is attacked by multiple pieces

            return state


        

        def is_promotion(self, move):
            # Assuming 'move' is a string like 'e7e8'
            move = move.uci()
            start, end = move[:2], move[2:]
            if start[1] == '7' and end[1] == '8':
                return True
            return False


        def get_reward(self, board, color, move, original_piece_type, selfplay=False, move_num=0):
            reward = 0
            mobility = len(list(board.legal_moves))
            reward += 1 * mobility

            # Define piece values
            piece_values = {
                chess.PAWN: 10,
                chess.KNIGHT: 30,
                chess.BISHOP: 30,
                chess.ROOK: 50,
                chess.QUEEN: 90,
                chess.KING: 1000
            }


            # Reward based on material
            for piece, value in piece_values.items():
                reward += len(board.pieces(piece, color)) * value
                reward -= len(board.pieces(piece, not color)) * value

            # Additional rewards for controlling the center
            center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
            for square in center_squares:
                piece = board.piece_at(square)
                if piece is not None and piece.color == color:
                    reward += 100

            # Reward for rooks on the seventh rank
            for rook_square in board.pieces(chess.ROOK, color):
                if (color == chess.WHITE and chess.square_rank(rook_square) == 6) or (color == chess.BLACK and chess.square_rank(rook_square) == 1):
                    reward += 10
            if self.is_promotion(move):
                reward += 50
            # Reward for controlling open files
            for file in range(8):
                pawns_on_file = board.pieces(chess.PAWN, color) & chess.BB_FILES[file]
                if not pawns_on_file:
                    rooks_on_file = board.pieces(chess.ROOK, color) & chess.BB_FILES[file]
                    if rooks_on_file:
                        reward += 1

            # Reward for connected pawns
            for pawn_square in board.pieces(chess.PAWN, color):
                pawn_file = chess.square_file(pawn_square)
                pawn_rank = chess.square_rank(pawn_square)
                if pawn_file > 0 and board.piece_type_at(chess.square(pawn_rank, pawn_file - 1)) == chess.PAWN:
                    reward += 1
                if pawn_file < 7 and board.piece_type_at(chess.square(pawn_rank, pawn_file + 1)) == chess.PAWN:
                    reward += 1

            # Reward for castling
            if board.has_castling_rights(color):
                reward += 100

            # Penalize for king safety
            king_square = board.king(color)
            if king_square is not None:
                for square in chess.SQUARES:
                    if board.is_attacked_by(not color, square):
                        reward -= 100
                    if board.is_attacked_by(color, square):
                        reward += 100

            # Check for terminal states
            if board.is_checkmate():
                reward += 1000 if board.turn != color else -1000
            elif board.is_check():
                reward += 100 if board.turn != color else -100

            elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition() or board.is_variant_draw():
                reward += -10000

            if len(board.move_stack) >= 2 and move == board.peek():
                reward -= 600
            
            if selfplay:
                reward -= move_num
            
            reward = reward * 0.1
            return reward


        def get_opponent_move(self, board, counter):

            # Get the latest game state from streamed_game
            try:

                opponent_color = chess.BLACK if self.color == chess.WHITE else chess.WHITE
                opponent_move = chess.Move.from_uci(self.opponent_move)
                board.turn = opponent_color

                print(f"DEBUG: RESULT OF CONVERTION: {opponent_move}")
                
                if self.opponent_move is None:
                    pass
                
                elif self.is_draw == True:

                     done = True
                     state = self.board_to_state(board)
                     reward = -2000
                     self.clearscreen()
                     self.print_board(board)
                     print("Draw!")
                     self.draws += 1
                     next_state = self.board_to_state(board)
                     self.update_model(state, opponent_move, reward)
                     self.remember(state, opponent_move, reward, next_state, done)

                elif self.repeat_count > 20:
                    if self.backupfen != self.lastfen:
                        board.set_fen(self.backupfen)
                        self.lastfen = self.backupfen
                        board.turn = self.color
                        self.clearscreen()
                        self.print_board(board)
                        if len(self.memory) >= batch_size:
                            print("WARNING: No more memory, training stage 2 is suspended until the end of the game")
                        else:
                            print("Amount of wins: ", str(self.wins))
                            print("Amount of draws: ", str(self.draws))
                            print("Amount of losses: ", str(self.losses))
                            print("DEBUG: Memory size (long_term): ", len(self.memory))
                            print("DEBUG: Memory left: ", batch_size - len(self.memory))
                            print("DEBUG: Batch_Size: ", str(self.batch_size))
                            print("DEBUG: Color: ", self.color, " (", self.my_color, ")")
                            print(f"DEBUG: Epsilon: {self.epsilon}")
                        print(f"DEBUG: Loaded fen: {self.backupfen}")
                        self.repeat_count = 0
                    else:
                        pass

                else:
                   
                    # Convert UCI string to chess.Move
                    opponent_move = chess.Move.from_uci(self.opponent_move)
                    print(f"DEBUG: RESULT OF CONVERTION: {opponent_move}")

                    
                    opponent_color = chess.BLACK if self.color == chess.WHITE else chess.WHITE
                    board.turn = opponent_color

                    if opponent_move in board.legal_moves:

                         done = board.is_game_over()
                         state = self.board_to_state(board)
                         original_piece_type = board.piece_at(opponent_move.from_square).piece_type if board.piece_at(opponent_move.from_square) else None
                         board.push(opponent_move)
                         reward = self.get_reward(board, opponent_color, opponent_move, original_piece_type)
                         self.clearscreen()
                         self.print_board(board)
                         if len(self.memory) >= batch_size:
                             print("WARNING: No more memory, training stage 2 is suspended until the end of the game")
                         else:
                             print("Amount of wins: ", str(self.wins))
                             print("Amount of draws: ", str(self.draws))
                             print("Amount of losses: ", str(self.losses))
                             print("DEBUG: Memory size (long_term): ", len(self.memory))
                             print("DEBUG: Memory left: ", batch_size - len(self.memory))
                             print("DEBUG: Batch_Size: ", str(self.batch_size))
                             print("DEBUG: Color: ", self.color, " (", self.my_color, ")")
                             print(f"DEBUG: Epsilon: {self.epsilon}") 
 
                         next_state = self.board_to_state(board)
                         self.model_learn(state, opponent_move, reward)
                         self.remember(state, opponent_move, reward, next_state, done)
                         board.turn = self.color
                         
                    elif self.is_promotion(opponent_move):
                        board.set(self.backup_fen)
                        self.clearscreen()
                        self.print_board(board)
                        if len(self.memory) >= batch_size:
                            print("WARNING: No more memory, training stage 2 is suspended until the end of the game")
                        else:
                            print("Amount of wins: ", str(self.wins))
                            print("Amount of draws: ", str(self.draws))
                            print("Amount of losses: ", str(self.losses))
                            print("DEBUG: Memory size (long_term): ", len(self.memory))
                            print("DEBUG: Memory left: ", batch_size - len(self.memory))
                            print("DEBUG: Batch_Size: ", str(self.batch_size))
                            print("DEBUG: Color: ", self.color, " (", self.my_color, ")")
                            print(f"DEBUG: Epsilon: {self.epsilon}") 

                    
                    else:
                        time.sleep(2.5)
                            
     
            except Exception:
                if self.lastfen == self.backupfen:
                    pass
            
                else:
                    board.set_fen(self.backupfen)
                    self.lastfen = self.backupfen
                    board.turn = self.color
                    self.clearscreen()
                    self.print_board(board)
                    if len(self.memory) >= batch_size:
                        print("WARNING: No more memory, training stage 2 is suspended until the end of the game")
                    else:
                        print("this is acting up:")

                        traceback.print_exc()
                        print("Amount of wins: ", str(self.wins))
                        print("Amount of draws: ", str(self.draws))
                        print("Amount of losses: ", str(self.losses))
                        print("DEBUG: Memory size (long_term): ", len(self.memory))
                        print("DEBUG: Memory left: ", batch_size - len(self.memory))
                        print("DEBUG: Batch_Size: ", str(self.batch_size))
                        print("DEBUG: Color: ", self.color, " (", self.my_color, ")")
                        print(f"DEBUG: Epsilon: {self.epsilon}")
                    print(f"DEBUG: Loaded fen: {self.backupfen}")
            



except Exception as e:
    traceback.print_exc()




        
def random_move(board, counter):
    counter = 0
    action = random.choice(list(board.legal_moves))
    board.push(action)

if __name__ == "__main__":
    try:
        board = chess.Board()
        board.turn = chess.WHITE
        batch_size = 270
        agent = DQNAgent()
        episodes = 1000
        agent.train(999999999999999999999, batch_size, board)
    except Exception:
        traceback.print_exc()

