import chess.engine
import gc

# NOTE: you may need to install stockfish if you want to use this code for bootstrapping.
# STOCKFISH_PATH = "./Stockfish/src/stockfish" # For Linux
# STOCKFISH_PATH = "./Stockfish/src/stockfish.exe" # For Windows

### NOTE: I am not taking credit for stockfish's agent, this is to only fetch a move from it for boot strapping, all credit to stockfish developters ###
def getstockfishmove(board):
    action = None  # Initialize action
    engine = None  # Initialize engine
    try:
        print("DEBUG: Initing engine")
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        print("DEBUG: Inited engine")
        result = engine.play(board, chess.engine.Limit(time=0.1))
        print("DEBUG: Making move")
        action = result.move
    finally:
        print("DEBUG: Quiting engine")
        engine.quit()

        del engine
        gc.collect()
    
    return action
