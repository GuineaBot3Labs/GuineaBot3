import chess
import multiprocessing

# Define a simple evaluation function that counts material advantage.
def evaluate_board(board):
    piece_values = {
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 20,
        'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': -20,
    }

    evaluation = 0

    # Determine the AI's color based on the board's turn.
    ai_color = board.turn

    # Check if the game is in checkmate.
    if board.is_checkmate():
        evaluation += 1000  # Reward for checkmate.
  
    # Check for threefold repetition and penalize it.
    if board.can_claim_threefold_repetition():
        evaluation -= 1000  # Penalty for threefold repetition.

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_value = piece_values.get(piece.symbol(), 0)
            # Adjust the piece value based on the piece's color.
            if piece.color == ai_color:
                evaluation += piece_value
            else:
                evaluation -= piece_value

    return evaluation


# Minimax with alpha-beta pruning function.
def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# Function to evaluate a move using minimax and return the evaluation.
def evaluate_move(board, move, depth):
    board.push(move)
    eval = minimax(board, depth - 1, float('-inf'), float('inf'), False)
    board.pop()
    return eval

# Function to choose the best move using multiprocessing and minimax.
def choose_move(board, depth):
    best_move = None
    best_eval = float('-inf')
    alpha = float('-inf')
    beta = float('inf')

    # Create a multiprocessing pool with the desired number of processes.
    num_processes = 3 # Use all available CPU cores
    print(f"DEBUG: amount of cores used: {str(num_processes)}")
    pool = multiprocessing.Pool(processes=num_processes)

    # Create a list of move-evaluation tuples to store results.
    move_evaluations = []

    for move in board.legal_moves:
        move_evaluations.append((move, pool.apply_async(evaluate_move, (board.copy(), move, depth))))

    # Wait for the evaluations to complete and collect the results.
    for move, evaluation in move_evaluations:
        eval = evaluation.get()  # Get the result from the asynchronous evaluation
        if eval > best_eval:
            best_eval = eval
            best_move = move
        alpha = max(alpha, eval)

    # Close the multiprocessing pool.
    pool.close()
    pool.join()

    return best_move

    return best_move
