import chess
import chess.svg
import chess.engine
import pygame
import cairosvg
import io

class ChessEnvironment:
    def __init__(self, screen_size=600, render=False):
        self.board = chess.Board()
        self.screen_size = screen_size
        self.render = render
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_size, screen_size))
            pygame.display.set_caption("Chess Game")
        self.engine = chess.engine.SimpleEngine.popen_uci("/home/priyanshu/Downloads/stockfish/stockfish-ubuntu-x86-64-avx2")
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        self.previous_board_value = self.evaluate_board()
        self.last_st = 0


    def evaluate_board(self):
        """Evaluate the current board state."""
        value = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    value += piece_value
                else:
                    value -= piece_value
        return value
    def get_piece_square_tables(self):
        """Define piece-square tables for positional evaluation."""
        # These tables encourage pieces to move to strategically advantageous positions
        # Values are for white pieces. For black pieces, the table should be reversed.
        pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50,
        ]
        bishop_table = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20,
        ]
        rook_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ]
        queen_table = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]
        king_table = [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ]
        return {
            chess.PAWN: pawn_table,
            chess.KNIGHT: knight_table,
            chess.BISHOP: bishop_table,
            chess.ROOK: rook_table,
            chess.QUEEN: queen_table,
            chess.KING: king_table
        }
    
    def evaluate_position(self):
        """Evaluate the positional strength of the current board state."""
        piece_square_tables = self.get_piece_square_tables()
        position_value = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                if piece.color == chess.WHITE:
                    position_value += piece_square_tables[piece.piece_type][square]
                else:
                    position_value -= piece_square_tables[piece.piece_type][63 - square]
        return position_value / 100
    

    def render_board(self):
        if not self.render:
            return
        svg = chess.svg.board(self.board, size=self.screen_size)
        png_io = io.BytesIO()
        cairosvg.svg2png(bytestring=svg, write_to=png_io)
        png_io.seek(0)
        surface = pygame.image.load(png_io, 'PNG')
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    def reset(self):
        self.board.reset()
        self.last_st = 0
        return self.get_state()

    def get_state(self):
        """Returns the current state of the board as a flattened string."""
        return self.get_model_input()

    def get_model_input(self):
        """
        Returns a string representation of the board state suitable for a model.

        Format: Each piece is represented by its color ('w' or 'b'), type (e.g., 'P', 'N', 'R'),
        and a unique identifier (1-indexed based on appearance). Empty squares are '0000'.
        The last two elements represent the last moved and last eliminated pieces.
        """
        def get_piece_notation(piece, square):
            color = 'w' if piece.color == chess.WHITE else 'b'
            piece_type = piece.symbol().upper()
            count = sum(1 for s, p in piece_counts.items() if p == f"{color}{piece_type}")
            piece_counts[square] = f"{color}{piece_type}"
            return f"{color}{piece_type}{count + 1}"

        piece_counts = {}
        board_representation = []

        for i in range(8):
            for j in range(8):
                square = chess.square(j, 7 - i)
                piece = self.board.piece_at(square)
                if piece:
                    notation = get_piece_notation(piece, square)
                    board_representation.append(notation.ljust(4))
                else:
                    board_representation.append('0000')

        # Get the last move made
        last_moved_piece = '0000'
        last_eliminated_piece = '0000'
        if len(self.board.move_stack) > 0:
            last_move = self.board.move_stack[-1]

            # Last moved piece
            moved_piece = self.board.piece_at(last_move.to_square)
            if moved_piece:
                last_moved_piece = get_piece_notation(moved_piece, last_move.to_square)

            # Last eliminated piece
            if self.board.is_capture(last_move):
                if self.board.is_en_passant(last_move):
                    captured_square = chess.Square(last_move.to_square ^ 8)
                else:
                    captured_square = last_move.from_square
                captured_piece = self.board.piece_at(captured_square)
                if captured_piece:
                    last_eliminated_piece = get_piece_notation(
                        captured_piece, captured_square
                    )

        board_representation.append(last_moved_piece)
        board_representation.append(last_eliminated_piece)

        return ' '.join(board_representation)

    def custom_to_standard_move(self, custom_move):
        """Converts a custom move (e.g., 'wP1_2U') to standard algebraic notation."""
        def find_piece_position(piece):
            flattened_map = self.get_model_input().split()
            for i, square in enumerate(flattened_map):
                if square.strip() == piece:
                    return i % 8, 7 - (i // 8)
            return None

        def calculate_new_position(x, y, move_parts):
            for part in move_parts:
                direction = part[1]
                distance = int(part[0])
                if direction == 'U':
                    y += distance
                elif direction == 'D':
                    y -= distance
                elif direction == 'L':
                    x -= distance
                elif direction == 'R':
                    x += distance
            return x, y

        def coordinates_to_algebraic(x, y):
            return f"{chr(97 + x)}{y + 1}"

        piece, move = custom_move.split('_', 1)
        move_parts = move.split('_')
        start_pos = find_piece_position(piece)
        if start_pos is None:
            return "fdsf"

        new_x, new_y = calculate_new_position(
            start_pos[0], start_pos[1], move_parts
        )
        if not (0 <= new_x < 8 and 0 <= new_y < 8):
            return "fdsf"

        start_square = coordinates_to_algebraic(start_pos[0], start_pos[1])
        end_square = coordinates_to_algebraic(new_x, new_y)
        return f"{start_square}{end_square}"
    def calculate_reward(self, move):
        """Calculate the reward for a given move."""
        reward = 0

        # Material balance change
        new_board_value = self.evaluate_board()
        material_change = new_board_value - self.previous_board_value
        reward += material_change * 10  # Amplify the material change

        # Positional strength
        positional_value = self.evaluate_position()
        reward += positional_value

        # Check and checkmate
        if self.board.is_check():
            reward += 1
        if self.board.is_checkmate():
            reward += 100 if self.board.turn == chess.BLACK else -100

        # Control of the center
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        for square in center_squares:
            piece = self.board.piece_at(square)
            if piece:
                reward += 0.5 if piece.color == chess.WHITE else -0.5

        # Pawn structure
        white_pawns = self.board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = self.board.pieces(chess.PAWN, chess.BLACK)
        reward += (len(white_pawns) - len(black_pawns)) * 0.1

        # Piece development (for the opening)
        if len(self.board.move_stack) < 20:
            reward += self.evaluate_development()

        # King safety
        reward += self.evaluate_king_safety()

        # Mobility (number of legal moves)
        self.board.push(move)
        opponent_mobility = len(list(self.board.legal_moves))
        self.board.pop()
        reward -= opponent_mobility * 0.01  # Slight penalty for increasing opponent's mobility

        self.previous_board_value = new_board_value
        return reward
    
    def evaluate_development(self):
        """Evaluate piece development in the opening."""
        development_score = 0
        back_rank = chess.SquareSet((chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1, chess.H1))
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for piece in self.board.pieces(piece_type, chess.WHITE):
                if piece not in back_rank:
                    development_score += 0.5
            for piece in self.board.pieces(piece_type, chess.BLACK):
                if piece not in chess.SquareSet(chess.A8 | chess.B8 | chess.C8 | chess.D8 | chess.E8 | chess.F8 | chess.G8 | chess.H8):
                    development_score -= 0.5
        return development_score
    
    def evaluate_king_safety(self):
        """Evaluate the safety of both kings."""
        def king_safety_score(color):
            king_square = self.board.king(color)
            if king_square is None:
                return 0
            
            pawns_near_king = 0
            for square in chess.SquareSet(chess.BB_KING_ATTACKS[king_square]):
                if self.board.piece_at(square) == chess.Piece(chess.PAWN, color):
                    pawns_near_king += 1
            
            castling_rights = 2 if self.board.has_castling_rights(color) else 0
            
            return pawns_near_king + castling_rights

        white_safety = king_safety_score(chess.WHITE)
        black_safety = king_safety_score(chess.BLACK)
        return (white_safety - black_safety) * 0.1


    def step(self, custom_move):
        standard_move = self.custom_to_standard_move(custom_move)
        info = {"message": ""}
        reward = 0
        done = False
        if standard_move == "fdsf":
            info["message"] = "Invalid move or piece not found!"
            reward = -50
            done = True
        else:
            try:
                move = chess.Move.from_uci(standard_move)
                if move in self.board.legal_moves:
                    reward = self.calculate_reward(move)
                    self.board.push(move)
                    info["message"] = f"Move successful: {move}"
                    # Bot's turn (if game is not over)
                    if not self.board.is_game_over():
                        result = self.engine.play(self.board, chess.engine.Limit(time=2.0))
                        bot_reward = self.calculate_reward(result.move)
                        self.board.push(result.move)
                        info["message"] += f" Bot moved: {result.move}"
                        reward -= bot_reward  # Subtract bot's reward from player's reward
                    done = self.board.is_game_over()

                else:
                    info["message"] = "Illegal move!"
                    reward = -20
                    if self.last_st == 2:
                        done = True
                
                self.last_st +=1
            except ValueError:
                info["message"] = "Invalid move format!"
                reward = -30
                done = True

        return self.get_state(), reward ,done , info


    def get_piece_value(self, piece):
        """Returns the value of a chess piece."""
        if piece.piece_type == chess.PAWN:
            return 1
        elif piece.piece_type == chess.KNIGHT:
            return 3
        elif piece.piece_type == chess.BISHOP:
            return 3
        elif piece.piece_type == chess.ROOK:
            return 5
        elif piece.piece_type == chess.QUEEN:
            return 9
        else:  # King
            return 0  # We don't want to encourage capturing the king directly

    def print_position_map(self):
        """Prints a visual representation of the board."""
        print("\nPosition Map:")
        print("  a b c d e f g h")
        for i in range(8):
            row = []
            for j in range(8):
                piece = self.board.piece_at(chess.square(j, 7 - i))
                row.append(piece.symbol() if piece else '.')
            print(f"{8 - i} {' '.join(row)} {8 - i}")
        print("  a b c d e f g h\n")




    # # Example usage:
    # env = ChessEnvironment(render=False)
    # state = env.reset()
    # env.print_position_map()

    # while True:
    #     custom_move = input("Enter your move (e.g., wP1_2U): ")
    #     new_state, reward, done, info = env.step(custom_move)
    #     print("Reward:", reward)
    #     print(info["message"])
    #     print("Flattened map:", new_state)
    #     env.print_position_map()

    #     if done:
    #         print("Game Over")
    #         print(f"Result: {env.board.result()}")

    #         if env.board.result() == "1-0":
    #             reward = 1000
    #         elif env.board.result() == "0-1":
    #             reward = -1000
    #         else:
    #             reward = 0
    #         break