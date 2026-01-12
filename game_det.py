from dataclasses import dataclass
from typing import List
import numpy as np
from obj_det import ArucoDetector, BallDetector
from board_est import BoardEstimator

@dataclass
class GameState:
    """Represents the current state of all game objects."""
    balls: List = None      # List of BallState objects
    players: List = None    # List of PlayerState objects
    board_transform: np.ndarray = None  # 4x4 transformation matrix from board to camera
    pnp_result: tuple = None  # PnP solution (rvec, tvec, etc.)
    detector: 'GameDetector' = None  # Reference to parent GameDetector
    timestamp: float = None  # Detection timestamp
    
    def __post_init__(self):
        if self.balls is None:
            self.balls = []
        if self.players is None:
            self.players = []

@dataclass
class BallState:
    """Represents a single ball's state."""
    x: float
    y: float
    detection: 'DetectedObject' = None  # Original detection object

@dataclass
class PlayerState:
    """Represents a single player's state."""
    id: int
    x: float
    y: float
    angle: float
    detection: 'DetectedAruco' = None  # Original detection object

class GameDetector:
    """Detects game objects and returns GameState."""
    
    def __init__(self, board_estimator, ball_detector=None, ball_height=0.0, 
                 aruco_detector=None, player_height=0.0):
        """Initialize game detector.
        
        Args:
            board_estimator: BoardEstimator instance for detecting board pose
            ball_detector: Optional ObjectDetector for ball detection
            ball_height: Ball height above board (m) for parallax correction
            aruco_detector: Optional ObjectDetector for player ArUco marker detection
            player_height: Player height above board (m) for parallax correction
        """
        self.board_estimator = board_estimator

        self.ball_detector = ball_detector or BallDetector()
        self.ball_height = ball_height

        self.player_detector = aruco_detector
        self.player_height = player_height

    def _localize(self, frame, centroid, pnp_result, height):
        """Detect object and return 3D board coordinates with parallax correction."""
        
        if centroid is None:
            return None
        
        x, y = self.board_estimator.project_point_to_board(
            pnp_result, centroid, frame.shape, z=height
        )
        
        return (x, y, height)
    
    def detect(self, frame, drawing_frame=None):
        """Detect all game objects and return GameState.
        
        Args:
            frame: Input image frame
            drawing_frame: Optional frame to draw detections on
            
        Returns:
            GameState with detected balls and players
        """
        import time
        
        # Detect board first
        result = self.board_estimator.get_board_transform(frame)
        
        if result is None:
            return GameState()  # Return empty state if board not detected
        
        board_T, pnp_result = result
        timestamp = time.time()
        
        balls = []
        players = []
        
        # Detect balls
        if self.ball_detector is not None:
            ball_detections = self.ball_detector.detect(frame)
            for ball in ball_detections:
                xyz = self._localize(frame, ball.centroid, pnp_result, self.ball_height)
                if xyz is not None:
                    balls.append(BallState(
                        x=xyz[0],
                        y=xyz[1],
                        detection=ball
                    ))
                    
                    # Draw ball contour
                    if drawing_frame is not None and ball.contour is not None:
                        cv2.drawContours(drawing_frame, [ball.contour], -1, (0, 255, 255), 2)
        
        # Detect players
        if self.player_detector is not None:
            player_detections = self.player_detector.detect(frame)
            for player in player_detections:
                # Skip board markers
                if (self.board_estimator.config.board_marker_ids is not None and 
                    player.id in self.board_estimator.config.board_marker_ids):
                    continue
                
                xyz = self._localize(frame, player.centroid, pnp_result, self.player_height)
                if xyz is not None and player.angle is not None:
                    # Transform angle from image space to board space
                    # Get camera rotation in board frame
                    cam_T_in_board = np.linalg.inv(board_T)
                    cam_R = cam_T_in_board[:3, :3]
                    
                    # Extract Z rotation (gamma) from camera orientation
                    from matrix_help import extract_euler_zyx
                    alpha, beta, gamma = extract_euler_zyx(cam_R)
                    
                    # Transform angle
                    angle_board = (player.angle - gamma + np.pi) % (2 * np.pi)
                              
                    players.append(PlayerState(
                        id=player.id,
                        x=xyz[0],
                        y=xyz[1],
                        angle=angle_board,
                        detection=player
                    ))
                    
                    # Draw player triangle using image-space angle
                    if drawing_frame is not None and player.centroid is not None:
                        cx, cy = int(player.centroid[0]), int(player.centroid[1])
                        
                        # Triangle size
                        length = 20
                        width = 12
                        
                        # Calculate triangle points (isosceles pointing in direction of angle)
                        # Tip of triangle (flipped from marker angle)
                        tip_x = cx + length * np.cos(-player.angle)
                        tip_y = cy + length * np.sin(-player.angle)
                        
                        # Base corners (perpendicular to angle)
                        base_angle = -player.angle + np.pi / 2
                        base1_x = cx + (width / 2) * np.cos(base_angle)
                        base1_y = cy + (width / 2) * np.sin(base_angle)
                        base2_x = cx - (width / 2) * np.cos(base_angle)
                        base2_y = cy - (width / 2) * np.sin(base_angle)
                        
                        # Draw filled triangle
                        pts = np.array([[tip_x, tip_y], [base1_x, base1_y], [base2_x, base2_y]], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.fillPoly(drawing_frame, [pts], (255, 100, 255))
        
        return GameState(
            balls=balls,
            players=players,
            board_transform=board_T,
            pnp_result=pnp_result,
            detector=self,
            timestamp=timestamp
        )

if __name__ == "__main__":
    import cv2
    from board_config import board_config_letter
    from cam_config import global_cam
    
    # Setup GameDetector
    game_detector = GameDetector(
        board_estimator=BoardEstimator(board_config_letter, K=global_cam.K, D=global_cam.D, rotate_180=True),
        ball_detector=BallDetector(),
        ball_height=0.02,
        aruco_detector=ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)),
        player_height=0.04
    )
    
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break
        
        frame = global_cam.get_frame()
        if frame is None:
            continue
        
        drawing_frame = frame.copy()
        
        # Detect game state
        game_state = game_detector.detect(frame, drawing_frame)
        
        # Annotate balls
        for i, ball in enumerate(game_state.balls):
            cv2.putText(drawing_frame, f"Ball: ({ball.x:.3f}, {ball.y:.3f})m",
                       (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Annotate players
        for i, player in enumerate(game_state.players):
            angle_deg = np.degrees(player.angle)
            cv2.putText(drawing_frame, f"Player {player.id}: ({player.x:.3f}, {player.y:.3f})m, {angle_deg:.1f}deg",
                       (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)
        
        # Display
        cv2.imshow("Game Detection", drawing_frame)
        cv2.setWindowProperty("Game Detection", cv2.WND_PROP_TOPMOST, 1)
