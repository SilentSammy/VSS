import cv2
import numpy as np
from board_est import BoardEstimator
from board_config import board_config
from plotter3d import BoardPlotter3D
from cam_config import global_cam

if __name__ == "__main__":
    be = BoardEstimator(
        board_config=board_config,
        K=global_cam.K,
        D=global_cam.D
    )
    
    plotter = BoardPlotter3D(board_config, axis_limit=0.5, camera_at_origin=True)
    
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
        # Get frame
        frame = global_cam.get_frame()
        drawing_frame = frame.copy()
    
        # Estimate
        res = be.get_board_transform(frame, drawing_frame=drawing_frame)
    
        if res is not None:
            board_T, _ = res
            plotter.update(board_T)
    
        # Display
        cv2.imshow("Vision Sensor", drawing_frame)
        cv2.setWindowProperty("Vision Sensor", cv2.WND_PROP_TOPMOST, 1)
