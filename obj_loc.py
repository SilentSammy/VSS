class ObjectLocalizer:
    """Locates detected objects in 3D board coordinates."""
    
    def __init__(self, detector, board_estimator, height=0.0):
        """Initialize localizer with detector, board estimator, and object height."""
        self.detector = detector
        self.board_estimator = board_estimator
        self.height = height
    
    def localize(self, frame, pnp_result, drawing_frame=None):
        """Detect object and return board coordinates (x, y) or None."""
        centroid, contour = self.detector.detect(frame, drawing_frame)
        
        if centroid is None:
            return None
        
        return self.board_estimator.project_point_to_board(
            pnp_result, centroid, frame.shape, z=self.height
        )
