import cv2
import numpy as np
import json
import os

# ========== CONFIGURATION ==========
CONFIG_FILE = "roi_config.json"

class ROIConfigurator:
    def __init__(self, video_path, mode="LINE"):
        self.video_path = video_path
        self.frame = None
        self.display_frame = None
        
        # Frame dimensions
        self.frame_h = None
        self.frame_w = None
        self.display_h = 1080
        self.display_w = 1920
        self.scale_x = 1.0
        self.scale_y = 1.0
        
        # Configuration mode: "LINE" (L1/L2/Detection) or "RECTANGLE"
        self.config_mode = mode
        
        # Drawing state - LINE mode
        self.l1_points = []  # L1 line: [start, end]
        self.l2_points = []  # L2 line: [start, end]
        self.detection_line = []  # Detection line: [start, end]
        self.aoi_polygon = []  # Filled polygon between L1 and L2
        
        # Drawing state - RECTANGLE mode
        self.rectangle_points = []  # Rectangle: [top-left, bottom-right]
        self.rectangle_roi = None  # Final rectangle ROI
        
        self.current_mode = "L1" if self.config_mode == "LINE" else "RECTANGLE"
        self.drawing = False
        self.current_point = None
        
        # Load first frame from video
        self.load_first_frame()
        
    def load_first_frame(self):
        """Load the first frame from video"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {self.video_path}")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            self.frame = frame
            self.frame_h, self.frame_w = frame.shape[:2]
            
            # Calculate scale factors
            self.scale_x = self.frame_w / self.display_w
            self.scale_y = self.frame_h / self.display_h
            
            print(f"✓ Loaded video frame: {frame.shape}")
            print(f"✓ Actual resolution: {self.frame_w} x {self.frame_h}")
            print(f"✓ Scale factors: X={self.scale_x:.2f}, Y={self.scale_y:.2f}")
            
            self.display_frame = cv2.resize(frame, (self.display_w, self.display_h))
            return True
        return False
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing"""
        # Convert display coordinates to actual frame coordinates
        actual_x = int(x * self.scale_x)
        actual_y = int(y * self.scale_y)
        
        # Update cursor position for real-time visualization (keep display coords for drawing)
        if event == cv2.EVENT_MOUSEMOVE:
            self.current_point = (x, y)  # Display coords for drawing
            self.redraw_frame()
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_point = (x, y)
            
            if self.config_mode == "LINE":
                # ===== LINE MODE (L1/L2/Detection lines) =====
                if self.current_mode == "L1":
                    self.l1_points.append((actual_x, actual_y))  # Save actual coords
                    print(f"Point {len(self.l1_points)}: Display=({x},{y}) -> Actual=({actual_x},{actual_y})")
                    if len(self.l1_points) == 2:
                        print(f"OK L1 line saved: {self.l1_points}")
                        self.current_mode = "L2"
                        print(f"\nNow draw L2 line (2 clicks)")
                        self.draw_ui_text()
                        
                elif self.current_mode == "L2":
                    self.l2_points.append((actual_x, actual_y))  # Save actual coords
                    print(f"Point {len(self.l2_points)}: Display=({x},{y}) -> Actual=({actual_x},{actual_y})")
                    if len(self.l2_points) == 2:
                        print(f"OK L2 line saved: {self.l2_points}")
                        self.calculate_aoi()
                        self.current_mode = "DETECTION"
                        print(f"\nNow draw DETECTION line (2 clicks)")
                        self.draw_ui_text()
                        
                elif self.current_mode == "DETECTION":
                    self.detection_line.append((actual_x, actual_y))  # Save actual coords
                    print(f"Point {len(self.detection_line)}: Display=({x},{y}) -> Actual=({actual_x},{actual_y})")
                    if len(self.detection_line) == 2:
                        print(f"OK Detection line saved: {self.detection_line}")
                        self.current_mode = "REVIEW"
                        print(f"\nConfiguration complete! Press 's' to save or 'r' to reset")
                        self.draw_ui_text()
            
            elif self.config_mode == "RECTANGLE":
                # ===== RECTANGLE MODE =====
                self.rectangle_points.append((actual_x, actual_y))  # Save actual coords
                print(f"Rectangle Point {len(self.rectangle_points)}: Display=({x},{y}) -> Actual=({actual_x},{actual_y})")
                if len(self.rectangle_points) == 2:
                    print(f"OK Rectangle saved: {self.rectangle_points}")
                    self.calculate_rectangle_roi()
                    self.current_mode = "REVIEW"
                    print(f"\nRectangle ROI defined! Press 's' to save or 'r' to reset")
                    self.draw_ui_text()
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_point = (x, y)
            self.redraw_frame()
    
    def calculate_aoi(self):
        """Calculate the AOI polygon between L1 and L2"""
        if len(self.l1_points) == 2 and len(self.l2_points) == 2:
            # Create polygon in counter-clockwise order
            self.aoi_polygon = np.array([
                self.l1_points[0],  # L1 start
                self.l1_points[1],  # L1 end
                self.l2_points[1],  # L2 end
                self.l2_points[0]   # L2 start
            ], dtype=np.int32)
            print(f"✓ AOI polygon calculated with {len(self.aoi_polygon)} points")
    
    def calculate_rectangle_roi(self):
        """Calculate rectangle ROI from two corner points"""
        if len(self.rectangle_points) == 2:
            p1 = self.rectangle_points[0]
            p2 = self.rectangle_points[1]
            
            # Ensure top-left and bottom-right order
            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
            
            # Create rectangle ROI as polygon (4 corners in order)
            self.rectangle_roi = np.array([
                (x1, y1),  # Top-left
                (x2, y1),  # Top-right
                (x2, y2),  # Bottom-right
                (x1, y2)   # Bottom-left
            ], dtype=np.int32)
            print(f"OK Rectangle ROI: TL({x1},{y1}) -> BR({x2},{y2})")
    
    def redraw_frame(self):
        """Redraw the frame with current drawings"""
        # Start with resized display frame
        self.display_frame = cv2.resize(self.frame, (self.display_w, self.display_h))
        
        # Helper function to convert actual coords to display coords
        def to_display(point):
            return (int(point[0] / self.scale_x), int(point[1] / self.scale_y))
        
        if self.config_mode == "LINE":
            # ===== DRAW LINE MODE ELEMENTS =====
            # Draw L1 line
            if len(self.l1_points) > 0:
                p0_disp = to_display(self.l1_points[0])
                cv2.circle(self.display_frame, p0_disp, 8, (255, 0, 0), -1)
                cv2.putText(self.display_frame, "L1_START", p0_disp, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                if len(self.l1_points) > 1:
                    p1_disp = to_display(self.l1_points[1])
                    cv2.line(self.display_frame, p0_disp, p1_disp, (255, 0, 0), 3)
                    cv2.circle(self.display_frame, p1_disp, 8, (255, 0, 0), -1)
                    cv2.putText(self.display_frame, "L1_END", p1_disp, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                elif self.current_mode == "L1" and self.current_point:
                    cv2.line(self.display_frame, p0_disp, self.current_point, (255, 100, 0), 2)
            
            # Draw L2 line
            if len(self.l2_points) > 0:
                p0_disp = to_display(self.l2_points[0])
                cv2.circle(self.display_frame, p0_disp, 8, (0, 255, 0), -1)
                cv2.putText(self.display_frame, "L2_START", p0_disp, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if len(self.l2_points) > 1:
                    p1_disp = to_display(self.l2_points[1])
                    cv2.line(self.display_frame, p0_disp, p1_disp, (0, 255, 0), 3)
                    cv2.circle(self.display_frame, p1_disp, 8, (0, 255, 0), -1)
                    cv2.putText(self.display_frame, "L2_END", p1_disp, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif self.current_mode == "L2" and self.current_point:
                    cv2.line(self.display_frame, p0_disp, self.current_point, (100, 200, 0), 2)
            
            # Draw AOI polygon
            if len(self.aoi_polygon) > 0:
                points_disp = np.array([to_display(p) for p in self.aoi_polygon], np.int32)
                cv2.polylines(self.display_frame, [points_disp], True, (0, 255, 255), 3)
                # Create transparent overlay
                overlay = self.display_frame.copy()
                cv2.fillPoly(overlay, [points_disp], (0, 100, 100))
                cv2.addWeighted(overlay, 0.3, self.display_frame, 0.7, 0, self.display_frame)
                aoi_label_pos = to_display(self.aoi_polygon[0])
                cv2.putText(self.display_frame, "AOI", aoi_label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Draw detection line
            if len(self.detection_line) > 0:
                p0_disp = to_display(self.detection_line[0])
                cv2.circle(self.display_frame, p0_disp, 8, (255, 0, 0), -1)
                cv2.putText(self.display_frame, "DET_START", p0_disp, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                if len(self.detection_line) > 1:
                    p1_disp = to_display(self.detection_line[1])
                    cv2.line(self.display_frame, p0_disp, p1_disp, (255, 0, 0), 3)
                    cv2.circle(self.display_frame, p1_disp, 8, (255, 0, 0), -1)
                    cv2.putText(self.display_frame, "DET_END", p1_disp, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                elif self.current_mode == "DetectionLine" and self.current_point:
                    cv2.line(self.display_frame, p0_disp, self.current_point, (200, 100, 0), 2)
        
        elif self.config_mode == "RECTANGLE":
            # ===== DRAW RECTANGLE MODE ELEMENTS =====
            # Draw rectangle during creation
            if len(self.rectangle_points) > 0:
                p0_disp = to_display(self.rectangle_points[0])
                cv2.circle(self.display_frame, p0_disp, 10, (255, 100, 100), -1)
                cv2.putText(self.display_frame, "TL", p0_disp, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
                
                # Draw preview line to current cursor during first point selection
                if len(self.rectangle_points) == 1 and self.current_point:
                    cv2.line(self.display_frame, p0_disp, self.current_point, (100, 150, 255), 2)
            
            # Draw final rectangle ROI
            if self.rectangle_roi is not None:
                rect_disp = np.array([to_display(p) for p in self.rectangle_roi], np.int32)
                cv2.polylines(self.display_frame, [rect_disp], True, (0, 255, 0), 3)
                cv2.fillPoly(self.display_frame, [rect_disp], (0, 150, 0), alpha=0.3)
                tl_disp = to_display(self.rectangle_roi[0])
                cv2.putText(self.display_frame, "ROI", tl_disp, 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw corner coordinates
                for i, pt in enumerate(self.rectangle_roi):
                    pt_disp = to_display(pt)
                    cv2.circle(self.display_frame, pt_disp, 6, (0, 255, 0), -1)
        
        # Draw precise cursor/crosshair at current mouse position
        if self.current_point:
            x, y = self.current_point
            
            # Draw crosshair lines
            size = 20
            thickness = 2
            color = (255, 255, 0)  # Cyan color
            
            # Horizontal line
            cv2.line(self.display_frame, (x - size, y), (x + size, y), color, thickness)
            # Vertical line
            cv2.line(self.display_frame, (x, y - size), (x, y + size), color, thickness)
            
            # Draw center dot
            cv2.circle(self.display_frame, (x, y), 5, color, -1)
            
            # Draw outer circle for better visibility
            cv2.circle(self.display_frame, (x, y), 12, color, 2)
            
            # Show coordinates near cursor
            coord_text = f"({x}, {y})"
            cv2.putText(self.display_frame, coord_text, (x + 20, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        self.draw_ui_text()
    
    def draw_ui_text(self):
        """Draw UI instructions on frame"""
        h, w = self.display_frame.shape[:2]
        
        # Mode indicator
        mode_text = f"MODE: {self.config_mode} - {self.current_mode}"
        color = (0, 255, 0) if self.current_mode != "REVIEW" else (0, 255, 255)
        cv2.putText(self.display_frame, mode_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Instructions based on mode
        if self.config_mode == "LINE":
            if self.current_mode == "L1":
                msg = "Click 2 points to draw L1 line"
                count = f"Points: {len(self.l1_points)}/2"
            elif self.current_mode == "L2":
                msg = "Click 2 points to draw L2 line"
                count = f"Points: {len(self.l2_points)}/2"
            elif self.current_mode == "DETECTION":
                msg = "Click 2 points to draw DETECTION line"
                count = f"Points: {len(self.detection_line)}/2"
            else:
                msg = "LINE Configuration complete!"
                count = "Press 's' to SAVE or 'r' to RESET"
        elif self.config_mode == "RECTANGLE":
            if self.current_mode == "RECTANGLE":
                msg = "Click 2 points: Top-Left, then Bottom-Right"
                count = f"Points: {len(self.rectangle_points)}/2"
            else:
                msg = "RECTANGLE Configuration complete!"
                count = "Press 's' to SAVE or 'r' to RESET"
        
        cv2.putText(self.display_frame, msg, (20, h - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(self.display_frame, count, (20, h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.putText(self.display_frame, "Press 'u' to UNDO | 'q' to QUIT", (20, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)
    
    def save_config(self):
        """Save configuration to JSON file"""
        if self.config_mode == "LINE":
            if len(self.l1_points) != 2 or len(self.l2_points) != 2 or len(self.detection_line) != 2:
                print("ERROR: Configuration incomplete! All lines must have 2 points each.")
                return False
            
            config = {
                "mode": "LINE",
                "l1_line": {
                    "start": list(self.l1_points[0]),
                    "end": list(self.l1_points[1])
                },
                "l2_line": {
                    "start": list(self.l2_points[0]),
                    "end": list(self.l2_points[1])
                },
                "detection_line": {
                    "start": list(self.detection_line[0]),
                    "end": list(self.detection_line[1])
                },
                "aoi_polygon": self.aoi_polygon.tolist() if len(self.aoi_polygon) > 0 else []
            }
            
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"OK Configuration saved to {CONFIG_FILE}")
            print(f"\nSaved LINE Configuration:")
            print(f"  L1:        {self.l1_points}")
            print(f"  L2:        {self.l2_points}")
            print(f"  Detection: {self.detection_line}")
            print(f"  AOI:       {len(self.aoi_polygon)} polygon points")
            return True
        
        elif self.config_mode == "RECTANGLE":
            if self.rectangle_roi is None:
                print("ERROR: Rectangle ROI not defined!")
                return False
            
            config = {
                "mode": "RECTANGLE",
                "rectangle_roi": self.rectangle_roi.tolist()
            }
            
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"OK Configuration saved to {CONFIG_FILE}")
            print(f"\nSaved RECTANGLE Configuration:")
            print(f"  ROI Corners: {self.rectangle_roi.tolist()}")
            return True
        
        return False
    
    def reset(self):
        """Reset configuration"""
        self.l1_points = []
        self.l2_points = []
        self.detection_line = []
        self.aoi_polygon = []
        self.rectangle_points = []
        self.rectangle_roi = None
        self.current_mode = "L1" if self.config_mode == "LINE" else "RECTANGLE"
        self.drawing = False
        self.current_point = None
        print(f"OK {self.config_mode} Configuration reset")
        self.redraw_frame()
    
    def undo(self):
        """Undo last point"""
        if self.config_mode == "LINE":
            if self.current_mode == "L1" and len(self.l1_points) > 0:
                self.l1_points.pop()
                print(f"OK Undid L1 point. Remaining: {len(self.l1_points)}")
            elif self.current_mode == "L2" and len(self.l2_points) > 0:
                self.l2_points.pop()
                if len(self.l2_points) == 0:
                    self.aoi_polygon = []
                print(f"OK Undid L2 point. Remaining: {len(self.l2_points)}")
            elif self.current_mode == "DETECTION" and len(self.detection_line) > 0:
                self.detection_line.pop()
                print(f"OK Undid Detection point. Remaining: {len(self.detection_line)}")
            elif self.current_mode == "REVIEW":
                print("INFO: Use 'r' to reset all points")
        
        elif self.config_mode == "RECTANGLE":
            if self.current_mode == "RECTANGLE" and len(self.rectangle_points) > 0:
                self.rectangle_points.pop()
                if len(self.rectangle_points) == 0:
                    self.rectangle_roi = None
                print(f"OK Undid Rectangle point. Remaining: {len(self.rectangle_points)}")
            elif self.current_mode == "REVIEW":
                print("INFO: Use 'r' to reset all points")
        
        self.redraw_frame()
    
    def run(self):
        """Run the configurator"""
        if self.frame is None:
            print("ERROR: Failed to load video frame")
            return
        
        print("\n" + "="*70)
        print("  MEHSANA ROI CONFIGURATOR - Interactive Drawing Tool")
        print("="*70)
        print(f"\nMode: {self.config_mode}")
        print("\nInstructions:")
        
        if self.config_mode == "LINE":
            print("  1. Draw L1 line (2 clicks)")
            print("  2. Draw L2 line (2 clicks)")
            print("  3. Draw DETECTION line (2 clicks)")
        elif self.config_mode == "RECTANGLE":
            print("  1. Click Top-Left corner")
            print("  2. Click Bottom-Right corner")
        
        print("  3. Press 's' to save configuration")
        print("\nControls:")
        print("  LEFT CLICK  : Add point")
        print("  'u'         : Undo last point")
        print("  'r'         : Reset all points")
        print("  's'         : Save configuration")
        print("  'q'         : Quit without saving")
        print("="*70 + "\n")
        
        window_name = "MEHSANA ROI Configurator"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.redraw_frame()
        
        while True:
            # Resize for display
            display = cv2.resize(self.display_frame, (1920, 1080))
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q'):
                print("✗ Quit without saving")
                break
            elif key == ord('s'):
                if self.save_config():
                    print("\n✓ Configuration saved successfully!")
                    break
            elif key == ord('u'):
                self.undo()
            elif key == ord('r'):
                self.reset()
        
        cv2.destroyAllWindows()


def load_existing_config():
    """Load existing configuration"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            print(f"✓ Loaded existing configuration from {CONFIG_FILE}")
            return config
        except Exception as e:
            print(f"✗ Error loading config: {e}")
            return None
    return None


if __name__ == "__main__":
    import sys
    
    VIDEO_PATH = "input.MP4"
    
    print("\n" + "="*70)
    print("  Starting Mehsana ROI Configurator")
    print("="*70)
    
    # Check command line arguments
    mode = "LINE"  # Default mode
    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        if arg in ["LINE", "RECTANGLE", "L", "R"]:
            if arg in ["R"]:
                mode = "RECTANGLE"
            else:
                mode = "LINE" if arg != "RECTANGLE" else "RECTANGLE"
    
    # If no argument, ask user
    if len(sys.argv) == 1:
        print("\nSelect ROI Configuration Mode:")
        print("  1. LINE (L1, L2, Detection lines) - default")
        print("  2. RECTANGLE (2-point rectangle)")
        choice = input("\nEnter choice (1/2) or press Enter for LINE: ").strip()
        if choice == "2":
            mode = "RECTANGLE"
        else:
            mode = "LINE"
    
    print(f"\nSelected Mode: {mode}")
    print("="*70 + "\n")
    
    configurator = ROIConfigurator(VIDEO_PATH, mode=mode)
    configurator.run()
    
    # Load and display saved config
    saved_config = load_existing_config()
    if saved_config:
        print("\nCurrent Configuration:")
        config_mode = saved_config.get('mode', 'LINE')
        print(f"  Mode: {config_mode}")
        if config_mode == "LINE":
            print(f"  L1 Line:        {saved_config.get('l1_line', {})}")
            print(f"  L2 Line:        {saved_config.get('l2_line', {})}")
            print(f"  Detection Line: {saved_config.get('detection_line', {})}")
        elif config_mode == "RECTANGLE":
            print(f"  Rectangle ROI:  {saved_config.get('rectangle_roi', [])}")
