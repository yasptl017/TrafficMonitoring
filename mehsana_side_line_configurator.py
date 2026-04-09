import cv2
import json


CONFIG_FILE = "roi_config.json"
VIDEO_PATH = "input.MP4"


class SideLineConfigurator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.frame = None
        self.display_frame = None

        self.frame_h = None
        self.frame_w = None
        self.display_h = 1080
        self.display_w = 1920
        self.scale_x = 1.0
        self.scale_y = 1.0

        self.divider_line = []
        self.detection_line = []
        self.current_mode = "DIVIDER"
        self.current_point = None

        self.load_first_frame()

    def load_first_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {self.video_path}")
            return False

        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Error: Could not read first frame from {self.video_path}")
            return False

        self.frame = frame
        self.frame_h, self.frame_w = frame.shape[:2]
        self.scale_x = self.frame_w / self.display_w
        self.scale_y = self.frame_h / self.display_h
        self.redraw_frame()

        print(f"Loaded frame: {self.frame_w}x{self.frame_h}")
        print(f"Display size: {self.display_w}x{self.display_h}")
        return True

    def to_actual(self, point):
        x, y = point
        return int(x * self.scale_x), int(y * self.scale_y)

    def to_display(self, point):
        x, y = point
        return int(x / self.scale_x), int(y / self.scale_y)

    def save_config(self):
        if len(self.divider_line) != 2 or len(self.detection_line) != 2:
            print("Error: Draw both the divider line and the detection line before saving.")
            return False

        config = {
            "mode": "SIDE_LINE",
            "process_side": "RIGHT",
            "divider_line": {
                "start": list(self.divider_line[0]),
                "end": list(self.divider_line[1]),
            },
            "detection_line": {
                "start": list(self.detection_line[0]),
                "end": list(self.detection_line[1]),
            },
            "notes": {
                "description": "Vehicles on the right side of the divider line are processed. Left side is ignored. Detection line is used for counting.",
                "created_by": "mehsana_side_line_configurator.py",
            },
        }

        with open(CONFIG_FILE, "w") as file_obj:
            json.dump(config, file_obj, indent=4)

        print(f"Configuration saved to {CONFIG_FILE}")
        print(f"Divider line: {self.divider_line}")
        print(f"Detection line: {self.detection_line}")
        print("Processing side: RIGHT")
        return True

    def reset(self):
        self.divider_line = []
        self.detection_line = []
        self.current_mode = "DIVIDER"
        self.current_point = None
        self.redraw_frame()
        print("Lines reset.")

    def undo(self):
        if self.current_mode == "DETECTION" and self.detection_line:
            removed = self.detection_line.pop()
            self.redraw_frame()
            print(f"Removed detection-line point: {removed}")
        elif self.current_mode in {"DETECTION", "DIVIDER"} and self.divider_line:
            if self.current_mode == "DETECTION" and not self.detection_line:
                self.current_mode = "DIVIDER"
            removed = self.divider_line.pop()
            self.redraw_frame()
            print(f"Removed divider-line point: {removed}")
        else:
            print("No points to undo.")

    def redraw_frame(self):
        if self.frame is None:
            return

        self.display_frame = cv2.resize(self.frame, (self.display_w, self.display_h))

        if len(self.divider_line) >= 1:
            start_disp = self.to_display(self.divider_line[0])
            cv2.circle(self.display_frame, start_disp, 8, (0, 255, 255), -1)
            cv2.putText(self.display_frame, "DIV START", (start_disp[0] + 10, start_disp[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if len(self.divider_line) == 2:
                end_disp = self.to_display(self.divider_line[1])
                cv2.line(self.display_frame, start_disp, end_disp, (0, 0, 255), 3)
                cv2.circle(self.display_frame, end_disp, 8, (0, 255, 255), -1)
                cv2.putText(self.display_frame, "DIV END", (end_disp[0] + 10, end_disp[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif self.current_mode == "DIVIDER" and self.current_point is not None:
                cv2.line(self.display_frame, start_disp, self.current_point, (0, 165, 255), 2)

        if len(self.detection_line) >= 1:
            start_disp = self.to_display(self.detection_line[0])
            cv2.circle(self.display_frame, start_disp, 8, (255, 200, 0), -1)
            cv2.putText(self.display_frame, "DET START", (start_disp[0] + 10, start_disp[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            if len(self.detection_line) == 2:
                end_disp = self.to_display(self.detection_line[1])
                cv2.line(self.display_frame, start_disp, end_disp, (255, 0, 255), 3)
                cv2.circle(self.display_frame, end_disp, 8, (255, 200, 0), -1)
                cv2.putText(self.display_frame, "DET END", (end_disp[0] + 10, end_disp[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            elif self.current_mode == "DETECTION" and self.current_point is not None:
                cv2.line(self.display_frame, start_disp, self.current_point, (255, 100, 255), 2)

        if self.current_point is not None:
            x, y = self.current_point
            cv2.line(self.display_frame, (x - 20, y), (x + 20, y), (255, 255, 0), 2)
            cv2.line(self.display_frame, (x, y - 20), (x, y + 20), (255, 255, 0), 2)
            cv2.circle(self.display_frame, (x, y), 4, (255, 255, 0), -1)

        self.draw_ui()

    def draw_ui(self):
        h, _ = self.display_frame.shape[:2]
        ready = len(self.divider_line) == 2 and len(self.detection_line) == 2
        if ready:
            status = "READY TO SAVE"
        elif self.current_mode == "DIVIDER":
            status = "DRAW DIVIDER LINE"
        else:
            status = "DRAW DETECTION LINE"
        color = (0, 255, 0) if ready else (0, 255, 255)

        cv2.putText(self.display_frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.putText(self.display_frame, "Step 1: draw divider line for right-side processing", (20, h - 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(self.display_frame, "Step 2: draw detection line for counting", (20, h - 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(self.display_frame, "Right side of divider line will be processed", (20, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(self.display_frame, "Left side of divider line will be ignored", (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        cv2.putText(self.display_frame, "Keys: s=save  u=undo  r=reset  q=quit", (20, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

    def mouse_callback(self, event, x, y, flags, param):
        self.current_point = (x, y)

        if event == cv2.EVENT_MOUSEMOVE:
            self.redraw_frame()
            return

        if event == cv2.EVENT_LBUTTONDOWN and self.current_mode == "DIVIDER" and len(self.divider_line) < 2:
            actual_point = self.to_actual((x, y))
            self.divider_line.append(actual_point)
            print(f"Divider point {len(self.divider_line)}: display=({x}, {y}) actual={actual_point}")
            if len(self.divider_line) == 2:
                self.current_mode = "DETECTION"
                print("Divider line complete. Now draw the detection line.")
            self.redraw_frame()
        elif event == cv2.EVENT_LBUTTONDOWN and self.current_mode == "DETECTION" and len(self.detection_line) < 2:
            actual_point = self.to_actual((x, y))
            self.detection_line.append(actual_point)
            print(f"Detection point {len(self.detection_line)}: display=({x}, {y}) actual={actual_point}")
            self.redraw_frame()

    def run(self):
        if self.frame is None:
            return

        print("\n" + "=" * 70)
        print("  MEHSANA SIDE-LINE CONFIGURATOR")
        print("=" * 70)
        print("  1. Click 2 points to draw the divider line")
        print("  2. Vehicles on the RIGHT side of divider line will be processed")
        print("  3. Vehicles on the LEFT side of divider line will be ignored")
        print("  4. Click 2 points to draw the detection line")
        print("  5. Press 's' to save to roi_config.json")
        print("=" * 70 + "\n")

        window_name = "Mehsana Side Line Configurator"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while True:
            cv2.imshow(window_name, self.display_frame)
            key = cv2.waitKey(50) & 0xFF

            if key == ord("q"):
                print("Quit without saving.")
                break
            if key == ord("u"):
                self.undo()
            elif key == ord("r"):
                self.reset()
            elif key == ord("s"):
                if self.save_config():
                    break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    configurator = SideLineConfigurator(VIDEO_PATH)
    configurator.run()
