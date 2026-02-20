import cv2
import numpy as np
from ultralytics import YOLO
import json

print("=" * 50)
print("HOMEMADE SWINGVISION — PLAYERS ONLY")
print("=" * 50)

class SimpleSwingVision:
    def __init__(self):
        print("Initializing...")
        self.model = YOLO('yolov8n.pt')

        # Tracking
        self.player1_pos = []
        self.player2_pos = []

        # Stats
        self.frame_count = 0

        # Colors
        self.colors = {
            'player1': (0, 255, 0),      # Green
            'player2': (255, 165, 0)     # Orange
        }

    def detect_players(self, frame):
        """Detect and track two main players"""
        results = self.model(frame, verbose=False)
        players = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls) == 0:  # Person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if (x2 - x1) > 80 and (y2 - y1) > 160:
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        area = (x2 - x1) * (y2 - y1)
                        players.append((area, cx, cy, x1, y1, x2, y2))

        players.sort(reverse=True, key=lambda x: x[0])

        player_boxes = []
        for i, (_, cx, cy, x1, y1, x2, y2) in enumerate(players[:2]):
            if i == 0:
                player_boxes.append(('player1', x1, y1, x2, y2, cx, cy))
                self.player1_pos.append((cx, cy))
            else:
                player_boxes.append(('player2', x1, y1, x2, y2, cx, cy))
                self.player2_pos.append((cx, cy))

        return player_boxes

    def create_heatmap_overlay(self, frame):
        """Create player movement heatmap"""
        height, width = frame.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)

        for positions in [self.player1_pos, self.player2_pos]:
            for x, y in positions[-30:]:
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(heatmap, (x, y), 25, 1.0, -1)

        heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        return cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Video: {width}x{height} @ {fps:.1f} FPS")
        print("Press 'q' to quit, 'p' to pause")

        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

            player_boxes = self.detect_players(frame)

            display = frame.copy()
            display = self.create_heatmap_overlay(display)

            for label, x1, y1, x2, y2, cx, cy in player_boxes:
                color = self.colors[label]
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.circle(display, (cx, cy), 5, color, -1)
                cv2.putText(
                    display,
                    label.capitalize(),
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            cv2.putText(
                display,
                f"Frame: {self.frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("SwingVision — Players Only", display)

            if not paused:
                self.frame_count += 1

            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused

        cap.release()
        cv2.destroyAllWindows()
        self.save_results()

    def save_results(self):
        results = {
            "frames_processed": self.frame_count,
            "player1_positions": len(self.player1_pos),
            "player2_positions": len(self.player2_pos)
        }

        with open("player_tracking_results.json", "w") as f:
            json.dump(results, f, indent=2)

        np.save("player1_positions.npy", self.player1_pos)
        np.save("player2_positions.npy", self.player2_pos)

        print("\nAnalysis complete.")
        print("Saved:")
        print("✓ player_tracking_results.json")
        print("✓ player1_positions.npy")
        print("✓ player2_positions.npy")


# RUN
if __name__ == "__main__":
    import os

    video_file = "test.mp4"
    if not os.path.exists(video_file):
        video_file = input("Enter video filename: ").strip()

    if os.path.exists(video_file):
        analyzer = SimpleSwingVision()
        analyzer.process_video(video_file)
    else:
        print("Video file not found.")