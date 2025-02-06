from __future__ import annotations

from typing import Generator, List, Optional, Tuple
import cv2
import numpy as np
import torch
import os
from dataclasses import dataclass
from yolox.tracker.byte_tracker import BYTETracker, STrack
from tqdm import tqdm


def generate_frames(video_file: str) -> Generator[np.ndarray, None, None]:
    video = cv2.VideoCapture(video_file)
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        yield frame
    video.release()     


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    def pad(self, padding: float) -> Rect:
        return Rect(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2 * padding,
            height=self.height + 2 * padding
        )

    def contains_point(self, point: Tuple[float, float]) -> bool:
        px, py = point
        return self.x < px < self.x + self.width and self.y < py < self.y + self.height


@dataclass
class Detection:
    rect: Rect
    class_id: int
    confidence: float
    tracker_id: Optional[int] = None

    @classmethod
    def from_results(cls, pred: np.ndarray) -> List[Detection]:
        return [
            Detection(
                rect=Rect(
                    x=float(x_min),
                    y=float(y_min),
                    width=float(x_max - x_min),
                    height=float(y_max - y_min),
                ),
                class_id=int(class_id),
                confidence=float(confidence),
            )
            for x_min, y_min, x_max, y_max, confidence, class_id in pred
        ]


def detections2boxes(detections: List[Detection], with_confidence: bool = True) -> np.ndarray:
    return np.array([
        [
            detection.rect.x,  # x_min
            detection.rect.y,  # y_min
            detection.rect.x + detection.rect.width,  # x_max
            detection.rect.y + detection.rect.height,
            detection.confidence,  # confidence
            detection.class_id,  # class_id
        ] if with_confidence else [
            detection.rect.x,
            detection.rect.y,
            detection.rect.x + detection.rect.width,
            detection.rect.y + detection.rect.height
        ]
        for detection in detections
    ], dtype=float)


def match_detections_with_tracks(detections: List[Detection], tracks: List[STrack]) -> List[Detection]:
    detection_boxes = detections2boxes(detections, with_confidence=False)
    tracks_boxes = np.array([track.tlbr for track in tracks])
    iou = np.maximum(0, 1 - np.abs(tracks_boxes[:, None, :] - detection_boxes[None, :, :]).sum(axis=-1))
    track2detection = np.argmax(iou, axis=1)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] > 0:
            detections[detection_index].tracker_id = tracks[tracker_index].track_id
    return detections


@dataclass(frozen=True)
class VideoConfig:
    fps: float
    width: int
    height: int


def get_video_writer(target_video_path: str, video_config: VideoConfig) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(target_video_path), exist_ok=True)
    return cv2.VideoWriter(
        target_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_config.fps,
        (video_config.width, video_config.height),
    )


SOURCE_VIDEO_PATH = "/home/finn/ByteTrack/clips/121364_0.mp4"
TARGET_VIDEO_PATH = "/home/finn/ByteTrack/clips/output.mp4"
WEIGHTS_BEST = "/home/finn/ByteTrack/best.pt"

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', WEIGHTS_BEST, device=0)

# Initialize BYTETracker
byte_tracker = BYTETracker(BYTETrackerArgs())

# Video configuration
video_config = VideoConfig(fps=30, width=1920, height=1080)
video_writer = get_video_writer(target_video_path=TARGET_VIDEO_PATH, video_config=video_config)

# Process video
for frame in tqdm(generate_frames(SOURCE_VIDEO_PATH)):
    results = model(frame, size=1280)
    detections = Detection.from_results(pred=results.pred[0].cpu().numpy())

    # Track detections
    boxes = detections2boxes(detections)
    boxes_tensor = torch.from_numpy(boxes).float()  # Convert to PyTorch tensor
    tracks = byte_tracker.update(boxes_tensor, frame.shape[:2], frame.shape[:2])
    tracked_detections = match_detections_with_tracks(detections, tracks)

    # Annotate frame
    for detection in tracked_detections:
        x, y, x2, y2 = detection.rect.x, detection.rect.y, detection.rect.x + detection.rect.width, detection.rect.y + detection.rect.height
        color = (0, 255, 0) if detection.tracker_id else (0, 0, 255)
        cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), color, 2)
        if detection.tracker_id:
            cv2.putText(frame, f"ID: {detection.tracker_id}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    video_writer.write(frame)

video_writer.release()