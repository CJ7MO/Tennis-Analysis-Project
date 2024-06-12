from utils.common import read_video, save_video
from trackers.player_trackers import PlayerTracker
from trackers.ball_tracker import BallTracker
from court_line_detector.court_line_detector import CourtLineDetector


def main():
    # read video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # detect players
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stub=True, 
                                                     stub_path="tracker_stubs/player_detections.pkl")
    
    # detect ball
    ball_tracker = BallTracker(model_path='models/last.pt')
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=True,
                                                 stub_path="tracker_stubs/ball_detections.pkl")
    
    # detect court lines
    court_model_path = "models\keypoints_model.pth"
    court_line_detector = CourtLineDetector(model_path=court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # draw player bounding box
    output_video_frames = player_tracker.draw_bbox(video_frames, player_detections)

    # draw ball bounding box
    output_video_frames = ball_tracker.draw_bbox(video_frames, ball_detections)

    # draw court lines keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # save video
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()