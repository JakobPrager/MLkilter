import cv2
import os

#extract video frames

def extract_unique_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    frame_count = 0

    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        if not frames_are_equal(prev_frame, frame):
            frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            prev_frame = frame
            frame_count += 1
            

    cap.release()

def frames_are_equal(frame1, frame2):
    if frame1 is None or frame2 is None:
        return False
    norm_value = cv2.norm(frame1, frame2)
    threshold = 0.005 * frame1.size  # 1% of the total number of pixels
    return norm_value <= threshold

# Example usage
extract_unique_frames('recording_holds.mp4', 'output_frames')