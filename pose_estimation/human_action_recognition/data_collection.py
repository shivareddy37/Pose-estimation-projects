"""
A script to collect data for diffrent actions using video feed from your webcam
"""

from pose_estimation.human_action_recognition.helper import (
    actions,
    DATA_PATH,
    extract_landmarks,
    num_sequences,
    mp_holistic,
    mp_drawing,
    mediapipe_detection,
    sequence_length,
    draw_styled_landmarks,
)
import os
import numpy as np
import cv2


def collect_data():
    # making file structure to collect data
    for action in actions:
        for sequence in range(num_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except Exception:
                pass


cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:

    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(num_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                #                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # NEW Apply wait logic
                if frame_num == 0:
                    cv2.putText(
                        image,
                        "STARTING COLLECTION",
                        (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        4,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        image,
                        "Collecting frames for {} Video Number {}".format(
                            action, sequence
                        ),
                        (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    # Show to screen
                    cv2.imshow("OpenCV Feed", image)
                    cv2.waitKey(2500)
                else:
                    cv2.putText(
                        image,
                        "Collecting frames for {} Video Number {}".format(
                            action, sequence
                        ),
                        (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    # Show to screen
                    cv2.imshow("OpenCV Feed", image)

                # NEW Export keypoints
                keypoints = extract_landmarks(results)
                npy_path = os.path.join(
                    DATA_PATH, action, str(sequence), str(frame_num)
                )
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()
