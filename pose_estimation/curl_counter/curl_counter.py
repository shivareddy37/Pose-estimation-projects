"""
This a simple program which counts the number of Biceps Curl done 
using a Tensorflow MoveNet-lightning network.
"""

import tensorflow as tf
import numpy as np
import cv2

# Edge collection as per model
EDGES = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (2, 4): "c",
    (0, 5): "m",
    (0, 6): "c",
    (5, 7): "m",
    (7, 9): "m",
    (6, 8): "c",
    (8, 10): "c",
    (5, 6): "y",
    (5, 11): "m",
    (6, 12): "c",
    (11, 12): "y",
    (11, 13): "m",
    (13, 15): "m",
    (12, 14): "c",
    (14, 16): "c",
}
NAMED_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right ankle": 16,
}


def draw_connection(frame, keypoints, edges, confidence_thresh):
    h, w = frame.shape[0:2]
    shaped = np.squeeze(np.multiply(keypoints, [h, w, 1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if c1 > confidence_thresh and c2 > confidence_thresh:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def draw_keypoints(frame, keypoints, confidence_thresh):
    h, w = frame.shape[0:2]
    # get to image frame as coordinates are normalized
    shaped = np.squeeze(np.multiply(keypoints, [h, w, 1]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_thresh:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def main():
    # loading movenet lightning model
    model = tf.lite.Interpreter(
        model_path="lite-model_movenet_singlepose_lightning_3.tflite"
    )
    model.allocate_tensors()
    counter = 0
    stage = None
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        # Image reshape
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)

        # setup inpu and output
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        # make preds
        model.set_tensor(input_details[0]["index"], np.array(input_image))
        model.invoke()
        try:
            keypoints_with_score = model.get_tensor(output_details[0]["index"])

            draw_keypoints(frame, keypoints_with_score, 0.4)
            draw_connection(frame, keypoints_with_score, EDGES, 0.4)

            h, w = frame.shape[0:2]
            shaped = np.squeeze(np.multiply(keypoints_with_score, [h, w, 1]))
            left_shoulder = shaped[NAMED_KEYPOINTS["left_shoulder"]]
            left_elbow = shaped[NAMED_KEYPOINTS["left_elbow"]]
            left_wrist = shaped[NAMED_KEYPOINTS["left_wrist"]]
            # Calculate angle
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            # Visualize angle
            # cv2.putText(frame, str(angle),
            #                 (int(left_elbow[0]), int(left_elbow[1])),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #             )

            # curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == "down":
                stage = "up"
                counter += 1
        except:
            pass

        # rendering count information onto frame
        cv2.rectangle(frame, (0, 0), (225, 73), (245, 117, 16), -1)
        # Rep data
        cv2.putText(
            frame,
            "REPS",
            (15, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            str(counter),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("moveNet Lightning", frame)
        c = cv2.waitKey(1)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
