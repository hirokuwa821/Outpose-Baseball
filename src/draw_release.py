import cv2


def draw_release(video_path, release_frame):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        "output/release_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame = 0

    while True:

        ret, img = cap.read()

        if not ret:
            break

        if frame == release_frame:

            cv2.putText(
                img,
                "RELEASE!",
                (40,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0,0,255),
                4
            )

        writer.write(img)

        frame += 1

    cap.release()
    writer.release()

    print("リリース動画を保存しました！")