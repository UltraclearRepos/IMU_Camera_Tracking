import cv2

def get_camera_info(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Cannot open camera with index {index}")
        return

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera (index {index}):")
    print(f"Resolution: {int(width)}x{int(height)}")
    print(f"FPS: {fps}")

    cap.release()

if __name__ == "__main__":
    # Check index 1
    get_camera_info(1)
    # Check index 0 in case index 1 is unavailable
    print("-" * 20)
    get_camera_info(0)
