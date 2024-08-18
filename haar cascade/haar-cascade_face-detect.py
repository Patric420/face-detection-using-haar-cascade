import cv2

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        coords.append((x, y, w, h))
    
    return coords


def detect_faces_and_eyes(img, faceCascade, eyeCascade):
    colors = {"face": (255, 0, 0), "eye": (0, 0, 255)}
    faces = draw_boundary(img, faceCascade, 1.1, 10, colors['face'], "Face")
    
    for (x, y, w, h) in faces:
        roi_img = img[y:y + h, x:x + w]
        draw_boundary(roi_img, eyeCascade, 1.1, 12, colors['eye'], "Eye")
    
    return img


def main():
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        processed_frame = detect_faces_and_eyes(frame, faceCascade, eyeCascade)
        cv2.imshow("Face and Eye Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
