import os
import cv2
import numpy as np
from deepface import DeepFace

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Create Dataset
data_dir = "face_dataset"
os.makedirs(data_dir, exist_ok=True)

def create_dataset(name):
    person = os.path.join(data_dir, name)
    os.makedirs(person, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    while count < 50:  # Limit image capture to 50 images
        ret, frame = cap.read()
        if not ret:
            print("Cannot Capture Image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y+h, x:x+w]
            face_path = os.path.join(person, f"{name}_{count}.jpg")

            cv2.imwrite(face_path, face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Capturing Faces - Press 'q' to Stop", frame)

        # Exit loop when 'q' is pressed or 50 images are captured
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Dataset created for {name} with {count} images.")

# Train Facial Dataset
def train_dataset():
    embeddings = {}
    for person_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_name)

        if os.path.isdir(person_path):
            embeddings[person_name] = []

            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)

                try:
                    emb = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                    embeddings[person_name].append(emb)
                except Exception as e:
                    print(f"Failed to train image {img_name}: {e}")

    return embeddings

# Recognize Faces
def recognize_faces(embeddings):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            try:
                analysis = DeepFace.analyze(face_img, actions=["age", "gender", "emotion"], enforce_detection=False)

                if isinstance(analysis, list):
                    analysis = analysis[0]

                age = analysis["age"]
                gender = analysis["gender"]
                gender = gender if isinstance(gender, str) else max(gender, key=gender.get)
                emotion = max(analysis["emotion"], key=analysis["emotion"].get)

                face_embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]

                match = None
                max_similarity = -1

                for person, person_embeddings in embeddings.items():
                    for embed in person_embeddings:
                        similarity = np.dot(face_embedding, embed) / (np.linalg.norm(face_embedding) * np.linalg.norm(embed))

                        if similarity > max_similarity:
                            max_similarity = similarity
                            match = person

                if max_similarity > 0.7:
                    label = f"{match} ({max_similarity:.2f})"
                else:
                    label = "Unknown Person"

                display_text = f"{label}, Age: {int(age)}, Gender: {gender}, Emotion: {emotion}"

                cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            except Exception as e:
                print("Cannot recognize face:", e)

        cv2.imshow("Recognizing Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Execution
if __name__ == "__main__":
    print("1. Create Face Dataset\n2. Train Face Dataset\n3. Recognize Face")

    choice = input("Enter Your Choice: ")

    if choice == "1":
        name = input("Enter name of the person: ")
        create_dataset(name)

    elif choice == "2":
        embeddings = train_dataset()
        np.save("embeddings.npy", embeddings)

    elif choice == "3":
        if os.path.exists("embeddings.npy"):
            embeddings = np.load("embeddings.npy", allow_pickle=True).item()
            recognize_faces(embeddings)
        else:
            print("File not found!")

    else:
        print("Invalid Choice")
