from flask import Flask, request, jsonify
import os, base64, pickle, datetime, re
import face_recognition

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # enable CORS

DATASET = "dataset"
ENC_FILE = "encodings.pickle"
ATT_FILE = "attendance.csv"

os.makedirs(DATASET, exist_ok=True)


# ---------- ADD STUDENT ----------
@app.route("/api/add-student", methods=["POST"])
def add_student():
    data = request.get_json()
    name = data["name"]
    photo_b64 = data["photo"]

    # Remove header
    photo_b64 = re.sub('^data:image/.+;base64,', '', photo_b64)
    img_bytes = base64.b64decode(photo_b64)

    # Create folder for student
    folder = os.path.join(DATASET, name)
    os.makedirs(folder, exist_ok=True)

    # Save image
    count = len(os.listdir(folder))
    with open(os.path.join(folder, f"{count}.jpg"), "wb") as f:
        f.write(img_bytes)

    return jsonify({"status": "success", "message": "Student image saved!"})


# ---------- TRAIN MODEL ----------
@app.route("/api/train", methods=["GET"])
def train():
    encodings, names = [], []

    for folder in os.listdir(DATASET):
        folder_path = os.path.join(DATASET, folder)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            image = face_recognition.load_image_file(img_path)
            face = face_recognition.face_encodings(image)
            if face:
                encodings.append(face[0])
                names.append(folder)

    pickle.dump({"enc": encodings, "names": names}, open(ENC_FILE, "wb"))
    return jsonify({"status": "success", "message": "Training completed!"})


# ---------- MARK ATTENDANCE ----------
@app.route("/api/attendance", methods=["POST"])
def attendance():
    data = pickle.load(open(ENC_FILE, "rb"))
    known_enc, known_names = data["enc"], data["names"]

    photo_b64 = request.json["photo"]
    photo_b64 = re.sub('^data:image/.+;base64,', '', photo_b64)
    img_bytes = base64.b64decode(photo_b64)

    with open("temp.jpg", "wb") as f:
        f.write(img_bytes)

    unknown_img = face_recognition.load_image_file("temp.jpg")
    unknown_encs = face_recognition.face_encodings(unknown_img)

    found_names = []

    for enc in unknown_encs:
        matches = face_recognition.compare_faces(known_enc, enc)
        if True in matches:
            name = known_names[matches.index(True)]
            now = datetime.datetime.now()
            with open(ATT_FILE, "a") as f:
                f.write(f"{name},{now}\n")
            found_names.append(name)

    return jsonify({"status": "success", "found": found_names})


if __name__ == "__main__":
    app.run(debug=True)
