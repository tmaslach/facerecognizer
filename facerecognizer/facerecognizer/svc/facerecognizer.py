import os
import cv2
import pickle
import numpy as np
import time

def to_fullpath(filename):
    this_dir = os.path.dirname(__file__)
    return os.path.join(this_dir, filename)    

class FaceRecognizer:
    embedder_file    = to_fullpath("data/nn4.small2.v1.t7")
    # All user data/cache files go into top level folder
    dataset_folder   = to_fullpath("../../../__datasets__")
    recognizer_pkl   = to_fullpath("../../../__cache__/svc/recognizer.pkl")
    labelencoder_pkl = to_fullpath("../../../__cache__/svc/le.pkl")

    def __init__(self):
        self.embedder = cv2.dnn.readNetFromTorch(self.embedder_file)
        
        try:
            with open(self.recognizer_pkl, "rb") as f:
                self.recognizer = pickle.loads(f.read())
            with open(self.labelencoder_pkl, "rb") as f:
                self.label_encoder = pickle.loads(f.read())
        except FileNotFoundError as e:
            self.recognizer = None
            self.label_encoder = None
            print ("No recognizer database found.  No face recognition will occur until faces are memorized")

    def recognize(self, face):
        # Is the image too small to work?
        height, width = face.shape[:2]
        if height < 20 or width < 20:
            return "small", 1

        if not self.recognizer or not self.label_encoder:
            return "unknown", 1

        embedding   = self.calculate_face_embedding(face)
        predictions = self.recognizer.predict_proba(embedding)[0]
        best_index  = np.argmax(predictions)
        probability = predictions[best_index]
        name        = self.label_encoder.classes_[best_index]

        return name, probability

    def get_all_names(self):
        names = [f.name for f in os.scandir(self.dataset_folder) if f.is_dir()]
        return names

    def remember(self, name, face):
        name_folder = os.path.join(self.dataset_folder, name)
        os.makedirs(name_folder, exist_ok=True)

        filename = time.strftime("%Y%m%d_%H%M%S.png")
        filename = os.path.join(name_folder, filename)
        cv2.imwrite(filename, face)

    def calculate_face_embedding(self, face):
        face_blob = cv2.dnn.blobFromImage(face,
            scalefactor = 1.0/255.0,
            size        = (96, 96),
            mean        = (0, 0, 0),
            swapRB      = True,
            crop        = False
        )
        self.embedder.setInput (face_blob)
        return self.embedder.forward()

    def train(self):
        from sklearn.preprocessing import LabelEncoder
        from sklearn.svm import SVC

        names, embeddings = self.get_embeddings()

        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(names)

        self.recognizer = SVC(C=1.0, kernel="linear", probability=True)
        self.recognizer.fit(embeddings, labels)

        self.save_recognizer_and_encoder()
        
    def save_recognizer_and_encoder(self):
        # Save for future runs of app
        os.makedirs(os.path.dirname(self.recognizer_pkl), exist_ok=True)
        os.makedirs(os.path.dirname(self.labelencoder_pkl), exist_ok=True)
        with open(self.recognizer_pkl, "wb") as f:
            f.write(pickle.dumps(self.recognizer))
        with open(self.labelencoder_pkl, "wb") as f:
            f.write(pickle.dumps(self.label_encoder))

    def get_embeddings(self):
        dataset_dirs = os.listdir(self.dataset_folder)
        dataset_dirs = [d for d in dataset_dirs if os.path.isdir(self.dataset_folder+"/"+d)]

        names = []
        embeddings = []

        for dataset_dir in dataset_dirs:
            name = dataset_dir
            dataset_dir = os.path.join(self.dataset_folder, dataset_dir)
            for file in os.listdir(dataset_dir):
                face = cv2.imread(os.path.join(dataset_dir, file))
                embedding = self.calculate_face_embedding(face)

                names.append(name)
                embeddings.append(embedding.flatten())
        return names, embeddings
