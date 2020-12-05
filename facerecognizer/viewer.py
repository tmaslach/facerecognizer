import cv2

from .facedetector import FaceDetector
from .facerecognizer.svc import FaceRecognizer

class BaseView:
    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()

    class KeyBindings:
        """Simple dictionary-like container of key bindings to action, help"""
        def __init__(self):
            self.bindings = dict()

        def add(self, key, action, help=""):
            if key == "ESC": 
                key_val = 27 
            else:            
                key_val = ord(key)
            self.bindings[key_val] = (key, action, help)

        def getAction(self, key):
            if key not in self.bindings: 
                return
            return self.bindings[key][1]

        def print(self):
            print ("Key Bindings:")
            for key, action, help in self.bindings.values():
                print (f"  {key}: {help}")

    def __init__ (self):
        self.name = 'Viewer'
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self.on_mouse)

        self.set_bindings()
        self.detections = []
        self.assumed_user = None

    def set_bindings(self):
        self.key_bindings = self.KeyBindings()
        self.key_bindings.add('ESC', self.destroy,   help="Quit the application")
        self.key_bindings.add('u', self.assume_user, 
            help="On mouse click, assumes user specified on 'u' to be the user. Enter no name for 'u' to stop.")
        self.key_bindings.add('t', self.face_recognizer.train, 
            help="Train new recognizer off faces remembered")
        self.key_bindings.print()

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.memorize_face(x, y)

    def memorize_face(self, x, y):
        for hit in self.detections:
            if hit.start_x <= x <= hit.end_x and hit.start_y <= y <= hit.end_y:
                break
        else:
            return
        
        name = self.assumed_user if self.assumed_user else input("Who is this: ")
        face = self.get_face(hit)
        self.face_recognizer.remember(name, face)

    def assume_user(self):
        self.assumed_user = input("Which user should I assume: ")
        
    def run(self):
        done = False
        while(not done):
            self.update_frame()

            # Escape or q quit event loop
            key = cv2.waitKey(1) & 0xFF
            key_action = self.key_bindings.getAction(key)
            if key_action:
                key_action()
            done = cv2.getWindowProperty(self.name, 0) < 0

        self.destroy()

    def destroy(self):
        cv2.destroyWindow(self.name)

    def show_frame(self, frame):
        self.raw_frame = frame.copy()
        self.detections = self.face_detector.scan(frame)

        for hit in self.detections:
            cv2.rectangle(frame, 
                pt1       = (hit.start_x, hit.start_y),
                pt2       = (hit.end_x, hit.end_y),
                color     = (255, 0, 255),
                thickness = 1
            )

            text = f"{hit.confidence*100:.2f}%"
            self.put_text(frame, text, hit.start_x, hit.start_y)

            face = self.get_face(hit)
            name, probability = self.face_recognizer.recognize(face)

            text = f"{name}: {probability*100:.2f}%"
            self.put_text(frame, text, hit.start_x, hit.end_y)

        if self.assumed_user:
            self.put_text(frame, f"User Assumed: {self.assumed_user}", 0, 0)

        cv2.imshow(self.name, frame)

    def put_text(self, frame, text, x, y):
        y = max(15, y)
        y = min(self.height, y)
        
        cv2.putText(frame, text, 
            org       = (x, y),
            fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.65,
            color     = (255, 0, 255),
            thickness = 1
        )

    def get_face(self, hit):
        return self.raw_frame[hit.start_y:hit.end_y, hit.start_x:hit.end_x]

class ImageView(BaseView):
    def __init__ (self, image):
        super().__init__()
        self.image = cv2.imread(image)
        self.height, self.width, _ = self.image.shape
        self.first_time = True

    def update_frame(self):
        if self.first_time:
            self.show_frame(self.image)
            self.first_time = False

class CameraView (BaseView):
    def __init__ (self):
        super().__init__()
        self.video_capture = cv2.VideoCapture(0)
        self.width  = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))  
        self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def update_frame(self):
        _, frame = self.video_capture.read()
        frame = cv2.flip(frame, 1)
        self.show_frame(frame)

    def destroy(self):
        super().destroy()
        self.video_capture.release()
