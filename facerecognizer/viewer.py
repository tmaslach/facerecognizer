"""This is the main application, the CameraView and ImageView.
However, most code resides in BaseView, since there is a huge
overlapping in behavior"""

import facerecognizer
import cv2
import tkinter
from tkinter import ttk

from PIL import Image, ImageTk

from .facedetector import FaceDetector
from .facerecognizer.svc import FaceRecognizer

class MainWindow:
    def __init__ (self, view):
        self.view = view
        self.view.window = self

        self.root = tkinter.Tk()
        self.root.protocol("WM_DELETE_WINDOW", lambda *args: self.view.destroy())
        
        self.image_frame = ttk.Label(self.root)
        self.input_frame = ttk.Frame(self.root)

        self.name_label = ttk.Label(self.input_frame, 
            text="Name of person clicked on: ")

        self.name_entry = ttk.Combobox(self.input_frame, 
            values=self.view.get_all_names())
        self.name_entry.bind("<Key>", self.set_name)
        self.name_entry.bind("<<ComboboxSelected>>", self.set_name)

        self.clear_name_button = ttk.Button(self.input_frame, text="Clear User",
            command=self.clear_name) 
        self.train_button = ttk.Button(self.input_frame, text="Train",
            command=self.view.train)

        self.image_frame.pack(fill=tkinter.BOTH, expand=True)
        
        self.input_frame.pack(fill=tkinter.X, expand=True)
        self.name_label.pack(side=tkinter.LEFT)
        self.name_entry.pack(fill = tkinter.X, expand=True, side=tkinter.LEFT)
        self.clear_name_button.pack(side=tkinter.LEFT)
        self.train_button.pack(side=tkinter.LEFT)

        self.set_bindings()

    def set_bindings(self):
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("t", lambda e: self.view.train())       
        self.image_frame.bind("<Button-1>", self.on_mouse_left_click)

    def on_mouse_left_click(self, event):
        self.view.memorize_face(event.x, event.y)

    def set_name (self, event):
        self.view.assumed_name = event.widget.get()
        event.widget["values"] = self.view.get_all_names()
        
    def clear_name (self):
        self.view.assumed_name = ""
        self.name_entry.set("")

    def show_error (self, title, text):
        from tkinter import messagebox
        messagebox.showerror(title, text)

    def show_image (self, rgba_image):
        img = Image.fromarray(rgba_image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_frame.imgtk = imgtk
        self.image_frame.configure(image=imgtk)

    def run(self):
        def update():
            self.view.update_frame()
            self.root.after(1, update)

        self.root.after(1, update)
        tkinter.mainloop()

class BaseView:
    """Base class for Viewers, which contains most behavior"""

    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()

    def __init__ (self):
        self.window = None # Will be set when put in main window
        self.detections = []
        self.assumed_name = None

    def memorize_face(self, x, y):
        if not self.assumed_name:
            self.window.show_error("Specify name first!", "Name must be specified before selecting faces")
            return

        for hit in self.detections:
            if hit.start_x <= x <= hit.end_x and hit.start_y <= y <= hit.end_y:
                break
        else:
            return

        face = self.get_face(hit)
        self.face_recognizer.remember(self.assumed_name, face)

    def destroy(self):
        """Override if anything needs to be cleared when destroying view"""

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

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        self.window.show_image(cv2image)

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

    def get_all_names(self):
        return self.face_recognizer.get_all_names()

    def train(self):
        self.face_recognizer.train()

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
        self.video_capture.release()
