import argparse
import os, sys

sys.path.append(os.path.dirname(__file__))
from facerecognizer import ImageView, CameraView

def get_command_line_args():
    P = argparse.ArgumentParser()
    P.add_argument("-i", "--image", metavar="file.png", 
        help="Use this static image instead of camera.")
    return P.parse_args()

if __name__ == "__main__":
    args = get_command_line_args()
    window = ImageView(args.image) if args.image else CameraView()
    window.run()
