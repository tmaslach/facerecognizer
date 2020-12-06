"""
Main launching point for the facerecognizer app.

Run with --help for more explanation
"""

import argparse
import os, sys

sys.path.append(os.path.dirname(__file__))
from facerecognizer import *

def get_command_line_args():
    P = argparse.ArgumentParser(description="""\
This app will recognize your face after some training.
When run with no options, you should see your camera view.
Remember faces by clicking on them and retrain within the app
itself.  When you run, all hotkeys will be displayed 
in the console.""")
    P.add_argument("-i", "--image", metavar="file.png", 
        help="Use this static image instead of camera.")
    return P.parse_args()

if __name__ == "__main__":
    args = get_command_line_args()
    view = ImageView(args.image) if args.image else CameraView()
    MainWindow(view).run()
