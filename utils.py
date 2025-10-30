# utils.py

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# Initialize the FaceAnalysis model
# This will download the model weights on the first run
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def get_face_embedding(image_bytes):
    """
    Processes an image and returns the facial embedding.
    Returns None if no face is detected.
    """
    # Read the image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Get faces from the image
    faces = app.get(img)

    # Check if any face is detected
    if not faces:
        return None

    # For simplicity, we use the embedding of the first detected face
    embedding = faces[0].normed_embedding
    return embedding