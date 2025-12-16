import base64
import re
import numpy as np
import cv2
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sklearn import datasets
from sklearn.svm import SVC
import uvicorn

# --- App Initialization ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- AI Model Training ---
# Load the digits dataset from scikit-learn and train a simple SVM classifier.
# This happens only once when the server starts.
digits = datasets.load_digits()
model = SVC(gamma=0.001, C=100.)
model.fit(digits.data, digits.target)

# --- Pydantic Model for Request Body ---
class ImageData(BaseModel):
    image_data: str

# --- Image Processing Function ---
def process_image(image_data_url: str) -> np.ndarray:
    """
    Takes a base64 image data URL from the canvas, processes it, 
    and returns a numpy array suitable for the scikit-learn model.
    """
    # 1. Decode Base64
    # Remove the "data:image/png;base64," prefix
    img_str = re.search(r'base64,(.*)', image_data_url).group(1)
    img_bytes = base64.b64decode(img_str)
    
    # 2. Convert to numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 3. Preprocess for the model
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to 8x8 (the model's expected input size)
    resized_img = cv2.resize(gray_img, (8, 8), interpolation=cv2.INTER_AREA)
    
    # The scikit-learn digits dataset uses 0-16 range.
    # The canvas gives us 0-255. We need to scale it.
    # We also need to make sure the background is black (0) and digit is white/gray (non-zero),
    # which is already the case from our canvas.
    scaled_img = resized_img / 16 # from 0-255 to 0-15.9375
    
    # Flatten the 8x8 image to a 1D array of 64 features
    flattened_img = scaled_img.reshape(1, 64)

    return flattened_img

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main HTML page with the drawing canvas."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(image_data: ImageData):
    """
    Receives image data from the frontend, processes it,
    predicts the digit using the trained model, and returns the prediction.
    """
    try:
        # 1. Process the incoming image
        processed_image = process_image(image_data.image_data)
        
        # 2. Make a prediction
        prediction = model.predict(processed_image)
        
        # 3. Return the result
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

# --- Main entry point for running the server ---
if __name__ == "__main__":
    # This allows running the script directly for development
    # Command: python main.py
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
