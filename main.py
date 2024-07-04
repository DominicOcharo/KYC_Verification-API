from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import easyocr
import numpy as np
from PIL import Image
import io
import os
import uuid
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
import logging
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS for local development and testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en', 'sw'], gpu=False)

# Load YOLOv8 models
model = YOLO(r"C:\Users\Admin\Documents\vs python\KenyanIDVerification\Model\best.pt")
cr12_model = YOLO(r"C:\Users\Admin\Documents\vs python\KenyanIDVerification\Model\best1.pt")

# Directory to save processed images
output_dir = "processed_images"
os.makedirs(output_dir, exist_ok=True)

# Explicitly set the path to pdftoppm
poppler_path = r'C:\Program Files (x86)\poppler\Library\bin'

# Define target classes for both models
target_classes_model = [
    'Administrative_Details', 'Birth_District', 'Bottom_Part', 'DOB', 'Fingerprint', 'ID_Back', 
    'ID_Front', 'ID_No', 'Image_Back', 'Image_Front', 'Issue_Date', 'Issue_Place', 'Names', 
    'SEX', 'Serial_No', 'Sign'
]

target_classes_cr12 = [
    'Bar_Code', 'Business_Reg_Service_Address', 'CR12', 'Company_Address', 'Company_Name', 
    'Company_Number', 'Date', 'Description', 'Harambee_Logo', 'Name', 'Nationality', 'PO_BOX', 'REF_No', 'Registration_Date', 'Shares', 'Title'
]

def process_image(image_np, image_id, yolo_model, target_classes):
    try:
        # Detect objects using YOLOv8
        results = yolo_model(image_np)

        detected_regions = {cls: [] for cls in target_classes}
        detected_classes = set()
        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0].numpy().astype(int)
                cls = box.cls[0]
                class_name = yolo_model.names[int(cls)]
                detected_classes.add(class_name)
                if class_name in target_classes:
                    detected_regions[class_name].append(bbox)

        detected_texts = {}
        for cls in detected_classes:
            texts = []
            for bbox in detected_regions[cls]:
                x1, y1, x2, y2 = bbox
                cropped_img = image_np[y1:y2, x1:x2]
                detections = reader.readtext(cropped_img)
                texts.extend([text for _, text, _ in detections])
            detected_texts[cls] = texts if texts else ["Not available"]

        # Save the image (without bounding boxes)
        output_path = os.path.join(output_dir, f"{image_id}.png")
        cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        return detected_texts, output_path
    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        raise HTTPException(status_code=500, detail="Failed to process image")

@app.post("/predict")
async def predict(front_image: UploadFile = File(...), back_image: UploadFile = File(...)):
    try:
        # Read the images
        front_contents = await front_image.read()
        back_contents = await back_image.read()

        front_image_np = np.array(Image.open(io.BytesIO(front_contents)).convert("RGB"))
        back_image_np = np.array(Image.open(io.BytesIO(back_contents)).convert("RGB"))

        front_id = str(uuid.uuid4())
        back_id = str(uuid.uuid4())

        front_texts, front_output_path = process_image(front_image_np, front_id, model, target_classes_model)
        back_texts, back_output_path = process_image(back_image_np, back_id, model, target_classes_model)

        response = {
            "front_image_id": front_id,
            "back_image_id": back_id,
            "front_texts": front_texts,
            "back_texts": back_texts
        }

        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/cr12")
async def cr12(pdf_file: UploadFile = File(...)):
    try:
        # Read the PDF file
        pdf_contents = await pdf_file.read()

        # Convert PDF to images
        pdf_info = pdfinfo_from_bytes(pdf_contents, poppler_path=poppler_path)
        images = convert_from_bytes(pdf_contents, poppler_path=poppler_path)

        if not images:
            raise HTTPException(status_code=400, detail="No images found in PDF")

        # Process the first image from the PDF (can be extended to process all pages)
        image_np = np.array(images[0].convert("RGB"))

        image_id = str(uuid.uuid4())

        detected_texts, output_path = process_image(image_np, image_id, cr12_model, target_classes_cr12)

        response = {
            "image_id": image_id,
            "detected_texts": detected_texts
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"CR12 prediction failed: {e}")
        raise HTTPException(status_code=500, detail="CR12 prediction failed")

@app.get("/image/{image_id}")
async def get_image(image_id: str):
    try:
        image_path = os.path.join(output_dir, f"{image_id}.png")
        if os.path.exists(image_path):
            return FileResponse(image_path)
        else:
            return JSONResponse(content={"message": "Image not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Failed to retrieve image: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve image")
