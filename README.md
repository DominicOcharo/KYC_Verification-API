# KYC_Verification API

This FastAPI-based API serves as a backend for processing and analyzing various types of documents, including images and PDFs. It utilizes computer vision models for object detection and OCR (Optical Character Recognition) to extract text from images. The API supports uploading images for ID verification and CR12 PDFs for extraction of textual information.

## API Endpoints

### POST /predict - ID Verification

This endpoint accepts two image files (front_image and back_image) of KENYAN ID cards as input. It processes each image using a YOLOv8 model for object detection and EasyOCR for text extraction. It returns detected texts with the two images id.

#### URL
```
http://127.0.0.1:8000/predict
```

#### Request Body
- **formdata**
  - `front_image`: Image file of the front side of the ID card
  - `back_image`: Image file of the back side of the ID card

#### Example Request
```bash
curl --location 'http://127.0.0.1:8000/predict' \
--form 'front_image=@"/C:/Users/Admin/Desktop/desktop folders/ID CARD/IMG_20220214_082553.jpg"' \
--form 'back_image=@"/C:/Users/Admin/Desktop/desktop folders/ID CARD/back.png"'
```

#### Example Response
```json
{
  "front_image_id": "846cc8a5-3ba0-4984-9783-81d9df6e7431",
  "back_image_id": "0506b640-ec01-4ead-963b-a11c08fe1c9a",
  "front_texts": {
    "PLACE OF ISSUE": "CE****L",
    "FULL NAMES": "*****NIC ***NA *****RO",
    "GENDER": "MALE",
    "ID NUMBER": "36****80",
    "DATE OF ISSUE": "17.01.***8",
    "SERIAL NUMBER": "24*****57",
    "DISTRICT OF BIRTH": "KA*****A **N",
    "DATE OF BIRTH": "2*.*1.1***"
  },
  "back_texts": {
    "SERIALS MATCH": true,
    "ID MATCH": true,
    "NAME MATCH": true,
    "DISTRICT": "KI******",
    "DIVISION": "KI******",
    "LOCATION": "BE*****",
    "SUB-LOCATION": "NA****"
  }
}
```

### POST /cr12 - CR12 Verification

This endpoint accepts CR12 documents as a PDF file. It converts the PDF into images and processes the first page using a YOLOv8 model. It extracts text using EasyOCR and returns the extracted text along with the image id.

#### URL
```
http://localhost:8000/cr12
```

#### Request Body
- **formdata**
  - `pdf_file`: PDF file of the CR12 document

#### Example Request
```bash
curl --location 'http://localhost:8000/cr12' \
--form 'pdf_file=@"/C:/Users/Admin/Desktop/desktop folders/ID CARD/cr12.pdf"'
```

#### Example Response
```json
{
  "image_id": "bb1bcee6-5afc-4e77-ad97-987af7dea6c2",
  "detected_texts": {
    "Company Number": "CPR/2014/149365",
    "REF Number": "OS-S2FXKJBA",
    "Company Name": "Q-WAYS LIMITED",
    "Name and Description": [
      "NJOROGE O KIMANI: SECRETARY",
      "DHANANJAYAN SANGEETHA: DIRECTOR/SHAREHOLDER",
      "DHANANJAYAN KALATHIL PADUVILAN: DIRECTOR/SHAREHOLDER"
    ],
    "Name and Nationality": [
      "NJOROGE O KIMANI: N/A",
      "DHANANJAYAN SANGEETHA: INDIA",
      "DHANANJAYAN KALATHIL PADUVILAN: KENYAN"
    ],
    "Registration Date": "2ND JUL 2014"
  }
}
```

### GET /image/{image_id} - View Verified Image

This endpoint retrieves a processed image by its unique image_id. The image would have been previously processed and saved during the `/predict` or `/cr12` endpoint executions.

#### URL
```
http://localhost:8000/image/{image_id}
```

#### Example Request
```bash
curl --location 'http://localhost:8000/image/bb1bcee6-5afc-4e77-ad97-987af7dea6c2'
```

## Running the API
To run the API, use the following command in your terminal:
```bash
uvicorn main:app --reload
```

This command will start the FastAPI server, and you can access the endpoints as described above.
