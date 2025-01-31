from fastapi import FastAPI, File, UploadFile, Form
from pathlib import Path
import shutil
from fastapi.middleware.cors import CORSMiddleware
from model import load_model, predict_image, train_model, save_model

# Initialize FastAPI
app = FastAPI()

# CORS settings
origins = [
    "http://localhost:52330",  
    "http://127.0.0.1:52330", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model (this will be used for prediction)
model = load_model("model.pth")

# Directory to save uploaded images
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upload/")
def upload_file(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Call your prediction function
    class_name = predict_image( str(file_path))
    return {
        "Class": class_name,
        "message": "File uploaded successfully",
        "file_path": f"/uploads/{file.filename}"
    }

@app.post("/train/")
def train_model_endpoint():
    """
    Endpoint to train the model and save it.
    """
    train_model(num_epochs=10, learning_rate=0.0001)
    return {"message": "Model trained successfully"}

@app.post("/add_label/")
def add_label(label_name: str = Form(...)):
    """
    Endpoint to add a new label to the labels.txt file.
    """
    try:
        with open("labels.txt", "a") as file:
            with open("labels.txt", "r") as read_file:
                lines = read_file.readlines()
                next_index = len(lines)

            new_label = f"{next_index} {label_name}\n"
            file.write(new_label)

        return {
            "message": f"Label '{label_name}' added successfully as index {next_index}",
            "new_label": new_label.strip()
        }

    except Exception as e:
        return {
            "message": "Failed to add label",
            "error": str(e)
        }
