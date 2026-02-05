from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
import io, base64, tempfile
from PIL import Image
import speech_recognition as sr

# Initialize FastAPI app
app = FastAPI()

# CORS so frontend on port 3000 can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Gemini client
client = genai.Client(api_key="API KEY")

@app.post("/chat/")
async def chat_endpoint(text: str = Form(...)):
    try:
        # Send message to Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"You are a helpful medical assistant. {text}"
        )
        return {"reply": response.text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/vision/")
async def vision_endpoint(file: UploadFile = File(...), question: str = Form("")):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        img_bytes_io = io.BytesIO()
        image.save(img_bytes_io, format="PNG")
        img_b64 = base64.b64encode(img_bytes_io.getvalue()).decode("utf-8")

        contents = {
            "parts": [
                {"text": f"You are a helpful medical assistant. {question or ''}"},
                {"inline_data": {"mime_type": "image/png", "data": img_b64}}
            ]
        }
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=contents
        )
        return {"reply": response.text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/voice/")
async def voice_endpoint(file: UploadFile = File(...)):
    try:
        # Save temp wav file
        file_bytes = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # Transcribe speech
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(data)
            except sr.UnknownValueError:
                text = "[Could not understand audio]"
            except sr.RequestError as e:
                text = f"[Speech recognition error: {e}]"

        # Respond with AI output
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"You are a helpful medical assistant. {text}"
        )
        return {"recognized_text": text, "reply": response.text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
