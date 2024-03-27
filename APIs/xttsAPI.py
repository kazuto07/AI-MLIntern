from TTS.api import TTS
from fastapi import FastAPI
from fastapi.responses import FileResponse

app=FastAPI()

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

@app.get('/')
def index():
    return {'message': 'Go to  http://127.0.0.1:8000/docs/'}

@app.post('/',response_class=FileResponse)
def TextToSpeech(text:str):
    
    output=tts.tts_to_file(text=text,
                file_path="output.wav",
                speaker_wav="M.mp3",
                language="en",
                split_sentences=True
                )
    return output

