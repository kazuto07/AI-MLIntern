from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import librosa
app=FastAPI()

model = WhisperModel("large-v2")

@app.post("/whisper")
async def transcribe_audio(file: UploadFile = File(...)):

    audio_content,sr = librosa.load(file.file)
    transcription,info = model.transcribe(audio_content)
    print(transcription)
    strings_only = [entry[4] for entry in transcription]
    output_string = ' '.join(strings_only)

    return {"transcription": output_string}

