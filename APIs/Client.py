import requests


import io
import soundfile as sf
with open("M.mp3", "rb") as f:
        file_data = f.read()
        

#files = {'file': ('APIs/M.mp3', open('APIs/M.mp3', 'rb'))}
url={"stablelm":"http://127.0.0.1:8000/stablelm","Mistral":"http://127.0.0.1:8000/Mistral","T-5":"http://127.0.0.1:8000/t5","tess":"http://127.0.0.1:8000/tess"}
request_data={"stablelm":{"prompt":"dog"},"Mistral":{"prompt":"who is president of india"},"T-5":{"text":"tell me about AI"},"tess":{"prompt":"what is the formula for CO2"}}
#response=requests.post(url,json=request_data)
##print(response.content)
url1="http://127.0.0.1:8000/whisper"
responses=requests.post(url1,files={"file":file_data})
whisper_transcription = responses.content
print("Whisper Transcription:", whisper_transcription)
for model, link in url.items():
    response = requests.post(link, params=request_data[model])
    if response.status_code == 200:
        print(response.content)
    else:
        print(f"Error {response.status_code}: {response.text}")