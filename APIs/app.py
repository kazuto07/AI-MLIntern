from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from fastapi.responses import FileResponse
import librosa
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from vllm import LLM,SamplingParams
import os

app = FastAPI()




@app.get('/')
def index():
    return {'message': 'Go to  http://127.0.0.1:8000/docs/'}


@app.post("/whisper")
async def transcribe_audio(file: UploadFile = File(...)):
    model = WhisperModel("large-v2")
    #print(type(file.file))
    #print(file.file)
    audio_content,sr = librosa.load(file.file)
    transcription,info = model.transcribe(audio_content)
    print(transcription)
    strings_only = [entry[4] for entry in transcription]
    output_string = ' '.join(strings_only)

    return {"transcription": output_string}


@app.post('/stablelm',response_class=FileResponse)
async def TextToImage(prompt :str):
 model_id = "stabilityai/stable-diffusion-2"

 scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
 pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
 pipe = pipe.to("cuda")
 image = pipe(prompt).images[0]
    
 image.save("output.png")
 return "output.png" 

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

@app.post('/Mistral')
async def chat_with_mistral(prompt:str):
  model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
  tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
  model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    device_map="cuda:0")
  prompt_template=f'''<s>[INST] {prompt} [/INST]
  '''
  tokens = tokenizer(
  prompt_template,
  return_tensors='pt').input_ids.cuda()

  generation_params = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1
  }
  generation_output = model.generate(tokens,
    **generation_params)
  token_output = generation_output[0]
  text_output = tokenizer.decode(token_output)
  return ("model.generate output ", text_output)


@app.post('/t5')
def summarize_text(text: str):
    t5_summarizer = pipeline('summarization', model='t5-small', tokenizer='t5-small')
    summary = t5_summarizer(text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
    return {'model': 'T5', 'summary': summary}

@app.post("/tess")
async def Chat_with_tess(prompt: str):
   tokenizer = AutoTokenizer.from_pretrained("TheBloke/Tess-10.7B-v1.5b-AWQ")
   model2 = AutoModelForCausalLM.from_pretrained("TheBloke/Tess-10.7B-v1.5b-AWQ",low_cpu_mem_usage=True, device_map="cuda:0")
   prompt_template=f'''SYSTEM: {"Answer the queries"}USER: {prompt}ASSISTANT:'''

   generation_params = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1
}
   pipe = pipeline("text-generation",model=model2,tokenizer=tokenizer,**generation_params)

   pipe_output = pipe(prompt_template)[0]['generated_text']
   return ("pipeline output: ", pipe_output)