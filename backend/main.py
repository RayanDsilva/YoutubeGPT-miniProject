from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import YoutubeLink

app = FastAPI()

origins = [
    ""
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*']
)

link = YoutubeLink()

#get the youtube link
@app.post('/get_summary')
async def summary_generator(video_url:str):
    link.video_url = video_url
    return link.get_summary()


@app.post('/get_query_response')
async def query_answering(query:str):
    if link.video_url == None:
        return "Enter your youtube link in the summarizer section first"
    else:
        return link.answer_query(query)
