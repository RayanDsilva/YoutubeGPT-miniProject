import requests
from bs4 import BeautifulSoup
import gradio as gr
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceHubEmbeddings()

llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"max_new_tokens":1028,
                                                           "temperature":0.7,
                                                           "top_k" : 50,
                                                           "top_p" : 0.95}
)
summary_template = """<|system|>
You are a youtube video summarization bot. You need to give a detailed summary on the given youtube transcript
Transcript will include a video title, you need to include that in the beginning of the summary
You can include additional content to the summary by refering the title.
Keep the summary professional in bullet points and also preserve the context of the transcript.
Let your response be:
Title:
Summary:</s>
<|user|>
{text}</s>
<|Assistant|>"""

summary_prompt = PromptTemplate(template=summary_template, input_variables=["text"])

query_prompt = PromptTemplate(
      input_variables = ['question', 'docs'],
      template = """
<|system|>
You are a helpful Youtube assistant that can answer questions about videos based on the videos's transcript.
Only use the factual information from the transcript to answer the question.
If you feel like you don't have enough information to answer the question, say "I don't know"
</s>
<|user|>
Answer the following question in points along with context:{question}
By searching the following video transcript:{docs}</s>
<|Assistant|>
"""
)

query_chain = LLMChain(llm = llm, prompt = query_prompt)


def transcript_generator(video_url:str):

    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap =40)
    docs = text_splitter.split_documents(transcript)
    return docs

def summary_transcript(video_url, docs):
    response = requests.get(video_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title_element = soup.find('meta', property='og:title')
    video_title = title_element['content']
    text = []
    text.append(f"title:{video_title}")
    for d in docs[:6] + docs[-6:]:
        text.append(d.page_content)

    return text

def summary_fn(video_url, transcript_docs):
    text = summary_transcript(video_url, transcript_docs)
    summary_chain = LLMChain(prompt=summary_prompt, llm=llm)
    return summary_chain.run(text)

def QA_fn(query):
    docs = db.similarity_search(query, k=4)
    context = " ".join([d.page_content for d in docs])
    response = query_chain.run(question = query, docs = context)
    return response

def input_text(query,history):
    if "https://" in query:
        global db
        transcript_docs = transcript_generator(query)
        db = Chroma.from_documents(transcript_docs, embeddings)
        return summary_fn(query,transcript_docs)
    else:
        return QA_fn(query)

demo = gr.ChatInterface(fn=input_text, examples=["https://www.youtube.com/watch?v=r4wLXNydzeY"], title="Bot")
demo.launch()