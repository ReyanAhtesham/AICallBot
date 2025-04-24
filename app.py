import streamlit as st
from audiorecorder import audiorecorder
import assemblyai as aai
from langchain_core.documents import Document
import getpass
import os
from groq import Groq
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chat_models import init_chat_model
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# Instead of HuggingFacePipeline, call the pipeline directly
from transformers import pipeline


# Initialize the model
model_name = 'all-MiniLM-L6-v2' # assign the model name to a variable

# Create an embedding instance
embedding = SentenceTransformerEmbeddings(model_name=model_name) # use model_name keyword argument

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

os.environ["GROQ_API_KEY"] = "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"

if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")


model = init_chat_model("llama3-8b-8192", model_provider="groq")

client = Groq()


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,  # Increased chunk size
    chunk_overlap=50,  # Reduced chunk overlap
    length_function=len,
    is_separator_regex=False,

)


st.title("Audio Recorder")
audio = audiorecorder("Click to record", "Click to stop recording")
if len(audio) > 0:
    st.audio(audio.export().read())
    audio.export("audio.mp3", format="mp3")
    st.write(f"Frame rate: {audio.frame_rate}, Duration: {audio.duration_seconds} seconds")

# Set AssemblyAI API key (get from assembly.ai after signup)
aai.settings.api_key = "cce30862c26442f09e577fbe639108ed"

# Configure transcription with diarization
config = aai.TranscriptionConfig(speaker_labels=True)
transcriber = aai.Transcriber()

# Transcribe audio file (local or URL)
audio_file = "audio.mp3"  # Replace with your audio file
transcript = transcriber.transcribe(audio_file, config=config)

st.write(transcript.utterances[0].text)

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

# Create a LangChain prompt template (optional, you can skip this and call pipeline directly)
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Classify the sentiment of the following text: {text}"
)

# Create a function to use the pipeline for prediction
def predict_sentiment(text):
    result = sentiment_pipeline(text)[0]  # Get the prediction from the pipeline
    return result['label']  # Extract the sentiment label

# Example usage
text = transcript.utterances[0].text
# Instead of llm_chain.run, call predict_sentiment
result = predict_sentiment(text)
st.write(result)
query = transcript.utterances[0].text
results = vector_store.similarity_search(query, k=20)
context=" "
for r in results:
    context=context+r.page_content
#    print(f"Match: {r.page_content}")
ans=model.invoke(" Answer the question based on the following context if no context provided answer based on own intelligence:"+ context+"Answer the question based on the above context:"+query)
st.write(ans.content)
