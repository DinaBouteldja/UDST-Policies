import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from mistralai import Mistral
import os
from dotenv import load_dotenv
import time
import re

# Load environment variables from .env (for local use)
load_dotenv()

# Check GitHub Secrets first, then fallback to .env
api_key = os.getenv("MISTRAL_API_KEY")

# Debugging: Ensure API key is loaded
if not api_key:
    print("Error: API key not found! Make sure to set it as a GitHub Secret or in .env")
else:
    print(f"API Key Loaded: {api_key[:5]}*****")   # Partially print key for security

# Define the policies and their URLs
policies = {
    "Admissions Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/admissions-procedure",
    "Student Conduct": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy",
    "Grading System": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
    "Attendance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
    "Scholarship Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance",
    "Sports Facilities": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and",
    "Examination Ploicy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/examination-policy",
    "Library Usage": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/use-library-space-policy",
    "Library Study Room Booking Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/library-study-room-booking-procedure",
    "International Student Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-procedure"
}

# Streamlit UI
st.title("UDST Policies Chatbot")
st.write("Select a policy and ask your question about it.")

# Select policy from list
selected_policy = st.selectbox("Choose a policy:", list(policies.keys()))

# Input query
query = st.text_input("Enter your question:")

# Load and process selected policy
@st.cache_data
def get_policy_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    tag = soup.find("div")
    return tag.get_text() if tag else "No content found"

policy_text = get_policy_text(policies[selected_policy])
# Extract text and clean it
def clean_text(text):
    text = re.sub(r"\n+", "\n", text)  # Remove excessive newlines
    text = re.sub(r"\s{2,}", " ", text)  # Remove excessive spaces
    return text.strip()

policy_text = clean_text(policy_text)

# Chunking text
chunk_size = 512
chunks = [policy_text[i:i+chunk_size] for i in range(0, len(policy_text), chunk_size)]

# Get embeddings
def get_text_embedding(text_list):
    client = Mistral(api_key=api_key)
    response = client.embeddings.create(model="mistral-embed", inputs=text_list)
    time.sleep(5)  # Wait 5 seconds before the next request
    return np.array([emb.embedding for emb in response.data])

embeddings = get_text_embedding(chunks)

# Store in FAISS
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Handle query
if query:
    query_embedding = get_text_embedding([query])
    D, I = index.search(query_embedding, k=2)
    retrieved_chunks = [chunks[i] for i in I[0]]
    print("Retrieved Chunks:", retrieved_chunks)
    
    # Generate answer using Mistral
    def generate_answer(context, question):
        prompt = f"""
        Context information:
        ---------------------
        {context}
        ---------------------
        Based on the above, answer the question:
        {question}
        """
        client = Mistral(api_key=api_key)
        response = client.chat.complete(model="mistral-large-latest", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content

    answer = generate_answer(retrieved_chunks, query)

    # Display answer
    st.text_area("Answer:", answer, height=150)
