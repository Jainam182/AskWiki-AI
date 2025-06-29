import os
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings

load_dotenv()

# HuggingFace embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

INDEX_DIR = 'wiki_rag'

PAGES = [
    "Artificial intelligence", "History of artificial intelligence", "Machine learning",
    "Neural network", "Deep learning", "Convolutional neural network",
    "Reinforcement learning", "Natural language processing", "ChatGPT",
    "Transformer (machine learning model)", "Large language model", "Generative AI",
    "Recommender system", "Artificial general intelligence", "Artificial superintelligence"
]

@st.cache_resource
def get_index():
    if os.path.isdir(INDEX_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage_context)

    reader = WikipediaReader()
    docs = []
    for page in PAGES:
        try:
            result = reader.load_data(pages=[page], auto_suggest=True)
            docs.extend(result)
        except Exception:
            pass

    llm = OpenAI(
        model="llama3-70b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

    index = VectorStoreIndex.from_documents(docs, embed_model="local")
    index.storage_context.persist(persist_dir=INDEX_DIR)
    return index

@st.cache_resource
def get_query_engine():
    index = get_index()
    llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
    return index.as_query_engine(llm=llm, similarity_top_k=3)

def classify_topic(answer_text):
    answer_text = answer_text.lower()
    if "convolutional neural network" in answer_text or "cnn" in answer_text:
        return "cnn"
    elif "transformer" in answer_text or "gpt" in answer_text or "generative" in answer_text:
        return "gpt"
    elif "reinforcement learning" in answer_text:
        return "reinforcement"
    elif "natural language processing" in answer_text or "nlp" in answer_text:
        return "nlp"
    else:
        return "generic"

def generate_extras(topic_key):
    examples = {
        "cnn": """### ✨ Real-life example:
A **Convolutional Neural Network (CNN)** is like a robot looking at pictures to find patterns like cats or cars. Just like your brain spots your friend in a photo, CNNs recognize shapes.

### 💡 Fun Fact:
Apps like **Google Photos** or **iPhone Face ID** use CNNs to group photos or unlock phones!

### 🚀 Want to go even deeper?
Scroll down to read the original source 👇
""",
        "gpt": """### ✨ Real-life example:
Chatting with ChatGPT feels natural, right? That's GPT! It learns from tons of text to answer like a real person.

### 💡 Fun Fact:
**ChatGPT**, the app you're using now, is a GPT model!

### 🚀 Want to go even deeper?
Scroll down to read the original source 👇
""",
        "reinforcement": """### ✨ Real-life example:
Teaching a dog tricks with treats? That’s **Reinforcement Learning** — the computer tries, learns, and improves!

### 💡 Fun Fact:
**Self-driving cars** use it to figure out how to drive safely.

### 🚀 Want to go even deeper?
Scroll down to read the original source 👇
""",
        "nlp": """### ✨ Real-life example:
Used **Google Translate** or **autocorrect**? That’s **NLP** understanding your language.

### 💡 Fun Fact:
**Alexa** or **Siri** understand your voice using NLP.

### 🚀 Want to go even deeper?
Scroll down to read the original source 👇
""",
        "generic": """### ✨ Real-life example:
Think of AI like a smart student breaking big ideas into chunks to learn — just like you!

### 💡 Fun Fact:
**Netflix and YouTube** recommend videos based on your likes using AI.

### 🚀 Want to go even deeper?
Scroll down to read the original source 👇
"""
    }
    return examples.get(topic_key, examples["generic"])

def main():
    st.set_page_config(page_title="Wikipedia RAG with Groq", layout="centered")
    st.title("📚 Wikipedia RAG using Groq")
    st.markdown("Ask anything about **Artificial Intelligence** using Wikipedia-powered RAG.")

    if "response" not in st.session_state:
        st.session_state.response = None
        st.session_state.question = ""

    question = st.text_input("Enter your question:", placeholder="What is AI?")

    if st.button("🔍 Submit") and question:
        with st.spinner("🤔 Thinking..."):
            query_engine = get_query_engine()
            response = query_engine.query(question)
        st.session_state.response = response
        st.session_state.question = question

    if st.session_state.response:
        response = st.session_state.response
        st.subheader("✅ Here's what I found – in simple words")

        topic = classify_topic(response.response)
        st.markdown(response.response)
        st.markdown(generate_extras(topic))

        st.subheader("📄 Wikipedia Page Sources")

        dropdown_options = []
        source_mapping = {}
        pages_lower = {p.lower().replace(" ", "_"): p for p in PAGES}

        for i, src in enumerate(response.source_nodes):
            metadata = src.node.metadata
            wiki_key = metadata.get("page", "").lower().replace(" ", "_")
            display_name = pages_lower.get(wiki_key, f"Page {i+1}")

            dropdown_options.append(display_name)
            source_mapping[display_name] = {
                "content": src.node.get_content(),
                "page_title": display_name,
                "wiki_url": f"https://en.wikipedia.org/wiki/{wiki_key}" if wiki_key else None,
                "metadata": metadata
            }

        selected_title = st.selectbox("Select a Wikipedia page to view its content:", dropdown_options)

        if selected_title:
            data = source_mapping[selected_title]
            st.markdown(f"### 📘 {data['page_title']}")
            st.markdown(f"> {data['content']}")
            if data["wiki_url"]:
                st.markdown(f"🌐 [View on Wikipedia]({data['wiki_url']})")
            with st.expander("🔍 Show Metadata"):
                st.json(data["metadata"])

if __name__ == "__main__":
    main()