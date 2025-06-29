# AskWiki-AI
An interactive AI-powered app that answers your questions about Artificial Intelligence using Wikipedia and Retrieval-Augmented Generation (RAG). Built with Streamlit and Groq’s LLaMA 3 model, it delivers friendly, detailed explanations with real-world examples—perfect for students, learners, and AI enthusiasts!

---

## 🚀 How to Run This Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/wiki-rag.git
cd wiki-rag
```

### 2. Set up your environment

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file by copying the example:

```bash
cp .env.example .env
```

Edit the `.env` file and paste your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📦 File Structure

```
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── .env.example          # Sample environment variables file
├── README.md             # Project documentation
├── wiki_rag/             # (Optional) Cached vector index folder
├── .gitignore            # Prevents uploading sensitive files
```

---

## 🛡️ Note

Do **NOT** share your `.env` file or actual API key publicly.

