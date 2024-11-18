# KalamBot Chat Application

KalamBot is a chatbot designed to answer questions related to Kalambot's operations, values, policies, and more. It utilizes a combination of LangChain, OpenAI's GPT models, and FAISS for conversational retrieval and contextual question answering.

---

## Setup Instructions

### 1. Clone the Repository
Clone the repository and navigate into its folder:
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install Dependencies
Create a virtual environment (optional) and install the required libraries:
```bash
python3 -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add environment variables
Create a .env file in the root directory and add your OpenAI API key:
```bash
OPENAI_API_KEY=<your_openai_api_key>
LANGCHAIN_API_KEY=<your_langchain_api_key>
```

### 4. Prepare the Knowledge Base
Ensure the Kalambot_Info.docx file is present in the root directory. This document serves as the knowledge base for the chatbot.

### 5. Run the Application
Launch the application using the following command:
```bash
uvicorn api:app --host 127.0.0.1 --port <port number>
```

## API Endpoints
### POST /chat
This endpoint processes user input and returns a chatbot response.


