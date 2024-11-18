from fastapi import FastAPI, Depends
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['USER_AGENT'] = 'myagent'
# Initialize FastAPI app
app = FastAPI()

# Set up shared model and necessary components
model = ChatOpenAI(model="gpt-4o-mini")
loader = Docx2txtLoader("Kalambot_Info.docx")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, add_start_index=True)
all_splits = []
for doc in documents:
    splits = text_splitter.split_documents([doc])
    all_splits.extend(splits)

vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
message_history = ChatMessageHistory()
chat_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    chat_memory=message_history,
    k=20
)
rag_template = """
You are Kalambot, the chatbot for Kalambot, here to assist with questions strictly related to the company — its values, operations, policies, and other relevant information. Interpret any references to "you" or "your" as referring to Kalambot, the company.

Procedure:
1. Review the provided chat history.
2. Identify the parts of the chat history that are most relevant to the current question.
3. Search for and extract any relevant documents related to the question (e.g., technologies used).
4. If no relevant documents are found, use the chat history to answer the question.
5. If the question is unrelated to Kalambot (e.g., questions about fine-tuning/finetuning, personal advice, or unrelated topics), return an empty string.
6. Keep responses concise, using a maximum of three sentences.

Question: {question}
Context: {context}

Response:
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)

# Define the Conversational Retrieval Chain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=chat_memory,
    chain_type="stuff",
    get_chat_history=lambda h: h,
    combine_docs_chain_kwargs={"prompt": rag_prompt},
)

generic_template = """
Use the following pieces of context to answer the question at the end. You are a chatbot designed to answer questions both from a provided context and, if applicable, general knowledge. For customer-related questions, use only the relevant information from the context. If there is no context available for the question, provide a general answer if you can. 

Procedure:
1. Review the provided contextual chat history.
2. Identify relevant information from the chat history related to the current question.
3. Extract and use only relevant context to answer the question.
4. If the question cannot be answered from the context but is general knowledge, provide a general answer. If the question is outlandish, unethical, or irrelevant, return an empty string.

Context: {chat_history}
Question: {question}
Helpful Answer:
"""



generic_prompt = ChatPromptTemplate.from_template(generic_template)

base_chain = LLMChain(
    prompt=generic_prompt,
    llm=model,
    memory=chat_memory,
    output_parser=StrOutputParser()
)


check_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are KalamBot, an intelligent assistant tasked with determining if the user's current question is a follow-up to the same subject as the previous response. Follow these steps to decide:\n\n"
         "1. **Check if Previous Response Contains 'I don't know'**: If the previous response ('{previous_answer}') contains 'I don't know,' respond immediately with 'No Match.'\n\n"
         "2. **Identify the Main Subject**: Identify the main subject or entity in the previous response ('{previous_answer}').\n\n"
         "3. **Analyze the Current Question**: Check if the current question ('{text}') addresses the same subject. Look for indications that the question refers back to the same entity, such as references to 'you,' 'the company,' or other specific terms linked to the identified subject.\n\n"
         "4. **Determine Match or No Match**: If the current question logically follows from the previous response and both refer to the same subject or entity, categorize as 'match.' Otherwise, categorize as 'No Match'.\n\n"
         "Respond with one of the following:\n"
         "- 'match' if the question clearly follows from and refers to the same entity or concept as the previous response.\n"
         "- 'No Match' if the question is unrelated to the previous response or if the previous response contains 'I don't know.'"
        ),
        ("user", "Follow-up question: {text}. Previous answer: {previous_answer}.")
    ]
) # Add your check prompt messages here
check_prompt_chain = check_prompt_template | model | StrOutputParser()

# App state to maintain session info
app.state.chain_type = "rag"
app.state.previous_answer = ""


from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Request model for POST payload
class ChatRequest(BaseModel):
    user_input: str

# App state to maintain session info
app.state.chain_type = "rag"
app.state.previous_answer = ""

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.user_input

    # Prepare combined input for matching logic
    combined_input = f"Previous Response: {app.state.previous_answer} Current Question: {user_input}"
    
    # Determine if the new input matches the previous response context
    is_match = check_prompt_chain.invoke({
        "previous_answer": app.state.previous_answer,
        "text": user_input
    })

    if is_match == "match":
        # Determine the appropriate chain based on previous chain type
        if app.state.chain_type == "base":
            response = base_chain.invoke(user_input)["text"]
        else:
            response = rag_chain.invoke(user_input)["answer"]
    else:
        response = rag_chain.invoke(user_input)["answer"]

        # Switch to base_chain if rag_chain yields no response
        if not response:
            response = base_chain.invoke(user_input)["text"]
            app.state.chain_type = "base"
        else:
            app.state.chain_type = "rag"

    # Update the session state for next interaction
    app.state.previous_answer = response

    return {"response": response}
