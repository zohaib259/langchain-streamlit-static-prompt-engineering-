from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

st.header("Research tool")

paper_input = st.selectbox(
    "Select research paper name",
    [
        "Attention Is All You Need (Transformers)",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "LLaMA: Open and Efficient Foundation Language Models",
        "Deep Residual Learning for Image Recognition (ResNet)",
        "Retrieval-Augmented Generation (RAG)",
        "Mastering the Game of Go (AlphaGo)",
        "LoRA: Low-Rank Adaptation of Large Language Models",
    ],
)

style_input = st.selectbox(
    "Select respose style",
    ["Beginner friendly", "Technical", "Code-oriended", "Math style"],
)

lenght_input = st.selectbox(
    "Select text lenght",
    [
        "Short (1 paragraph)",
        "Medium (2-3 paragraphs)",
        "Detailed (Comprehensive summary with key takeaways)",
    ],
)

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

template = load_prompt("template.json")

if st.button("Submit"):

    chain = template | model
    with st.spinner("Analyzing..."):
        result = chain.invoke(
            {
                "paper_input": paper_input,
                "style_input": style_input,
                "length_input": lenght_input,
            }
        )
        st.write(result.text)
