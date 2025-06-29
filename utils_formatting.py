
# utils/formatting.py

import streamlit as st

def styled_answer(text: str) -> None:
    st.markdown(f"<div style='font-size: 16px; line-height: 1.6;'>{text}</div>", unsafe_allow_html=True)

def explain_like_a_friend(response: str) -> None:
    st.markdown(
        f"""
        ### ✅ Here's what I found – in simple words

        👋 Hey there! Let's break it down together.

        {response}

        ---

        ✨ **Real-life example:**<br>
        Think of a neural network like how you learn what a dog looks like by seeing many pictures of dogs.
        
        💡 **Fun Fact:**<br>
        Apps like Google Photos use AI to detect your face. That's a neural network working!

        🚀 Want to learn more? See the source below 👇
        """,
        unsafe_allow_html=True
    )
