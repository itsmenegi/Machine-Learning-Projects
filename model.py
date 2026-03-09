import streamlit as st

import pandas as pd
import numpy as np
import altair as alt

import joblib

pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮",
    "Excited": "🤩",
    "Content": "😌",
    "Angry": "😡",
    "Frustrated": "😤",
    "Confused": "😕",
    "Bored": "😒",
    "Relaxed": "😎",
    "Anxious": "😰",
    "Calm": "🧘",
    "Love": "❤️",
    "Disgust": "🤢",
    "Shy": "😊",
    "Embarrassed": "😳",
    "Curious": "🤔",
    "Jealous": "😒",
    "Grateful": "🙏"
}


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def main():
    # Initialize history in session state
    if 'history' not in st.session_state:
        st.session_state['history'] = []
        
    st.title("Mood_Mate")
    st.subheader("Detect Emotions In Text")
    st.write("Welcome to the moodmate!")
    # --- Recent History in Sidebar ---
    st.sidebar.title("Recent History")
    
    # Display the last 3 entries
    recent_entries = st.session_state['history'][-3:][::-1]
    
    if recent_entries:
        st.sidebar.markdown("**Recent Texts**:")
        for i, text in enumerate(recent_entries):
            # Display a short snippet and the full text in an expander
            with st.sidebar.expander(f"Entry {len(recent_entries) - i}"):
                st.caption(text)
    else:
        st.sidebar.info("No recent entries yet.")
        
    # Full history button
    if st.session_state['history']:
        if st.sidebar.button("Show Full History"):
             st.sidebar.markdown("---")
             st.sidebar.subheader("Full History")
             for entry in st.session_state['history'][::-1]:
                 st.sidebar.text(entry[:50] + "...") # Show first 50 chars
                 st.sidebar.markdown("---")
  
    with st.form(key='my_form'):
        raw_text = st.text_area("Enter Your Text Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text and raw_text:
        st.session_state['history'].append(raw_text)
        
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)
        
        with col1:
            st.success("Your Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "❓")
            st.write(f"{prediction}: {emoji_icon}")
            st.write(f"Confidence: {np.max(probability):.2f}")

        with col2:
            st.success("Prediction Probability")
            
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x=alt.X('emotions', sort='-y'), # Sort by probability descending
                y='probability', 
                color='emotions'
            ).properties(height=300)
            
            st.altair_chart(fig, use_container_width=True)
            
        st.subheader("Thank you!")

if __name__ == '__main__':
    main()

    
