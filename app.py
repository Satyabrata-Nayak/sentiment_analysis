import streamlit as st

import pandas as pd
import numpy as np
import altair as alt
import joblib

model = joblib.load(open('text_emotion.pkl','rb'))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}
def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions in Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label="Submit")

        if submit_text:
            col1,col2 = st.columns(2)

            prediction = model.predict([raw_text])[0]
            probability = model.predict_proba([raw_text])
            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write(f"{prediction}:{emoji_icon}")
                st.write(f"Confidence:{np.max(probability)}")
            with col2:
                st.success("Prediction Probability")
                # st.write(probability)
                proba_df = pd.DataFrame(probability, columns=model.classes_)
                # st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()



