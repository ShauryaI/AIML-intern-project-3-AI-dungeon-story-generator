import streamlit as st
from transformers import pipeline

# 1. Load Story Model (BLOOM-560M) - Lightweight & Multilingual
@st.cache_resource
def load_story_model():
    return pipeline("text-generation", model="bigscience/bloom-560m", device=-1)


# 2. Load Translation Model (NLLB-200-Distilled-600M) - High Accuracy
@st.cache_resource
def load_translator():
    # NLLB is a dedicated translation model from Meta
    return pipeline("translation", model="facebook/nllb-200-distilled-600M", device=-1)


story_gen = load_story_model()
translator = load_translator()

# Session State for history
if 'story' not in st.session_state: st.session_state.story = ""

st.set_page_config(page_title="AI Story & Translator", layout="wide")
st.title("🏰 AI Dungeon: Professional Story & Translator")

# Sidebar Configuration
with st.sidebar:
    st.header("Story Setup")
    char = st.text_input("Hero Name:", "Arjun")
    genre = st.selectbox("Genre:", ["Fantasy", "Mystery", "Horror", "Adventure"])

    st.divider()
    if st.button("🗑️ Reset All"):
        st.session_state.story = ""
        st.rerun()

# Generation Logic
lang_starters = {
    "English": f"In a {genre.lower()} world, {char} was a brave warrior. One day, ",
    "Hindi": f"एक {genre} की दुनिया में, {char} एक बहादुर योद्धा था। एक दिन, ",
    "Bengali": f"একটি {genre} জগতে, {char} একজন সাহসী যোদ্ধা ছিলেন। একদিন, ",
}

prompt = st.session_state.story if st.session_state.story else lang_starters.get("English")
user_input = st.text_area("Ongoing Tale:", value=prompt, height=200)

if st.button("✨ Generate Continuation"):
    with st.spinner("Writing..."):
        out = story_gen(user_input, max_new_tokens=60, temperature=0.7, repetition_penalty=1.2, do_sample=True)
        st.session_state.story = out[0]['generated_text']
        st.rerun()

# Professional Translation Logic
if st.session_state.story:
    st.divider()
    st.subheader("🌐 Professional Translation")

    # NLLB requires specific language codes
    target_lang = st.selectbox("Translate to:", ["Hindi", "Bengali", "Marathi", "Urdu", "Tamil", "Telugu", "Malayalam"])
    lang_codes = {
        "Hindi": "hin_Deva", "Bengali": "ben_Beng", "Marathi": "mar_Deva",
        "Urdu": "urd_Arab", "Tamil": "tam_Taml", "Telugu": "tel_Telu", "Malayalam": "mal_Mlym"
    }

    if st.button("Translate Now"):
        with st.spinner("Translating using NLLB-200..."):
            # NLLB maps English (eng_Latn) to the target Indic code
            res = translator(st.session_state.story, src_lang="eng_Latn", tgt_lang=lang_codes[target_lang])
            st.success(f"**{target_lang} Translation:**\n\n{res[0]['translation_text']}")

    st.download_button("💾 Download Original Story", st.session_state.story, file_name="story.txt")
