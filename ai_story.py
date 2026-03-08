import streamlit as st
from transformers import pipeline
import torch


# Load the lightweight Multilingual BLOOM model
@st.cache_resource
def load_story_model():
    model_name = "bigscience/bloom-560m"
    # device_map="auto" handles CPU/GPU placement automatically on HF Spaces
    # Removing device_map="auto" and low_cpu_mem_usage fixes the Meta Tensor error
    return pipeline("text-generation",
                    model=model_name,
                    device= -1  # Forces CPU usage (standard for HF Free Tier)
            )

generator = load_story_model()

# Initialize Session State for Story History
if 'story_history' not in st.session_state:
    st.session_state.story_history = ""

st.set_page_config(page_title="Indic AI Dungeon", page_icon="🏰")
st.title("🏰 Indic AI Dungeon (Lite)")
st.markdown("Create interactive stories in Indic languages using BLOOM-560M.")

# --- UI Sidebar & Configuration ---
with st.sidebar:
    st.header("Settings")
    char_name = st.text_input("Hero's Name:", "Arjun")
    genre = st.selectbox("Genre:", ["Fantasy", "Mystery", "Horror", "Adventure"])

    lang_map = {
        "Hindi": "Hindi", "Bengali": "Bengali", "Marathi": "Marathi",
        "Urdu": "Urdu", "Malayalam": "Malayalam", "Tamil": "Tamil", "Telugu": "Telugu"
    }
    selected_lang = st.selectbox("Language:", list(lang_map.keys()))

    # Advanced Parameters
    temp = st.slider("Creativity (Temperature)", 0.5, 1.5, 0.7)
    max_tokens = st.slider("Words to Generate", 20, 100, 50)

# Language-specific starters
starters = {
    "Hindi": f"{char_name} एक बहादुर योद्धा था। {genre} की दुनिया में, ",
    "Bengali": f"{char_name} একজন সাহসী যোদ্ধা ছিলেন। {genre} জগতে, ",
    "Marathi": f"{char_name} एक शूर योद्धा होता। {genre} जगात, ",
    "Urdu": f"{char_name} ایک بہادر جنگجو تھا۔ {genre} کی دنیا میں، ",
    "Malayalam": f"{char_name} ഒരു ധീര യോദ്ധാവായിരുന്നു. {genre} ലോകത്ത്, ",
    "Tamil": f"{char_name} ஒரு வீரர். {genre} உலகில், ",
    "Telugu": f"{char_name} ఒక ధైర్యవంతుడైన యోధుడు. {genre} ప్రపంచంలో, "
}

# --- Main Story Engine ---
current_prompt = st.session_state.story_history if st.session_state.story_history else starters[selected_lang]
user_input = st.text_area("Your Story Console:", value=current_prompt, height=250)

col1, col2 = st.columns(2)

with col1:
    if st.button("✨ Generate Next Part"):
        with st.spinner("The AI is writing..."):
            # Generate continuation
            results = generator(
                user_input,
                max_new_tokens=max_tokens,
                temperature=0.7,  # Lowered from 1.0+ to prevent "hallucination"
                top_p=0.92,  # Nucleus sampling: only picks from 92% most likely words
                top_k=50,  # Limits the word pool to the best 50 choices
                do_sample=True,
                # --- ANTI-REPETITION SETTINGS ---
                repetition_penalty=1.2,  # Penalizes words already used (1.0 to 1.5), Prevents getting stuck in loops
                no_repeat_ngram_size=3,  # Prevents any 3-word sequence from repeating
                truncation=True,
                pad_token_id=3
            )
            st.session_state.story_history = results[0]['generated_text']
            st.rerun()

with col2:
    if st.button("🗑️ Reset Story"):
        st.session_state.story_history = ""
        st.rerun()

# --- Output & Saving ---
if st.session_state.story_history:
    st.subheader("Your Tale so far:")
    st.info(st.session_state.story_history)

    st.download_button(
        label="💾 Save Story as Text File",
        data=st.session_state.story_history,
        file_name=f"{char_name}_story.txt",
        mime="text/plain"
    )