import streamlit as st
from transformers import pipeline


# 1. Load the lightweight BLOOM model (Fits in ~1.2GB RAM)
@st.cache_resource
def load_bloom():
    # 'device=-1' ensures it runs on CPU, which is the default for Streamlit Cloud
    return pipeline("text-generation", model="bigscience/bloom-560m", device=-1)


generator = load_bloom()

# --- Session State for History ---
if 'story' not in st.session_state:
    st.session_state.story = ""

st.set_page_config(page_title="Indic AI Dungeon Lite", page_icon="🏰")
st.title("🏰 Indic AI Dungeon Lite")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Hero & World")
    char = st.text_input("Hero Name:", "Arjun")
    genre = st.selectbox("Genre:", ["Fantasy", "Mystery", "Horror", "Adventure"])
    lang = st.selectbox("Current Language:", ["English", "Hindi", "Bengali", "Marathi", "Urdu", "Tamil", "Telugu"])

    st.divider()
    if st.button("🗑️ Reset All"):
        st.session_state.story = ""
        st.rerun()

# Language Starters Mapping
starters = {
    "English": f"In a {genre.lower()} world, {char} was a brave warrior. One day, ",
    "Hindi": f"एक {genre} की दुनिया में, {char} एक बहादुर योद्धा था। एक दिन, ",
    "Bengali": f"একটি {genre} জগতে, {char} একজন সাহসী যোদ্ধা ছিলেন। একদিন, ",
    "Marathi": f"एका {genre} जगात, {char} एक शूर योद्धा होता। एके दिवशी, ",
    "Urdu": f"ایک {genre} کی دنیا میں، {char} ایک بہادر جنگجو تھا۔ ایک دن، ",
    "Malayalam": f"ഒരു {genre} ലോകത്ത്, {char} ഒരു ധീര യോദ്ധാവായിരുന്നു. ഒരു ദിവസം, ",
    "Tamil": f"ஒரு {genre} உலகில், {char} ஒரு வீரர். ஒரு நாள், ",
    "Telugu": f"ఒక {genre} ప్రపంచంలో, {char} ఒక ధైర్యవంతుడైన యోధుడు. ఒక రోజు, "
}

# --- Main Interaction ---
prompt = st.session_state.story if st.session_state.story else starters.get(lang, starters["English"])
user_input = st.text_area("The Story Console:", value=prompt, height=200)

if st.button("✨ Generate Next Chapter"):
    with st.spinner("Writing..."):
        # Anti-repetition and Logic settings
        output = generator(
            user_input,
            max_new_tokens=60,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=3  # BLOOM specific
        )
        st.session_state.story = output[0]['generated_text']
        st.rerun()

# --- Translation Feature ---
if st.session_state.story:
    st.subheader("Your Tale so far:")
    st.info(st.session_state.story)

    st.divider()
    st.subheader("🌐 Translate Story")
    target_lang = st.selectbox("Select Target Language:", ["Hindi", "Bengali", "Marathi", "Urdu", "Tamil", "Telugu"])

    if st.button("Translate Now"):
        # Few-Shot Prompting for translation logic
        trans_prompt = (
            f"English: The sun rose over the mountains.\n"
            f"{target_lang}: सूरज पहाड़ों के ऊपर उग आया।\n\n"
            f"English: {st.session_state.story}\n"
            f"{target_lang}:"
        )
        with st.spinner("Translating..."):
            trans_out = generator(trans_prompt, max_new_tokens=100, do_sample=False)
            result = trans_out[0]['generated_text'].split(f"{target_lang}:")[-1].strip()
            st.success(f"**Translation ({target_lang}):**\n\n{result}")

    # --- Save Action ---
    st.download_button("💾 Download Story", st.session_state.story, file_name="ai_story.txt")
