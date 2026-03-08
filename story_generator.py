import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

# Create a folder for memory offloading to prevent RAM crashes
if not os.path.exists("offload"):
    os.makedirs("offload")


@st.cache_resource
def load_mgpt_8bit():
    model_name = "ai-forever/mGPT"

    # 8-bit config is essential for low-memory environments
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        offload_folder="offload",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    return tokenizer, model


# App Interface
st.set_page_config(page_title="Multilingual AI Dungeon", layout="wide")
st.title("🏰 mGPT Multilingual Story Generator")

try:
    tokenizer, model = load_mgpt_8bit()
except Exception as e:
    st.error(f"Model loading failed: {e}. Try refreshing or using a smaller model.")
    st.stop()

# --- Session State for Story ---
if 'story' not in st.session_state:
    st.session_state.story = ""

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Adventure Settings")
    char = st.text_input("Hero Name:", "Arjun")
    genre = st.selectbox("Genre:", ["Fantasy", "Mystery", "Horror"])
    lang = st.selectbox("Language:", ["English", "Hindi", "Bengali", "Marathi", "Urdu", "Tamil", "Telugu"])

    if st.button("🗑️ Reset Story"):
        st.session_state.story = ""
        st.rerun()

# --- Generation Logic ---
starters = {
    "English": f"In a {genre} world, {char} found a mysterious",
    "Hindi": f"एक {genre} की दुनिया में, {char} को एक रहस्यमय"
}
prompt = st.session_state.story if st.session_state.story else starters.get(lang, starters["English"])
user_input = st.text_area("Your Story:", value=prompt, height=200)

if st.button("✨ Write Next Chapter"):
    with st.spinner("Generating..."):
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            repetition_penalty=1.2,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        st.session_state.story = tokenizer.decode(output[0], skip_special_tokens=True)
        st.rerun()

# --- Translation Logic ---
if st.session_state.story:
    st.info(st.session_state.story)
    target_lang = st.selectbox("Translate to:", ["Hindi", "Bengali", "Marathi", "Urdu", "Tamil", "Telugu"])

    if st.button("🌐 Translate"):
        trans_prompt = f"English: {st.session_state.story}\n{target_lang}:"
        inputs = tokenizer(trans_prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=100)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        st.success(result.split(f"{target_lang}:")[-1])
