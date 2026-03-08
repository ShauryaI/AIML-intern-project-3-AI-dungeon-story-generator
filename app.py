import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 1. Load pretrained mGPT from Hugging Face
@st.cache_resource

def load_multilingual_model():
    # mGPT supports 61 languages including Hindi, Spanish, French, etc.
    # model_name = "ai-forever/mGPT"
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Configure 4-bit quantization to save ~75% RAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Explicitly load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with quantization and automatic device placement
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically uses GPU if available, else CPU
        low_cpu_mem_usage=True  # Reduces RAM peaks during loading
    )
    return tokenizer, model

tokenizer, model = load_multilingual_model()

# Streamlit UI Setup

# Initialize Session State
if 'story_history' not in st.session_state:
    st.session_state.story_history = ""

def clear_story():
    st.session_state.story_history = ""

st.title("🏰Multilingual AI Dungeon Story Generator")
st.markdown("Create interactive stories using Generative AI.")

# Sidebar for Advanced Controls
with st.sidebar:
    st.header("Generation Settings")
    top_p = st.slider("Top-P (Nucleus Sampling)", 0.0, 1.0, 0.9,
                      help="Higher includes more 'long-tail' word choices for variety.")
    # Temperature: Higher = more creative/random, Lower = more focused
    temp = st.slider("Temperature (Creativity)", 0.1, 1.5, 0.7, help="Higher is more creative; lower is more focused.")
    # Top-K: Limits the word pool to the 'K' most likely next words
    tk = st.slider("Top-K (Word Pool)", 1, 100, 50, help="Higher allows more diverse word choices.")
    # Max Tokens: Controls the length of the generated snippet
    max_len = st.slider("Max Story Length", 50, 300, 100)
    num_gen = st.slider("Number of continuations to show", 1, 3, 2)

# 6. Add genre and language selection

col1, col2 = st.columns(2)
with col1:
    genre = st.selectbox("Genre:", ["Fantasy", "Mystery", "Sci-Fi", "Horror"])
with col2:
    # Users can now select the output language
    lang_map = {
        "Hindi": "एक",
        "Bengali": "একটি",
        "Marathi": "एक",
        "Urdu": "ایک",
        "Malayalam": "ഒരു",
        "Tamil": "ஒரு",
        "Telugu": "ఒక"
    }
    lang = st.selectbox("Language:", list(lang_map.keys()))

# Language-specific starters
starters = {
    "Hindi": f"एक {genre} की कहानी में, ",
    "Bengali": f"একটি {genre} গল্পে, ",
    "Marathi": f"एका {genre} कथेत, ",
    "Urdu": f"ایک {genre} کہانی میں، ",
    "Malayalam": f"ഒരു {genre} കഥയിൽ, ",
    "Tamil": f"ஒரு {genre} கதையில், ",
    "Telugu": f"ఒక {genre} కథలో, "
}

# 2. Build prompt-based input
# If history is empty, use the default starter
current_prompt = st.session_state.story_history if st.session_state.story_history else starters[lang]
user_prompt = st.text_area("Story Starter:", value=starters[lang])

col_gen, col_clear = st.columns([1, 1])

with col_gen:
    if st.button("✨ Generate Story"):
        with st.spinner("The AI is Writing..."):
            # 1. Encode the input text
            inputs = tokenizer(user_prompt, return_tensors="pt")

            # 2. Generate tokens
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_len,  # Generate 60 new words
                num_return_sequences=num_gen,
                temperature=temp,
                top_k = tk,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            # 3. Decode back to text
            st.session_state.story_history = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            st.rerun() # Refresh to show new text in the area

with col_clear:
    # Clear History Button
    st.button("🗑️ Clear History", on_click=clear_story)

# Display the ongoing story
if st.session_state.story_history:
    st.subheader("Your Tale so far:")
    st.info(st.session_state.story_history)

    # 5. Save story as text file
    # Using Streamlit's download button for easy file saving
    st.download_button(
        label="💾 Save Story as Text File",
        data=st.session_state.story_history,
        file_name=f"{lang}_story.txt",
        mime="text/plain"
    )

# 7. Deployment Instruction
# To run this, use: streamlit run app.py