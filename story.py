import streamlit as st
from transformers import pipeline

# 1. Load pretrained GPT-2 from Hugging Face
@st.cache_resource
def load_model():
    # Using the pipeline API is the easiest way to load text generation models
    return pipeline("text-generation", model="gpt2")

generator = load_model()

# Streamlit UI Setup
st.title("🏰 AI Dungeon Story Generator")
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

# 6. Add genre selection
genre = st.selectbox("Select Genre:", ["Fantasy", "Mystery", "Sci-Fi", "Horror"])

# 2. Build prompt-based input
user_prompt = st.text_area("Enter your story starter:", value=f"In a {genre.lower()} world, a brave adventurer found...")

if st.button("Generate Story"):
    with st.spinner("The AI is writing..."):
        # 3. Generate story continuation & 4. Show multiple continuations
        # num_return_sequences allows generating multiple distinct versions
        # We must set do_sample=True to use temperature and top_k
        results = generator(
            user_prompt,
            max_length=max_len,
            num_return_sequences=num_gen,
            temperature=temp,
            top_k=tk,
            top_p=top_p,
            do_sample=True,
            # --- ANTI-REPETITION SETTINGS ---
            repetition_penalty=1.2,  # Penalizes words already used (1.0 to 1.5), Prevents getting stuck in loops
            no_repeat_ngram_size=3,  # Prevents any 3-word sequence from repeating
            truncation=True,
            pad_token_id=50256  # Standard GPT-2 EOS token
        )

        st.subheader("Generated Continuations:")
        full_story_text = ""
        for i, res in enumerate(results):
            story = res['generated_text']
            st.write(f"**Option {i + 1}:**")
            st.info(story)
            full_story_text += f"Option {i + 1}:\n{story}\n\n"

        # 5. Save story as text file
        # Using Streamlit's download button for easy file saving
        st.download_button(
            label="💾 Save Story as Text File",
            data=full_story_text,
            file_name="my_ai_story.txt",
            mime="text/plain"
        )

# 7. Deployment Instruction
# To run this, use: streamlit run app.py