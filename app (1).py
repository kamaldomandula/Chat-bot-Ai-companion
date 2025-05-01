import os
import time
import openai
import gradio as gr
from openai import OpenAI
from gtts import gTTS
import tempfile

# === Load OpenAI API Key ===
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
client = OpenAI(api_key=api_key)

# === AI Personality ===
persona = """
You are a kind, caring, and emotionally intelligent AI companion.
You speak warmly and naturally, like a close friend who listens well and gives thoughtful, encouraging replies.
You avoid sounding robotic or repetitive. Instead, you keep it real and genuine.
If someone sounds down, you comfort them. If they’re excited, you celebrate with them.
You can offer motivation, ideas, jokes, or just talk about life.
"""

# === Generate TTS audio from reply ===
def generate_tts(text):
    if not text:
        return None
    tts = gTTS(text)
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_path.name)
    return tmp_path.name

# === Chat response logic with typing effect ===
def respond(message, history, tts_enabled):
    messages = [{"role": "system", "content": persona}] + history
    messages.append({"role": "user", "content": message})
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        full_reply = response.choices[0].message.content
        displayed_reply = ""

        updated_history = history + [{"role": "user", "content": message}]
        for char in full_reply:
            displayed_reply += char
            time.sleep(0.01)
        updated_history.append({"role": "assistant", "content": displayed_reply})
        return updated_history, updated_history, displayed_reply if tts_enabled else None
    except Exception as e:
        error = f"❌ Error: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error})
        return history, history, None

# === Gradio UI Layout ===
with gr.Blocks(css="""
body {
    background: linear-gradient(135deg, #0f0f0f, #1e1e2f);
    color: #f8f8f8;
    font-family: 'Segoe UI', sans-serif;
}
#chatbot {
    background: rgba(30, 30, 45, 0.8);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.05);
    max-width: 800px;
    margin: auto;
}
.message.user {
    background-color: #333 !important;
    color: #fff !important;
    border-radius: 15px;
    padding: 10px;
}
.message.bot {
    background-color: #444 !important;
    color: #ddd !important;
    border-radius: 15px;
    padding: 10px;
}
textarea {
    background-color: #121212 !important;
    border: 1px solid #333 !important;
    color: #fff !important;
    font-size: 16px;
    padding: 10px;
}
h1, p {
    text-shadow: 1px 1px 2px #000;
}
""") as demo:

    # Header
    gr.Markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <h1 style="color: #4fc3f7; font-size: 40px;">💬 Your AI Companion</h1>
            <p style="color: #bbbbbb;">Always here to talk, support, and uplift you 🌟</p>
        </div>
    """)

    # Chatbot + Input + State
    chatbot = gr.Chatbot(elem_id="chatbot", type="messages")
    msg = gr.Textbox(placeholder="Say something...", show_label=False)
    state = gr.State([])

    # Voice Output
    tts_toggle = gr.Checkbox(label="🔊 Speak AI replies", value=False)
    tts_hidden = gr.Textbox(visible=False)
    audio_out = gr.Audio(label="🔈 AI Voice", interactive=False)

    # Voice Input (microphone)
    mic = gr.Audio(label="🎙️ Voice Input (Optional)", type="filepath", interactive=True)
    
    def transcribe(audio_path):
        if not audio_path:
            return ""
        with open(audio_path, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)
        return transcript["text"]
    
    mic.change(transcribe, mic, msg)

    # Submit flow
    msg.submit(respond, [msg, state, tts_toggle], [chatbot, state, tts_hidden]).then(
        generate_tts, tts_hidden, audio_out
    )

demo.launch()
