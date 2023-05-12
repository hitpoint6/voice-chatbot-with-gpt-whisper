import gradio as gr
import openai
import subprocess
from config import OPENAI_API_KEY, ELEVEN_LABS_API_KEY, SAMATHA_VOICE
import requests

openai.api_key = OPENAI_API_KEY

messages = [
    {"role": "system", "content": "You are an active listener."}]

def assistant_speak(content):
        # text to speech request with eleven labs
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{SAMATHA_VOICE}/stream"
    data = {
        "text": content,
        "voice_settings": {
            "stability": 0.1,
            "similarity_boost": 0.8
        }
    }
    r = requests.post(url, headers={'xi-api-key': ELEVEN_LABS_API_KEY}, json=data)

    output_filename = "assistant_message.mp3"
    with open(output_filename, "wb") as output:
        output.write(r.content)
    
    subprocess.call(["afplay", "assistant_message.mp3"])


def transcribe(audio):
    global messages

    audio_file = open(audio, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    messages.append({"role": "user", "content": transcript["text"]})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    assistant_message = response["choices"][0]["message"]["content"]

    assistant_speak(assistant_message)
    messages.append({"role": "assistant", "content": assistant_message})

    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return chat_transcript

ui = gr.Interface(fn=transcribe, inputs=gr.Audio(
    source="microphone", type="filepath"), outputs="text")

ui.launch(debug=True)
