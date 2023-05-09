import gradio as gr
import openai
import subprocess
import gtts
from config import OPENAI_API_KEY

openai.api_key =OPENAI_API_KEY

messages = [
    {"role": "system", "content": "You are an active listener. Respond as if you were a rapper Jay-z."}]

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

    tts = gtts.gTTS(assistant_message, lang='en')
    tts.save("assistant_message.mp3")
    subprocess.call(["afplay", "assistant_message.mp3"])
    messages.append({"role": "assistant", "content": assistant_message})

    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return chat_transcript

ui = gr.Interface(fn=transcribe, inputs=gr.Audio(
    source="microphone", type="filepath"), outputs="text")

ui.launch()
