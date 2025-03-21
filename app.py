import time
import threading
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from tts import TextToSpeechService
import torch

console = Console()

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console.print(f"[cyan]Using device: {device}")

# Load Whisper model on the appropriate device
stt = whisper.load_model("base.en", device="cuda" if torch.cuda.is_available() else "cpu")

# Initialize TTS service
tts = TextToSpeechService()

# Define the conversation prompt template
template = """
You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less 
than 20 words.

The conversation transcript is as follows:
{history}

And here is the user's follow-up: {input}

Your response:
"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=Ollama(),
)

def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)

def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.
    """
    try:
        # Normalize and move audio data to the appropriate device
        audio_tensor = torch.tensor(audio_np, dtype=torch.float32, device=device)
        result = stt.transcribe(audio_tensor, fp16=torch.cuda.is_available())
        text = result["text"].strip()
        return text
    except Exception as e:
        console.print(f"[red]Transcription error: {e}")
        return ""

def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the Llama-2 language model.
    """
    try:
        response = chain.predict(input=text)
        if response.startswith("Assistant:"):
            response = response[len("Assistant:") :].strip()
        return response
    except Exception as e:
        console.print(f"[red]LLM error: {e}")
        return "Sorry, I encountered an error generating a response."

def play_audio(sample_rate, audio_array):
    """
    Plays the given audio data using the sounddevice library.
    """
    try:
        sd.play(audio_array, sample_rate)
        sd.wait()
    except Exception as e:
        console.print(f"[red]Audio playback error: {e}")

if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input("\n[green]Press Enter to start recording, then press Enter again to stop.[/green]")

            data_queue = Queue()
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()  # Wait for the user to press Enter to stop recording
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            # Convert raw audio data to NumPy array and normalize to [-1.0, 1.0]
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe(audio_np)

                if text:
                    console.print(f"[yellow]You: {text}")

                    with console.status("Generating response...", spinner="earth"):
                        response = get_llm_response(text)

                    if response:
                        console.print(f"[cyan]Assistant: {response}")

                        with console.status("Synthesizing speech..."):
                            try:
                                sample_rate, audio_array = tts.long_form_synthesize(response)
                                play_audio(sample_rate, audio_array)
                            except Exception as e:
                                console.print(f"[red]TTS error: {e}")
                    else:
                        console.print("[red]Failed to generate a response.")
                else:
                    console.print("[red]No transcription available.")
            else:
                console.print("[red]No audio recorded. Please ensure your microphone is working.")

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
