import whisper
import sounddevice as sd
import numpy as np
import queue

def audio_callback(indata, frames, time, status):
    """This is called for each audio chunk from the microphone."""
    # Convert audio input to floating point (normalize)
    floatdata = indata.astype(np.float32)
    # Scale audio data to be between -1.0 and 1.0
    floatdata /= 32768
    q.put(floatdata.copy())

def transcribe_stream():
    # Define the sample rate
    SAMPLE_RATE = 16000

    # Load the Whisper model
    model = whisper.load_model("tiny")
    print("Model loaded successfully.")

    # Initialize a buffer to hold audio data
    audio_buffer = np.array([], dtype=np.float32)

    try:
        # Use the default microphone device as audio source
        with sd.InputStream(callback=audio_callback, dtype='int16', channels=1, samplerate=SAMPLE_RATE):
            print("Transcribing... (Press Ctrl+C to stop)")
            while True:
                data = q.get()
                # Append new data to the buffer
                audio_buffer = np.concatenate((audio_buffer, data.ravel()))

                # Check if we have enough data to transcribe
                if len(audio_buffer) >= SAMPLE_RATE * 5:  # 5 seconds of audio
                    # Transcribe the available audio specifying the language as English
                    result = model.transcribe(audio_buffer[:SAMPLE_RATE * 5], temperature=0, language="en")
                    print("Partial Transcription: ", result['text'])
                    # Remove the processed audio from the buffer
                    audio_buffer = audio_buffer[SAMPLE_RATE * 5:]
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == "__main__":
    q = queue.Queue()
    transcribe_stream()
