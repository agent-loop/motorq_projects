import pyaudio
import wave
import os
import whisper

def record_chunk(p, stream, file_path, chunk_length=1):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

# Function to transcribe a chunk of audio
def transcribe_chunk(model, file_path):
    result = model.transcribe(file_path)
    return result["text"]

def main2():
    model = whisper.load_model("base")  
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    accumulated_transcription = ""
    try:
        while True:
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file)
            transcription = transcribe_chunk(model, chunk_file)
            print(transcription)
            accumulated_transcription += transcription + "\n"
            os.remove(chunk_file)
    except KeyboardInterrupt:
        print("Stopping...")
        with open("log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)
    finally:
        print("LOG:\n" + accumulated_transcription)
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main2()
