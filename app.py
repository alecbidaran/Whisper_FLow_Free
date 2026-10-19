import gradio as gr 
import numpy as np 
import librosa
import torch
import os
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from fastrtc import (
    ReplyOnPause, AdditionalOutputs, Stream, AlgoOptions,
    audio_to_bytes, aggregate_bytes_to_16bit
)
transcription_memory=[]
processor = AutoProcessor.from_pretrained("openai/whisper-medium") #  whisper-small for faster interaction and light weight GPUs.
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-medium", device_map="cuda")


def transcribe_samples(y, sr):
    if np.issubdtype(y.dtype, np.integer):
        y = librosa.util.buf_to_float(y, dtype=np.float32)
    else:
        y = np.asarray(y, dtype=np.float32)

    # Resample to 16000 Hz (required by Whisper model)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    input_features = processor(y, sampling_rate=sr, return_tensors="pt").input_features 

    with torch.no_grad():
        generated_ids = model.generate(input_features.to('cuda'))

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return transcription[0]


def transcribe(audio):
    sr, y = audio
    yield AdditionalOutputs(transcribe_samples(y, sr))


def transcribe_uploaded_audio(audio_path):
    if not audio_path:
        raise gr.Error("Upload an audio file first.")

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    return transcribe_samples(y, sr)

def display_transcription(current_transcription: str | None, transcription: str) -> str:
    #transcription_memory.append(current_transcription or transcription )
    return transcription or current_transcription or ""

doc=gr.Textbox(label="Transcription")
stream = Stream(
    modality="audio",
    mode="send",
    handler=ReplyOnPause(
        transcribe,
        algo_options=AlgoOptions(
            audio_chunk_duration=1.0,
            speech_threshold=0.1,
        ),
    ),
    additional_inputs=[doc],
    additional_outputs=[doc],
    additional_outputs_handler=display_transcription,
    ui_args={"full_screen": False},
)

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Real-time"):
            stream.ui.render()
        with gr.Tab("Upload audio"):
            audio_file = gr.Audio(
                label="Audio file",
                sources=["upload"],
                type="filepath",
            )
            transcribe_button = gr.Button("Transcribe")
            uploaded_transcription = gr.Textbox(label="Transcription")
            transcribe_button.click(
                fn=transcribe_uploaded_audio,
                inputs=audio_file,
                outputs=uploaded_transcription,
            )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)