# Whisper flow Freestyle

Whisper Flow is a Gradio application for transcribing speech with OpenAI's
`whisper-medium` model. It supports live microphone transcription and
transcription of uploaded audio files.

## Use the `whisper-small` model

The app uses `openai/whisper-medium` by default. To use the smaller and faster
`whisper-small` model instead, edit `venv/app.py` and change both occurrences of
`openai/whisper-medium` to `openai/whisper-small` in the
`AutoProcessor.from_pretrained(...)` and
`AutoModelForSpeechSeq2Seq.from_pretrained(...)` calls.

Then rebuild the Docker image and start it using the commands below. The first
run downloads the small model into the same Hugging Face cache volume.

## Requirements

- Docker with the Docker Compose/Linux container engine available
- An NVIDIA GPU and NVIDIA Container Toolkit (the app loads Whisper onto CUDA)
- Internet access on first launch to download the model from Hugging Face

The first start downloads the model, which can take some time and disk space.
The Docker commands below keep the Hugging Face cache in a named volume so the
model does not need to be downloaded again each time.

## Run with Docker

From the project directory, build the image:

```powershell
docker build -t whisper-flow .
```

Start the app:

```powershell
docker run --rm --gpus all -p 7860:7860 -v whisper-flow-hf:/app/.cache/huggingface whisper-flow
```

Open [http://localhost:7860](http://localhost:7860) in a browser.

To stop the app, press `Ctrl+C` in the terminal running the container. The
`whisper-flow-hf` volume retains downloaded model files between runs.

## Use the application

- **Real-time:** open the Real-time tab, allow microphone access, and speak.
  The app detects pauses in speech and displays the transcription.
- **Upload audio:** open the Upload audio tab, choose an audio file, and select
  **Transcribe**. The resulting transcription appears in the text box.

## Dependencies

Python dependencies and their version constraints are listed in
[`requirements.txt`](requirements.txt). NumPy is pinned to `1.26.0`.
