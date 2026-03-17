# Real-Time Local Voice Assistant

A modular, high-performance voice assistant designed for low latency and local-first execution on macOS. This project integrates Speech-to-Text (ASR), Large Language Models (LLM), and Text-to-Speech (TTS) into a unified asynchronous pipeline.

## 🚀 Features

- **Dual-Mode LLM Support**: Switch between high-performance Google Gemini Flash via API or fully local inference using **Ollama** (supports `gemma:2b` and `llama3.2:1b`).
- **Interactive Model Selection**: Automatically prompts for your preferred local model when running in local mode.
- **High-Fidelity ASR**: Powered by `faster-whisper` (OpenAI Whisper) for accurate, real-time transcription.
- **Natural-Sounding TTS**: Uses **Piper** for high-quality, streaming speech synthesis.
- **Robust Voice Activity Detection (VAD)**: Custom dual-threshold logic to handle background noise and low mic sensitivity on laptops.
- **Observability**: Real-time performance metrics (TTFT, TTFB, Total Latency) logged to CSV for optimization.
- **Replay Mode**: Debugging functionality to test the pipeline using pre-recorded audio files.

---

## 🛠️ Prerequisites

1. **Python 3.10+**: Recommended environment.
2. **Ollama** (Optional): Only required for local LLM mode. [Download here](https://ollama.com/).
   - Pull the models: 
     ```bash
     ollama pull gemma:2b
     ollama pull llama3.2:1b
     ```
3. **Piper TTS**: The assistant expects `piper` to be available in your path or virtual environment.

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abhishekshukla15268-spec/Voice-Assistant-using-Ollama.git
   cd Voice-Assistant-using-Ollama
   ```

2. **Set up Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```text
   GEMINI_API_KEY=your_api_key_here
   ```

---

## 🎮 Usage

### 1. Live Mode (Gemini)
The default mode uses the Google Gemini API for fast, remote processing.
```bash
python3 main.py
```

### 2. Local Mode (Ollama)
Runs the entire LLM chain locally using Ollama. No API key required.
```bash
# Interactive mode (prompts for model choice)
python3 main.py --local

# Direct model selection (skips prompt)
python3 main.py --local --model llama3.2:1b
```

### 3. Replay Mode (Debugging)
Inject a specific audio file instead of using the microphone to test transcription and response accuracy.
```bash
python3 main.py --replay assets/test_input.wav
```

---

## 📊 Performance Metrics

The assistant tracks latency across every stage of the pipeline. Metrics are saved to `latency_logs.csv` and include:
- **ASR Processing**: Time taken to transcribe speech.
- **LLM TTFT**: Time to First Token from the LLM.
- **TTS TTFB**: Time to First Byte of generated audio.
- **Total Latency**: The "End-to-End" delay from silence detection to audio output.

---

## 🏗️ Technical Architecture

- **`main.py`**: Handles CLI arguments and orchestrates component initialization.
- **`src/pipeline.py`**: The core "Engine" containing asynchronous workers for VAD, ASR, LLM, and TTS.
- **`src/metrics.py`**: Thread-safe latenty tracking and data persistence.
- **`assets/`**: Contains fallback audio clips and pre-recorded test inputs.

---

## 🔧 Troubleshooting

- **Mic not picking up?**: The VAD threshold is tuned for the internal MacBook Air microphone. You can adjust `SPEECH_THRESHOLD` in `src/pipeline.py` if your environment is loud.
- **Hearing multiple voices?**: Ensure you don't have multiple instances of the assistant running. Use `pkill -9 python3` if stuck.
- **Piper not working?**: Confirm the Piper binary is accessible and the `.onnx` model file exists in `assets/`.
