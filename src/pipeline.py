import asyncio
import subprocess
import sounddevice as sd
import numpy as np
import google.generativeai as genai
import ollama
from ollama import AsyncClient
from faster_whisper import WhisperModel
import re
import os
import wave
import threading

# Crucial fix for macOS + asyncio + CTranslate2 deadlocks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.metrics import LatencyTracker

class VoiceAssistantPipeline:
    def __init__(self, api_key=None, llm_type='gemini', model_name='gemini-3-flash-preview', replay_file=None):
        if api_key and llm_type == 'gemini':
            genai.configure(api_key=api_key)
            
        self.llm_type = llm_type
        self.model_name = model_name
        self.replay_file = replay_file
        self.metrics = LatencyTracker()
        
        # System constraints & Budgets
        self.LLM_TIMEOUT = 3.0  # Phase 3: Strict timeout for network calls
        self.SAMPLE_RATE = 16000
        
        # Async Queues
        self.audio_queue = asyncio.Queue()
        self.llm_queue = asyncio.Queue()
        self.tts_queue = asyncio.Queue()
        
        # Initialize Whisper locally to avoid threading issues
        os.environ["OMP_NUM_THREADS"] = "1"
        self.asr_model = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=4)
        
        # State
        self.is_running = True
        self.loop = None

    async def play_fallback_audio(self, filepath):
        """Phase 3: Graceful degradation instead of hanging silently."""
        print(f"[Fallback] Playing: {filepath}")
        if not os.path.exists(filepath):
            print(f"[Error] Fallback file {filepath} not found.")
            return

        def _play():
            try:
                with wave.open(filepath, 'rb') as wf:
                    data = wf.readframes(wf.getnframes())
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    sd.play(audio_data, samplerate=wf.getframerate())
                    sd.wait()
            except Exception as e:
                print(f"[Fallback Play Error] {e}")

        await asyncio.to_thread(_play)

    async def vad_worker(self):
        """Phase 1: Captures audio or Phase 3: Reads from replay file."""
        if self.replay_file:
            print(f"[Replay Mode] Injecting {self.replay_file} into pipeline...")
            with wave.open(self.replay_file, 'rb') as wf:
                data = wf.readframes(wf.getnframes())
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self.metrics.start_interaction()
                self.metrics.record_vad_end()
                
                try:
                    # Run ASR sequentially to avoid thread deadlocks on macOS
                    segments, _ = await asyncio.to_thread(self.asr_model.transcribe, audio_np, beam_size=1)
                    transcription = "".join([s.text for s in segments]).strip()
                    self.metrics.record_asr_end()
                    
                    if transcription:
                        print(f"[User]: {transcription}")
                        await self.llm_queue.put(transcription)
                    else:
                        print("[Debug] ASR returned empty transcription.")
                except Exception as e:
                    print(f"[ASR Error] {e}")
                    await self.play_fallback_audio("assets/fallback_error.wav")
            return # Exit VAD worker after injecting replay

        print("[Mic] Listening...")
        buffer, silence_frames, is_speaking, peak_volume = [], 0, False, 0.0
        SPEECH_THRESHOLD = 50.0  # Higher to trigger
        SILENCE_THRESHOLD = 20.0 # Lower to continue
        
        while self.is_running:
            try:
                chunk = self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)
                continue
            
            # Ensure chunk is 1D for volume calculation
            chunk_flat = chunk.flatten()
            volume = np.abs(chunk_flat).mean()
            
            if is_speaking:
                peak_volume = max(peak_volume, volume)

            # Logging volume periodically
            if self.loop.time() % 3 < 0.02: 
                print(f"[Debug] Vol: {volume:.1f} | Peak: {peak_volume:.1f} | Speaking: {is_speaking}")

            if volume > SPEECH_THRESHOLD:
                if not is_speaking:
                    self.metrics.start_interaction()
                    is_speaking = True
                    peak_volume = volume
                    print("[VAD] Speech detected!")
                buffer.append(chunk_flat)
                silence_frames = 0
            elif is_speaking:
                buffer.append(chunk_flat)
                if volume < SILENCE_THRESHOLD:
                    silence_frames += len(chunk_flat)
                else:
                    silence_frames = 0 # reset if there is still "active" noise
                
                # 1.5 seconds of silence triggers ASR
                if silence_frames > (1.5 * self.SAMPLE_RATE):
                    self.metrics.record_vad_end()
                    audio_data = np.concatenate(buffer, axis=0)
                    
                    # Normalize
                    max_val = np.abs(audio_data).max()
                    audio_float32 = audio_data.astype(np.float32) / (max_val if max_val > 0 else 32768.0)
                    
                    duration = len(audio_float32) / self.SAMPLE_RATE
                    print(f"[Debug] VAD triggering ASR. Duration: {duration:.1f}s | Peak: {peak_volume:.1f}")
                    
                    try:
                        segments, _ = await asyncio.to_thread(self.asr_model.transcribe, audio_float32, beam_size=1)
                        transcription = "".join([s.text for s in segments]).strip()
                        self.metrics.record_asr_end()
                        
                        if transcription:
                            print(f"[User]: {transcription}")
                            await self.llm_queue.put(transcription)
                        else:
                            print("[Debug] ASR returned empty transcription.")
                    except Exception as e:
                        print(f"[ASR Error] {e}")
                        await self.play_fallback_audio("assets/fallback_error.wav")
                    
                    buffer, is_speaking, silence_frames = [], False, 0

    async def llm_worker(self):
        """Streams to local Ollama or remote Gemini with sentence buffering."""
        sentence_end = re.compile(r'(?<=[.!?]) +')
        
        # Initialize Gemini if needed
        chat = None
        if self.llm_type == 'gemini':
            model = genai.GenerativeModel(self.model_name)
            chat = model.start_chat(history=[])

        while self.is_running:
            user_text = await self.llm_queue.get()
            print(f"[Debug] LLM received ({self.llm_type}): {user_text}")
            
            try:
                buffer = ""
                first_token_recorded = False

                if self.llm_type == 'ollama':
                    # Use Ollama AsyncClient for streaming
                    async_client = AsyncClient()
                    response = await async_client.chat(
                        model=self.model_name,
                        messages=[{'role': 'user', 'content': user_text}],
                        stream=True
                    )
                    
                    async for chunk in response:
                        if not first_token_recorded:
                            self.metrics.record_llm_first_token()
                            first_token_recorded = True
                        
                        content = chunk['message']['content']
                        buffer += content
                        
                        # Sentence buffering
                        sentences = sentence_end.split(buffer)
                        if len(sentences) > 1:
                            for s in sentences[:-1]:
                                if s.strip():
                                    await self.tts_queue.put(s.strip())
                            buffer = sentences[-1]
                
                elif self.llm_type == 'gemini' and chat:
                    response = await chat.send_message_async(user_text, stream=True)
                    async for chunk in response:
                        if not first_token_recorded:
                            self.metrics.record_llm_first_token()
                            first_token_recorded = True
                            
                        buffer += chunk.text
                        sentences = sentence_end.split(buffer)
                        if len(sentences) > 1:
                            for s in sentences[:-1]:
                                if s.strip():
                                    await self.tts_queue.put(s.strip())
                            buffer = sentences[-1]

                if buffer.strip():
                    await self.tts_queue.put(buffer.strip())

            except Exception as e:
                print(f"[LLM Error] {e}")
                await self.play_fallback_audio("assets/fallback_error.wav")

    async def tts_worker(self):
        """Synthesizes text using Piper TTS subprocess for high-fidelity audio."""
        model_path = os.path.abspath("assets/en_US-lessac-low.onnx")
        
        # Piper outputs raw 16-bit 16kHz PCM audio
        piper_cmd = ["piper", "--model", model_path, "--output_raw"]
        
        while self.is_running:
            sentence = await self.tts_queue.get()
            print(f"[Debug] TTS received text: {sentence}")
            
            # 1. Start Piper subprocess
            process = await asyncio.create_subprocess_exec(
                *piper_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            
            if process.stdin is None or process.stdout is None:
                print("[TTS Error] Piper subprocess failed to open pipes.")
                continue
            
            # 2. Feed the sentence into Piper and close stdin
            process.stdin.write((sentence + "\n").encode('utf-8'))
            await process.stdin.drain()
            process.stdin.close()
            
            first_byte_recorded = False
            chunk_size = 4096 # ~0.12 seconds of audio
            
            print(f"[Assistant]: {sentence}")
            
            # 3. Stream the raw audio directly to sounddevice
            # Using RawOutputStream since piper gives us raw PCM bytes
            stream = sd.RawOutputStream(samplerate=16000, channels=1, dtype='int16')
            with stream:
                while True:
                    data = await process.stdout.read(chunk_size)
                    if not data:
                        break
                    
                    if not first_byte_recorded:
                        self.metrics.record_tts_first_byte()
                        first_byte_recorded = True
                        
                    # Write audio chunk to speakers (runs in thread to avoid blocking loop)
                    await asyncio.to_thread(stream.write, data)
            
            await process.wait()

    def audio_callback(self, indata, frames, time, status):
        if status: print(status)
        self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, indata.copy())

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.is_running = True
        
        workers = [
            asyncio.create_task(self.vad_worker()),
            asyncio.create_task(self.llm_worker()),
            asyncio.create_task(self.tts_worker())
        ]
        
        if not self.replay_file:
            stream = sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1, dtype='int16', callback=self.audio_callback)
            with stream:
                await asyncio.gather(*workers)
        else:
            await asyncio.gather(*workers)
