import asyncio
import sounddevice as sd
import numpy as np

async def test_mic():
    audio_queue = asyncio.Queue()
    
    def callback(indata, frames, time, status):
        loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy())
        
    loop = asyncio.get_running_loop()
    stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16', callback=callback)
    
    print("Testing mic... Please speak for 5 seconds...")
    with stream:
        for _ in range(50): # read ~5 seconds
            chunk = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
            volume = np.abs(chunk).mean()
            print(f"Volume: {volume:.2f}")

asyncio.run(test_mic())
