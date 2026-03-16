import time
import csv
import os

class LatencyTracker:
    def __init__(self, log_file="latency_logs.csv"):
        self.log_file = log_file
        self.current_interaction = {}
        
        # Initialize CSV with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "vad_latency_s", "asr_processing_s", "llm_ttft_s", "tts_ttfb_s", "total_system_latency_s"])

    def start_interaction(self):
        self.current_interaction = {'start_time': time.perf_counter()}

    def record_vad_end(self):
        self.current_interaction['vad_end'] = time.perf_counter()
        self.current_interaction['vad_latency'] = self.current_interaction['vad_end'] - self.current_interaction['start_time']

    def record_asr_end(self):
        self.current_interaction['asr_end'] = time.perf_counter()
        vad_end = self.current_interaction.get('vad_end', self.current_interaction.get('start_time', time.perf_counter()))
        self.current_interaction['asr_processing'] = self.current_interaction['asr_end'] - vad_end

    def record_llm_first_token(self):
        asr_end = self.current_interaction.get('asr_end', time.perf_counter())
        self.current_interaction['llm_ttft'] = time.perf_counter() - asr_end

    def record_tts_first_byte(self):
        now = time.perf_counter()
        asr_end = self.current_interaction.get('asr_end', now)
        llm_ttft = self.current_interaction.get('llm_ttft', 0)
        
        self.current_interaction['tts_ttfb'] = now - (asr_end + llm_ttft)
        vad_end = self.current_interaction.get('vad_end', now)
        self.current_interaction['total_latency'] = now - vad_end
        self.save_metrics()

    def save_metrics(self):
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                f"{self.current_interaction.get('vad_latency', 0):.3f}",
                f"{self.current_interaction.get('asr_processing', 0):.3f}",
                f"{self.current_interaction.get('llm_ttft', 0):.3f}",
                f"{self.current_interaction.get('tts_ttfb', 0):.3f}",
                f"{self.current_interaction.get('total_latency', 0):.3f}"
            ])
        print(f"\n[Metrics] Total Latency: {self.current_interaction.get('total_latency', 0):.2f}s | Logged to {self.log_file}\n")
