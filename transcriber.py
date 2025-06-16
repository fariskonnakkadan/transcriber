import argparse
import whisper
from pydub import AudioSegment
from tqdm import tqdm  # For showing a progress bar
import torch  # To check CUDA availability
import warnings
import os
import sys
import threading
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Global stop event for threads
stop_event = threading.Event()

class Silent:
    """
    Suppresses stdout and stderr, except for `tqdm` output when `allow_tqdm` is True.
    """
    def __init__(self, allow_tqdm=False):
        self.allow_tqdm = allow_tqdm

    def __enter__(self):
        if not self.allow_tqdm:
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.allow_tqdm:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr


def transcribe_chunk(model, chunk_file, device, transcription_results, index, progress_bar):
    """
    Transcribes a single audio chunk using the specified device and updates the progress bar.
    """
    result = model.transcribe(chunk_file, language="en")
    transcription_results[index] = result["text"]
    progress_bar.update(1)  # Increment progress bar after each chunk is processed


def transcribe_audio(file_path, output_path, device_select="cpu,cuda", model_select=None, chunk_duration_ms=30000):
    """
    Transcribes an audio file using Whisper and processes chunks in parallel,
    one on CPU and one on CUDA if available.
    """
    start_time = time.time()

    modelname = model_select if model_select else "large"  # Default model
    devices = device_select.split(",")
    models = {}

    with Silent(allow_tqdm=True):  # Silence all output except for tqdm
        # Load models for each device
        for device in devices:
            models[device] = whisper.load_model(model_select or modelname).to(device)
            print(f"Loaded model '{modelname}' on {device}")

        # Load the audio file
        audio = AudioSegment.from_file(file_path)

        # Calculate the number of chunks
        total_duration_ms = len(audio)
        num_chunks = total_duration_ms // chunk_duration_ms + (1 if total_duration_ms % chunk_duration_ms > 0 else 0)

        transcription_results = [None] * num_chunks

        # Create a shared progress bar
        progress_bar = tqdm(total=num_chunks, desc="Transcribing", unit="chunk", ncols=80)

        # Process audio chunks
        def process_chunks(chunk_indices, device):
            for index in chunk_indices:
                if stop_event.is_set():
                    break
                start_ms = index * chunk_duration_ms
                end_ms = min((index + 1) * chunk_duration_ms, total_duration_ms)
                chunk = audio[start_ms:end_ms]

                # Save the chunk as a temporary file
                chunk_file = f"chunk_{index}.wav"
                chunk.export(chunk_file, format="wav")

                try:
                    transcribe_chunk(models[device], chunk_file, device, transcription_results, index, progress_bar)
                finally:
                    os.remove(chunk_file)  # Cleanup

        # Assign chunks to devices based on input devices
        if "cuda" in devices and torch.cuda.is_available():
            # Assign chunks to CUDA and CPU when both are available
            cpu_indices = [i for i in range(num_chunks) if i % 2 == 0]
            cuda_indices = [i for i in range(num_chunks) if i % 2 == 1]
        elif "cpu" in devices:
            # If only CPU is selected, assign all chunks to CPU
            cpu_indices = list(range(num_chunks))
            cuda_indices = []  # No chunks for CUDA

        threads = []
        if "cpu" in devices:
            threads.append(threading.Thread(target=process_chunks, args=(cpu_indices, "cpu"), daemon=True))
        if "cuda" in devices and torch.cuda.is_available():
            threads.append(threading.Thread(target=process_chunks, args=(cuda_indices, "cuda"), daemon=True))

        try:
            # Start threads
            for thread in threads:
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Stopping...")
            stop_event.set()  # Signal threads to stop
            for thread in threads:
                thread.join()  # Ensure threads exit

        progress_bar.close()  # Close progress bar when done

        # Combine results
        transcription = "\n".join(filter(None, transcription_results))

        # Save transcription to the output file
        with open(output_path, "w") as f:
            f.write(transcription)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Transcription saved to {output_path}")
        print(f"Total processing time: {total_time:.2f} seconds")


if __name__ == "__main__":
    with Silent(allow_tqdm=True):  # Allow argparse messages
        parser = argparse.ArgumentParser(description="Transcribe an audio file using Whisper with progress and GPU support.")
        parser.add_argument("--input", required=True, help="Path to the input audio file (e.g., meeting_recording.wav)")
        parser.add_argument("--output", required=True, help="Path to the output text file (e.g., transcription.txt)")
        parser.add_argument("--device", default="cpu,cuda", help="Devices to use, e.g., 'cpu,cuda'")
        parser.add_argument("--model", required=False, help="Model: tiny, base, small, medium, large")
        args = parser.parse_args()

    # Transcribe the audio file and save to output file
    transcribe_audio(args.input, args.output, args.device, args.model)
