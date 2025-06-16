# Transcriber

## Whisper Audio Transcription Tool

This command-line tool transcribes an audio file using OpenAI's Whisper model, with support for progress tracking and GPU acceleration.

### Summary

The script uses Python's `argparse` module to handle command-line arguments for transcribing audio files. It allows users to specify:

- The input audio file
- The output text file
- The device(s) to use (CPU/GPU)
- The Whisper model variant

### Command-Line Arguments

| Argument       | Required | Description                                                                 |
|----------------|--------|-----------------------------------------------------------------------------|
| `--input`      | Yes   | Path to the input audio file (e.g., `meeting_recording.wav`)                |
| `--output`     | Yes   | Path to the output text file (e.g., `transcription.txt`)                    |
| `--device`     | No    | Devices to use for processing (default: `"cpu,cuda"`)                        |
| `--model`      | No    | Whisper model to use: `tiny`, `base`, `small`, `medium`, or `large`         |

### Example Usage

```bash
python transcribe.py --input meeting.wav --output transcript.txt --device cuda --model base
```

This command will:
- Transcribe `meeting.wav`
- Save the result to `transcript.txt`
- Use the GPU (`cuda`) for processing
- Use the `base` Whisper model
