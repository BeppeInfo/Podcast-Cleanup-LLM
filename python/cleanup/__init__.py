"""Decision-making half of the podcast cleanup pipeline.

The shell drives ffmpeg, Whisper and llama-server; everything in this package
reads and writes JSON in the episode work directory and never touches audio.
"""

__all__ = ["intervals", "vad", "transcript", "llm", "plan", "render"]
