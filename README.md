# LiveKit Voice AI Agent

A voice AI assistant powered by LiveKit with Deepgram STT, Groq LLM, and ElevenLabs TTS.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your `.env` file contains all the required API keys:
   - `DEEPGRAM_API_KEY` - for speech-to-text
   - `ELEVENLABS_API_KEY` - for text-to-speech  
   - `GROQ_API_KEY` - for the language model
   - `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` - for LiveKit connection

## Running the Agent

```bash
python agent.py
```

The agent will connect to your LiveKit room and provide voice AI assistance using:
- **STT**: Deepgram Nova-2 General
- **LLM**: Groq Llama-3.1-70B Versatile  
- **TTS**: ElevenLabs Eleven Turbo V2
- **VAD**: Silero Voice Activity Detection
