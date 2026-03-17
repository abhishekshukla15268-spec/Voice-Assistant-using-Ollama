import asyncio
import argparse
import os
from dotenv import load_dotenv
from src.pipeline import VoiceAssistantPipeline

def main():
    # Load environment variables
    load_dotenv()
    
    # Phase 3: CLI implementation for Replay Mode debugging
    parser = argparse.ArgumentParser(description="Real-Time Local Voice Assistant")
    parser.add_argument("--replay", type=str, help="Path to a .wav file to inject for testing (bypasses microphone).")
    parser.add_argument("--local", action="store_true", help="Use local Ollama instead of Gemini API.")
    parser.add_argument("--model", type=str, help="LLM model name (e.g., 'gemma:2b' for local or 'gemini-3-flash-preview' for remote).")
    args = parser.parse_args()

    llm_type = 'ollama' if args.local else 'gemini'
    
    # Defaults
    if not args.model:
        if args.local:
            print("\nSelect Local LLM Model:")
            print("1. gemma:2b")
            print("2. llama3.2:1b")
            choice = input("Enter choice (1 or 2, default 1): ").strip()
            
            if choice == '2':
                model_name = 'llama3.2:1b'
            else:
                model_name = 'gemma:2b'
        else:
            model_name = 'gemini-3-flash-preview'
    else:
        model_name = args.model

    api_key = os.getenv("GEMINI_API_KEY")
    if llm_type == 'gemini' and not api_key:
        print("Error: GEMINI_API_KEY not found. Use --local to skip Gemini API.")
        return

    # Initialize and run pipeline
    pipeline = VoiceAssistantPipeline(
        api_key=api_key, 
        llm_type=llm_type, 
        model_name=model_name, 
        replay_file=args.replay
    )
    
    print("===================================================")
    print(" Voice Assistant Initialized")
    print(f" LLM Type: {llm_type.upper()}")
    print(f" Model: {model_name}")
    print(f" Mode: {'REPLAY' if args.replay else 'LIVE MICROPHONE'}")
    print("===================================================")

    try:
        asyncio.run(pipeline.start())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        pipeline.is_running = False

if __name__ == "__main__":
    main()
