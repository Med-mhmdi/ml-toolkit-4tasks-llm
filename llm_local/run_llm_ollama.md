# Run Local LLM via Ollama

1. Install Ollama → https://ollama.com/download
2. Pull a model:
   ```bash
   ollama pull mistral:7b-instruct
   ```
3. Run it:
   ```bash
   ollama run mistral:7b-instruct
   ```
4. Or test with a prompt file:
   ```bash
   ollama run mistral:7b-instruct < prompts/hello.txt
   ```
