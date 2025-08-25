# LLM Hot-Swapping System

This package provides a hot-swapping system for Large Language Models (LLMs) in the LiveTalking project, allowing seamless switching between different LLM providers without interrupting WebRTC connections.

## Features

- **Hot-swapping**: Switch between LLM providers without restarting the application
- **Multiple Providers**: Support for Ollama and DeepSeek (OpenAI-compatible)
- **Unified Interface**: All providers implement the same interface for consistent behavior
- **Stream Processing**: Maintains the existing streaming response functionality
- **Think Chain Filtering**: Advanced filtering of `<think>...</think>` blocks across streaming chunks

## Supported Providers

### Ollama
- Local LLM inference server
- Supports various open-source models (Gemma, Llama, etc.)
- Configuration via URL and model name

### DeepSeek
- Cloud-based LLM service
- OpenAI-compatible API
- Requires API key configuration

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# DeepSeek API Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Optional: Override default Ollama settings
OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_MODEL=gemma3:4b
```

### API Endpoints

The system provides REST API endpoints for management:

- `GET /llm/providers` - Get available providers and current status
- `POST /llm/switch` - Switch to a different provider
- `POST /llm/test` - Test current provider with a simple query

## Usage

### Backend Integration

```python
from pkg.llm import LLMManager

# Initialize manager
llm_manager = LLMManager()

# Register providers
llm_manager.register_ollama_client(
    url="http://localhost:11434/api/chat",
    model="gemma3:4b",
    system_prompt="You are a helpful assistant."
)

llm_manager.register_deepseek_client(
    model="deepseek-chat",
    system_prompt="You are a helpful assistant."
)

# Use the manager
async for chunk in llm_manager.ask("Hello, world!"):
    print(chunk)

# Switch providers
await llm_manager.switch_provider("deepseek")
```

### Frontend Integration

The web interfaces (`webrtc1.html` and `demo.html`) include LLM switching controls:

1. **Provider Selection**: Dropdown to choose between Ollama and DeepSeek
2. **Switch Button**: Apply the selected provider
3. **Test Button**: Verify the current provider is working
4. **Status Display**: Shows current provider and model information

## Architecture

```
pkg/llm/
├── __init__.py          # Package exports
├── base.py              # Abstract base class
├── ollama_client.py     # Ollama implementation
├── deepseek_client.py   # DeepSeek implementation
├── manager.py           # Hot-swapping manager
└── README.md           # This file

api/
└── llm_api.py          # REST API endpoints
```

### Key Components

1. **BaseLLMClient**: Abstract base class defining the interface
2. **OllamaClient**: Implementation for local Ollama servers
3. **DeepSeekClient**: Implementation for DeepSeek cloud service
4. **LLMManager**: Orchestrates hot-swapping between providers

## WebRTC Stability

The hot-swapping system is designed to maintain WebRTC connection stability:

- **Non-blocking**: Provider switches happen asynchronously
- **Session Preservation**: WebRTC sessions remain active during switches
- **Error Isolation**: Provider failures don't affect WebRTC connections
- **Graceful Fallback**: System continues with current provider if switch fails

## Testing

Run the test script to verify the system:

```bash
python test_llm_system.py
```

This will:
1. Initialize both providers
2. Test the current provider
3. Switch to the alternate provider
4. Test the new provider
5. Display results and status information

## Troubleshooting

### Common Issues

1. **DeepSeek API Key Not Found**
   - Ensure `.env` file exists with `DEEPSEEK_API_KEY`
   - Check file permissions and encoding

2. **Ollama Connection Failed**
   - Verify Ollama server is running on specified URL
   - Check firewall and network connectivity
   - Ensure the specified model is available

3. **Provider Switch Failed**
   - Check logs for specific error messages
   - Verify provider configuration
   - Test individual providers before switching

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger('pkg.llm').setLevel(logging.DEBUG)
```

## Future Enhancements

- Support for additional providers (OpenAI, Anthropic, etc.)
- Provider health monitoring and automatic failover
- Load balancing across multiple instances
- Provider-specific configuration profiles
- Metrics and performance monitoring