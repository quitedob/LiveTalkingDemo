"""
LLM Management API
"""
from aiohttp import web
import json
from logger import logger

# Global LLM manager instance (will be set in app.py)
llm_manager = None


def set_llm_manager(manager):
    """
    Set the global LLM manager instance
    """
    global llm_manager
    llm_manager = manager


async def get_providers(request):
    """
    Get available LLM providers
    """
    try:
        if not llm_manager:
            return web.json_response({"error": "LLM manager not initialized"}, status=500)
        
        return web.json_response({
            "current_provider": llm_manager.get_current_provider(),
            "available_providers": llm_manager.get_available_providers(),
            "client_info": llm_manager.get_client_info()
        })
    except Exception as e:
        logger.error(f"Error getting LLM providers: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def switch_provider(request):
    """
    Switch LLM provider
    """
    try:
        if not llm_manager:
            return web.json_response({"error": "LLM manager not initialized"}, status=500)
        
        data = await request.json()
        if not data or 'provider' not in data:
            return web.json_response({"error": "Provider name required"}, status=400)
        
        provider = data['provider']
        success = await llm_manager.switch_provider(provider)
        
        if success:
            return web.json_response({
                "message": f"Successfully switched to {provider}",
                "current_provider": llm_manager.get_current_provider(),
                "client_info": llm_manager.get_client_info()
            })
        else:
            return web.json_response({"error": f"Failed to switch to {provider}"}, status=400)
            
    except Exception as e:
        logger.error(f"Error switching LLM provider: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def test_llm(request):
    """
    Test current LLM provider with a simple query
    """
    try:
        if not llm_manager:
            return web.json_response({"error": "LLM manager not initialized"}, status=500)
        
        data = await request.json()
        test_query = data.get('query', 'Hello, please respond with a simple greeting.')
        
        response_chunks = []
        async for chunk in llm_manager.ask(test_query):
            response_chunks.append(chunk)
        
        full_response = ''.join(response_chunks)
        
        return web.json_response({
            "provider": llm_manager.get_current_provider(),
            "query": test_query,
            "response": full_response,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error testing LLM: {e}")
        return web.json_response({"error": str(e), "success": False}, status=500)


def register_llm_routes(app):
    """
    Register LLM API routes with the aiohttp application
    """
    app.router.add_get('/llm/providers', get_providers)
    app.router.add_post('/llm/switch', switch_provider)
    app.router.add_post('/llm/test', test_llm)
    logger.info("LLM API路由注册完成")