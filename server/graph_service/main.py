from contextlib import asynccontextmanager
from http.client import HTTPException

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from graph_service.config import get_settings
from graph_service.routers import ingest, retrieve, bulk_ingest
from graph_service.zep_graphiti import initialize_graphiti
import logging
import sys


def setup_logging():
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the minimum level you want to capture

    # Remove existing handlers if necessary (to avoid duplicates)
    # for handler in logger.handlers[:]:
    #     logger.removeHandler(handler)

    # Create a stream handler writing to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO) # Set level for this handler

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the root logger
    logger.addHandler(handler)

# Call this setup function early in your application startup
setup_logging()

logger = logging.getLogger(__name__) 

# In graph_service/main.py
@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        print("Starting Graphiti initialization")
        settings = get_settings()
        print(f"Using settings: Neo4j URI={settings.neo4j_uri}, User={settings.neo4j_user}")
        await initialize_graphiti(settings)
        print("Graphiti initialization complete")
        yield
    except Exception as e:
        logger.error(f"Error during Graphiti initialization: {str(e)}", exc_info=True)
        # Still yield to allow the application to start even with errors
        yield
    finally:
        print("Shutdown complete")

app = FastAPI(lifespan=lifespan)


app.include_router(retrieve.router)
app.include_router(ingest.router)
app.include_router(bulk_ingest.router)


@app.get('/healthcheck')
async def healthcheck():
    return JSONResponse(content={'status': 'healthy'}, status_code=200)

# Add temporary endpoint in main.py
from graph_service.zep_graphiti import ZepGraphitiDep

@app.get("/debug_llm")
async def debug_llm(graphiti: ZepGraphitiDep):
    logger.info("--- DEBUG: Testing LLM Client ---")
    try:
        # Example: Make a simple call your LLM client supports
        # Replace with an actual simple call, e.g., embedding
        test_embedding = await graphiti.embedder.create("test")
        logger.info(f"--- DEBUG: Embedding successful: {test_embedding[:5]}... ---")

        # Example: Simple chat completion if applicable
        # response = await graphiti.llm_client.generate_response(
        #     [Message(role="user", content="Ping")], max_tokens=5
        # )
        # logger.info(f"--- DEBUG: LLM response: {response} ---")

        return {"status": "LLM/Embedding test initiated, check logs."}
    except Exception as e:
        logger.error("--- DEBUG: Error during LLM/Embedding test ---", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Debug test failed: {e}")
    