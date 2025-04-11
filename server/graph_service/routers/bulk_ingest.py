# graph_service/routers/bulk_ingest.py
# Defines the API endpoint for bulk episode ingestion.

import logging
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field # Import BaseModel for dummy classes if needed

from fastapi import APIRouter, Depends, HTTPException, status

logger = logging.getLogger(__name__)
# Assuming auth, DTOs, and Graphiti dependency setup are accessible
# Adjust these imports based on your project structure

from graph_service.auth import get_api_key
from graph_service.zep_graphiti import ZepGraphiti, ZepGraphitiDep


# Import the bulk processor function and necessary models from graphiti_core
# Adjust path if your structure is different
from graphiti_core.bulk_processor import process_episodes_bulk, RawEpisode, BulkProcessingResult


logger = logging.getLogger(__name__)
# Create the router instance for this module
router = APIRouter()

# --- Define Request Body Model ---
# Uses the RawEpisode model defined in graphiti_core.bulk_processor
class AddBulkEpisodesRequest(BaseModel):
    episodes: List[RawEpisode] = Field(..., description='A batch of raw episodes to process')
    # Optional: Add other parameters like entity_types if needed for process_episodes_bulk
    # entity_types: Optional[Dict[str, Any]] = None # Example


# --- Define Response Model ---
# Uses the BulkProcessingResult model defined in graphiti_core.bulk_processor
class BulkIngestResponse(BulkProcessingResult):
    """Response model indicating the results of the bulk processing batch."""
    pass # Inherits fields from BulkProcessingResult


# --- Bulk Ingestion Endpoint ---
@router.post(
    '/episodes', # Endpoint path relative to the router prefix (e.g., /bulk/episodes)
    response_model=BulkIngestResponse,
    status_code=status.HTTP_200_OK, # Return 200 on successful processing of batch
    dependencies=[Depends(get_api_key)],
    summary="Process a batch of episodes using high-quality bulk ingestion logic",
    description="Accepts a list of raw episode data, processes them using the internal bulk processor (reflexion, resolution, etc.), and saves results to the graph. This is a synchronous endpoint from the client's perspective - it waits for the batch processing to complete."
)
async def add_bulk_episodes(
    request: AddBulkEpisodesRequest,
    graphiti: ZepGraphitiDep, # Inject the Graphiti instance
):
    """
    Handles bulk ingestion of episodes by calling the internal
    process_episodes_bulk function from graphiti_core.
    """
    if not request.episodes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No episodes provided in the batch.")

    # Ensure graphiti dependency is resolved correctly
    # Note: ZepGraphitiDep might resolve to the Graphiti base class or a subclass like ZepGraphiti
    # Using isinstance(graphiti, Graphiti) should work for base or subclasses.
    if not isinstance(graphiti, ZepGraphiti):
         logger.error("Graphiti dependency not resolved correctly in bulk endpoint.")
         raise HTTPException(status_code=500, detail="Internal server configuration error.")

    logger.info(f"Received bulk request to process {len(request.episodes)} episodes.")
    # Group ID is expected within each RawEpisode object in the list

    try:
        # --- Call the core bulk processing logic ---
        # This call is awaited, so the endpoint waits for completion
        result: BulkProcessingResult = await process_episodes_bulk(
            graphiti=graphiti,
            raw_episodes_batch=request.episodes,
            # Pass other necessary params from graphiti instance or request if needed
            # entity_types=request.entity_types, # If added to request model
            store_raw_content=getattr(graphiti, 'store_raw_episode_content', True) # Safely get attribute
        )
        logger.info(f"Bulk batch processed successfully in {result.duration_seconds:.2f}s. Results: {result.model_dump()}")
        # Return the detailed result using the response model
        return BulkIngestResponse(**result.model_dump())

    except NotImplementedError as nie:
         logger.error(f"Bulk processing function not implemented or import failed: {nie}")
         raise HTTPException(status_code=501, detail="Bulk processing feature not available due to server configuration error.")
    except Exception as e:
        # Log the error and return a generic 500
        logger.error(f"Error during bulk processing: {e}", exc_info=True)
        # Provide a more specific error message if possible
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error during bulk processing: {str(e)}")

