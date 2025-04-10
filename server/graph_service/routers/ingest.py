# Revised: server/graph_service/routers/ingest.py
# Reverted to processing individual messages via add_episode for reliability

import asyncio
import logging
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, List, Optional
import sys

from pydantic import BaseModel

from fastapi.params import Depends
from graph_service.auth import get_api_key

from fastapi import APIRouter, FastAPI, HTTPException, status
# Ensure EpisodeType is imported if not already
from graphiti_core.nodes import EpisodeType # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data # type: ignore

from graph_service.dto import AddEntityNodeRequest, AddMessagesRequest, Message, Result
# Import ZepGraphiti directly for type hinting the instance
from graph_service.zep_graphiti import ZepGraphiti, ZepGraphitiDep

logger = logging.getLogger(__name__)

# --- Improved AsyncWorker (keeps improved error handling) ---
class AsyncWorker:
    def __init__(self):
        self.queue: asyncio.Queue[partial[Any]] = asyncio.Queue()
        self.task: asyncio.Task | None = None

    async def worker(self):
        print("--- PRINT: AsyncWorker worker() task entered ---", file=sys.stdout, flush=True)
        logger.info("--- LOGGER: AsyncWorker worker() task entered ---")
        while True:
            job_details_str = "N/A"
            job_func: partial[Any] | None = None
            try:
                print("--- PRINT: Worker waiting for job on queue.get() ---", file=sys.stdout, flush=True)
                logger.info("--- LOGGER: Worker waiting for job on queue.get() ---")
                job_func = await self.queue.get()
                print(f"--- PRINT: Worker received job from queue (qsize={self.queue.qsize()}) ---", file=sys.stdout, flush=True)
                logger.info(f"--- LOGGER: Worker received job from queue (qsize={self.queue.qsize()}) ---")

                # Extract details for logging (assuming args[0] is graphiti, args[1] is message, args[2] is group_id)
                if hasattr(job_func, 'func') and hasattr(job_func, 'args') and len(job_func.args) >= 3:
                    msg_data = job_func.args[1] # The Message object
                    group_id = job_func.args[2] # The group_id string
                    if isinstance(msg_data, Message) and isinstance(group_id, str):
                         job_details_str = (
                             f"GroupID: {group_id}, "
                             f"MsgRole: {msg_data.role or msg_data.role_type}, "
                             f"MsgTS: {msg_data.timestamp.isoformat()}"
                         )
                else:
                     job_details_str = "Job details unavailable (unexpected task format)"

                logger.info(f"Processing job (Queue size: {self.queue.qsize()}). Details: {job_details_str}")
                print(f"--- PRINT: Processing job. Details: {job_details_str} ---", file=sys.stdout, flush=True)

                await job_func() # Execute add_episode task

                logger.info(f"Successfully processed job. Details: {job_details_str}")
                print(f"--- PRINT: Successfully processed job. Details: {job_details_str} ---", file=sys.stdout, flush=True)

            except asyncio.CancelledError:
                logger.info("Worker task cancelled.")
                print("--- PRINT: Worker task cancelled. ---", file=sys.stdout, flush=True)
                break
            except Exception:
                logger.error(
                    f"!!! Error processing background job. Details: {job_details_str} !!!",
                    exc_info=True
                )
                print(f"--- PRINT: !!! Error processing background job. Details: {job_details_str} !!! ---", file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc(file=sys.stderr)
            finally:
                 if job_func:
                    self.queue.task_done()
                    print(f"--- PRINT: Worker called task_done for job. Details: {job_details_str} ---", file=sys.stdout, flush=True)
                    logger.info(f"--- LOGGER: Worker called task_done for job. Details: {job_details_str} ---")

    async def start(self):
        if not self.task or self.task.done():
             print("--- PRINT: AsyncWorker start() called - Starting task ---", file=sys.stdout, flush=True)
             logger.info("Starting background worker...")
             self.task = asyncio.create_task(self.worker())
        else:
             print("--- PRINT: AsyncWorker start() called - Task already running ---", file=sys.stdout, flush=True)
             logger.warning("Worker task already running.")

    async def stop(self):
        if self.task and not self.task.done():
            print("--- PRINT: AsyncWorker stop() called - Cancelling task ---", file=sys.stdout, flush=True)
            logger.info("Stopping background worker...")
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                 print("--- PRINT: Worker task successfully cancelled on stop. ---", file=sys.stdout, flush=True)
                 logger.info("Worker task successfully cancelled.")
            except Exception as e:
                 print(f"--- PRINT: Exception during worker task stop: {e} ---", file=sys.stderr, flush=True)
                 logger.error("Exception during worker task stop:", exc_info=True)
        else:
            print("--- PRINT: AsyncWorker stop() called - Task already stopped or not started ---", file=sys.stdout, flush=True)
        logger.info("Background worker stopped.")
        if not self.queue.empty():
             logger.warning(f"Worker stopped with {self.queue.qsize()} items remaining in queue.")

async_worker = AsyncWorker()
print("--- PRINT: AsyncWorker instance created at module level. ---", file=sys.stdout, flush=True)
logger.info("AsyncWorker instance created at module level.")

@asynccontextmanager
async def lifespan(_: FastAPI):
    print("--- PRINT: FastAPI lifespan startup entered. ---", file=sys.stdout, flush=True)
    logger.info("FastAPI lifespan startup: Starting worker...")
    await async_worker.start()
    print("--- PRINT: FastAPI lifespan startup - Worker start awaited. ---", file=sys.stdout, flush=True)
    yield
    print("--- PRINT: FastAPI lifespan shutdown entered. ---", file=sys.stdout, flush=True)
    logger.info("FastAPI lifespan shutdown: Stopping worker...")
    await async_worker.stop()
    print("--- PRINT: FastAPI lifespan shutdown - Worker stop awaited. ---", file=sys.stdout, flush=True)
    logger.info("FastAPI lifespan shutdown complete.")

router = APIRouter(lifespan=lifespan)



class BuildCommunitiesRequest(BaseModel):
    group_ids: Optional[List[str]] = None

@router.post('/communities/build_specific', status_code=status.HTTP_200_OK, dependencies=[Depends(get_api_key)])
async def trigger_build_specific_communities(
    request: BuildCommunitiesRequest, # Use request body for potentially long list of group_ids
    graphiti: ZepGraphitiDep,
):
    """
    Triggers the community building process.
    Optionally specify group_ids to limit the scope.
    """
    try:
        # The build_communities method returns the created community nodes
        community_nodes = await graphiti.build_communities(group_ids=request.group_ids)
        # You might want to return just a success message or info about created communities
        return Result(message=f'Successfully built {len(community_nodes)} communities.', success=True)
    except Exception as e:
        # Add proper error handling/logging
        logger.error(f"Error building communities: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to build communities: {str(e)}")

@router.post('/communities/build_all', status_code=status.HTTP_200_OK, response_model=Result, dependencies=[Depends(get_api_key)])
async def trigger_build_communities_all(
    graphiti: ZepGraphitiDep,
):
    """
    Triggers the community building process for all group IDs in the graph.
    This can be a long-running operation.
    """
    logger.info("Received request to build communities for all group IDs.")
    try:
        # Call build_communities with group_ids=None to process all groups
        community_nodes = await graphiti.build_communities(group_ids=None) # [cite: uploaded:graphiti_core/graphiti.py, uploaded:graphiti_core/utils/maintenance/community_operations.py]
        msg = f'Successfully built/updated {len(community_nodes)} communities across all group IDs.'
        logger.info(msg)
        return Result(message=msg, success=True)
    except Exception as e:
        logger.error(f"Error building communities: {e}", exc_info=True)
        # Raise HTTPException so the client gets a proper error response
        raise HTTPException(status_code=500, detail=f"Failed to build communities: {str(e)}")


# --- Reverted /messages endpoint to queue individual add_episode calls ---
@router.post('/messages', status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(get_api_key)])
async def add_messages(
    request: AddMessagesRequest,
    graphiti: ZepGraphitiDep, # Inject Graphiti instance
):
    """
    Accepts a batch of messages, queues individual tasks for background
    processing using graphiti.add_episode.
    """
    if not request.messages:
         logger.warning(f"Received empty message list for group_id: {request.group_id}")
         return Result(message='No messages provided in the batch', success=False)

    queued_count = 0
    for m in request.messages:
        try:
            # Define the task function for a single message
            async def process_single_message_task(graphiti_instance: ZepGraphiti, msg: Message, group_id: str):
                # This function runs in the background worker
                # It calls the standard add_episode method
                await graphiti_instance.add_episode(
                    uuid=msg.uuid, # Pass optional UUID
                    group_id=group_id,
                    name=msg.name or f"SlackMsg_{msg.timestamp.isoformat()}",
                    # Format body as expected by add_episode
                    episode_body=f"{msg.role or msg.role_type}: {msg.content}",
                    reference_time=msg.timestamp,
                    source=EpisodeType.message, # Set source type
                    source_description=msg.source_description or f"Slack message in group {group_id}"
                    # Add entity_types=... here if needed globally
                )

            # Create a partial function for the task
            task_func = partial(process_single_message_task, graphiti_instance=graphiti, msg=m, group_id=request.group_id)
            # Put the single message task onto the queue
            await async_worker.queue.put(task_func)
            queued_count += 1
        except Exception as e:
             # Log if queuing itself fails, though unlikely with asyncio.Queue
             logger.error(f"Failed to queue message for group_id {request.group_id} (TS: {m.timestamp}): {e}", exc_info=True)
             # Decide if we should stop queuing the rest of the batch

    logger.info(f"Queued {queued_count}/{len(request.messages)} messages for group_id {request.group_id} for individual processing.")
    return Result(message=f'{queued_count} messages added to processing queue', success=True)


# --- Other endpoints remain the same ---
@router.post('/entity-node', status_code=status.HTTP_201_CREATED, dependencies=[Depends(get_api_key)])
async def add_entity_node(
    request: AddEntityNodeRequest,
    graphiti: ZepGraphitiDep,
):
    # (Code remains the same as previous version)
    try:
        logger.info(f"Adding entity node: Name='{request.name}', UUID='{request.uuid}', GroupID='{request.group_id}'")
        node = await graphiti.save_entity_node(
            uuid=request.uuid,
            group_id=request.group_id,
            name=request.name,
            summary=request.summary,
        )
        return node
    except HTTPException:
         raise
    except Exception as e:
         logger.error(f"Unexpected error in add_entity_node endpoint for Name='{request.name}': {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Unexpected server error adding entity node")

@router.delete('/entity-edge/{uuid}', status_code=status.HTTP_200_OK, dependencies=[Depends(get_api_key)])
async def delete_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    # (Code remains the same as previous version)
     try:
        logger.info(f"Received request to delete entity edge: {uuid}")
        await graphiti.delete_entity_edge(uuid)
        return Result(message='Entity Edge deleted', success=True)
     except HTTPException:
         raise
     except Exception as e:
         logger.error(f"Unexpected error in delete_entity_edge endpoint for {uuid}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Unexpected server error deleting entity edge")

@router.delete('/group/{group_id}', status_code=status.HTTP_200_OK, dependencies=[Depends(get_api_key)])
async def delete_group(group_id: str, graphiti: ZepGraphitiDep):
    # (Code remains the same as previous version)
    try:
        logger.info(f"Received request to delete group: {group_id}")
        await graphiti.delete_group(group_id)
        return Result(message='Group deleted', success=True)
    except HTTPException:
         raise
    except Exception as e:
         logger.error(f"Unexpected error in delete_group endpoint for {group_id}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Unexpected server error deleting group")

@router.delete('/episode/{uuid}', status_code=status.HTTP_200_OK, dependencies=[Depends(get_api_key)])
async def delete_episode(uuid: str, graphiti: ZepGraphitiDep):
    # (Code remains the same as previous version)
    try:
        logger.info(f"Received request to delete episode: {uuid}")
        await graphiti.delete_episodic_node(uuid)
        return Result(message='Episode deleted', success=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in delete_episode endpoint for {uuid}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected server error deleting episode")

@router.post('/clear', status_code=status.HTTP_200_OK, dependencies=[Depends(get_api_key)])
async def clear(
    graphiti: ZepGraphitiDep,
):
    # (Code remains the same as previous version)
    try:
        logger.warning("Received request to clear ALL graph data!")
        await clear_data(graphiti.driver)
        await graphiti.build_indices_and_constraints()
        logger.info("Graph clear operation completed successfully.")
        return Result(message='Graph cleared and indices rebuilt', success=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.critical(f"CRITICAL: Failed during graph clear operation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Critical error during graph clear operation")
    