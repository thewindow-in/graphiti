# ingest.py (Assuming this is graph_service/routers/ingest.py)
# MODIFIED TO:
# - Run multiple concurrent worker tasks.
# - Use a bounded asyncio.Queue.
# - Return 503 Service Unavailable if the queue is full.

import asyncio
import logging
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, List, Optional, cast, Set, Tuple # Added Set, Tuple, datetime
from datetime import datetime # Added datetime explicitly
import sys
import traceback

from pydantic import BaseModel

from fastapi.params import Depends
# Assuming auth logic is defined here or imported correctly
# Replace with your actual auth import if different
from graph_service.auth import get_api_key

from fastapi import APIRouter, FastAPI, HTTPException, status

# Assuming graphiti_core and DTOs are installed/accessible
from graphiti_core.nodes import EntityNode, EpisodeType # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data # type: ignore

# Assuming DTOs are in this path, adjust if needed
from graph_service.dto import AddEntityNodeRequest, AddMessagesRequest, Result, Message

# Import ZepGraphiti directly for type hinting the instance
# Assuming zep_graphiti is defined here or imported correctly
from graph_service.zep_graphiti import ZepGraphiti, ZepGraphitiDep

logger = logging.getLogger(__name__)

# --- Configuration ---
QUEUE_MAX_SIZE = 20000 # Max items in the processing queue
NUM_WORKERS = 3 # Number of concurrent workers processing the queue


# --- AsyncWorker with Multiple Tasks ---
class AsyncWorker:
    """Manages a pool of asynchronous worker tasks processing jobs from a queue."""
    def __init__(self, num_workers: int):
        self.queue: asyncio.Queue[partial[Any]] = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
        self.worker_tasks: Set[asyncio.Task] = set()
        self.num_workers = num_workers
        self._stop_event = asyncio.Event() # Event to signal workers to stop gracefully

    async def worker(self, worker_id: int):
        """The coroutine run by each worker task."""
        logger.info(f"--- Worker {worker_id}: Started ---")
        while not self._stop_event.is_set():
            job_details_str = "N/A"
            job_func: Optional[partial[Any]] = None
            try:
                # Wait for an item with a timeout so the stop_event can be checked periodically
                job_func = await asyncio.wait_for(self.queue.get(), timeout=1.0)

                # Log reception and queue size
                qsize = self.queue.qsize() # Get size after taking item
                logger.info(f"--- Worker {worker_id}: Received job (Queue approx size: {qsize}) ---")

                # Extract details for logging more reliably
                if hasattr(job_func, 'func') and hasattr(job_func, 'keywords'):
                    job_keywords = job_func.keywords
                    group_id = job_keywords.get('group_id', 'N/A')
                    msg_ts_obj = job_keywords.get('reference_time') # Expecting datetime
                    msg_ts_str = msg_ts_obj.isoformat() if isinstance(msg_ts_obj, datetime) else 'N/A'
                    channel = job_keywords.get('source_channel_name') or job_keywords.get('source_channel_id') or 'N/A'
                    is_human = job_keywords.get('is_human_message', 'N/A')
                    job_details_str = (
                        f"GroupID: {group_id}, "
                        f"MsgTS: {msg_ts_str}, "
                        f"Channel: {channel}, "
                        f"IsHuman: {is_human}"
                    )
                else:
                     job_details_str = "Job details unavailable (unexpected task format)"

                logger.info(f"--- Worker {worker_id}: Processing job. Details: {job_details_str} ---")
                # Execute the actual task (e.g., graphiti.add_episode)
                await job_func()
                logger.info(f"--- Worker {worker_id}: Successfully processed job. Details: {job_details_str} ---")

            except asyncio.TimeoutError:
                # No job received within timeout, loop again to check stop_event
                continue
            except asyncio.CancelledError:
                logger.info(f"--- Worker {worker_id}: Cancelled ---")
                # If cancelled while holding a job, potentially requeue or log failure
                if job_func:
                     logger.warning(f"--- Worker {worker_id}: Cancelled while holding job: {job_details_str}")
                     # Decide on handling: requeue? log as failed?
                break # Exit loop if cancelled directly
            except Exception:
                # Log exceptions occurring within the job_func (add_episode)
                logger.error(
                    f"!!! Worker {worker_id}: Error processing background job. Details: {job_details_str} !!!",
                    exc_info=True
                )
                # Depending on the error, you might want different handling
                # For now, it just logs and the task_done is called in finally
            finally:
                 # Ensure task_done is called even if errors occur within the job
                 if job_func:
                    try:
                        self.queue.task_done()
                        logger.info(f"--- Worker {worker_id}: Called task_done. Details: {job_details_str} ---")
                    except ValueError:
                         # Can happen if task_done called too many times (e.g., during cancellation)
                         logger.warning(f"--- Worker {worker_id}: task_done() called too many times?. Details: {job_details_str} ---")
                         pass # Ignore error

        logger.info(f"--- Worker {worker_id}: Stopped ---")

    async def start(self):
        """Starts the pool of worker tasks."""
        if self.worker_tasks:
             logger.warning("Worker tasks seem to be already running.")
             return

        self._stop_event.clear()
        logger.info(f"Starting {self.num_workers} background worker tasks...")
        for i in range(self.num_workers):
             task = asyncio.create_task(self.worker(worker_id=i+1))
             self.worker_tasks.add(task)
             # Remove task from set when it's done to prevent memory leaks
             task.add_done_callback(self.worker_tasks.discard)
        logger.info(f"{len(self.worker_tasks)} worker tasks started.")

    async def stop(self):
        """Signals workers to stop and waits for them to finish."""
        if not self.worker_tasks:
            logger.info("No worker tasks running to stop.")
            return

        logger.info(f"Stopping {len(self.worker_tasks)} background worker tasks...")
        self._stop_event.set() # Signal workers to stop consuming new items

        # Wait for the queue to be fully processed by running workers
        logger.info("Waiting for queue to empty...")
        await self.queue.join() # Wait until all task_done() calls are received
        logger.info("Queue empty.")

        # Now that the queue is empty, cancel any remaining worker tasks
        # (they might be stuck in await queue.get() or processing a long task)
        tasks_to_cancel = list(self.worker_tasks) # Create copy for iteration
        if tasks_to_cancel:
            logger.info(f"Cancelling {len(tasks_to_cancel)} worker tasks...")
            for task in tasks_to_cancel:
                task.cancel()
            # Wait for cancellations to complete
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            logger.info("Worker tasks cancelled.")

        self.worker_tasks.clear()
        logger.info("All worker tasks stopped.")

        if not self.queue.empty(): # Should be empty after join, but check just in case
             logger.warning(f"Worker stopped with {self.queue.qsize()} items remaining in queue (unexpected).")


# Instantiate the worker pool globally
async_worker = AsyncWorker(num_workers=NUM_WORKERS)
logger.info(f"AsyncWorker instance created for {NUM_WORKERS} workers with queue size {QUEUE_MAX_SIZE}.")

# FastAPI Lifespan context manager
@asynccontextmanager
async def lifespan(_: FastAPI):
    """Handles startup and shutdown of the background worker pool."""
    logger.info("FastAPI lifespan startup: Starting workers...")
    await async_worker.start()
    logger.info("FastAPI lifespan startup: Workers started.")
    yield
    logger.info("FastAPI lifespan shutdown: Stopping workers...")
    await async_worker.stop()
    logger.info("FastAPI lifespan shutdown complete.")

# Create the FastAPI router with the lifespan manager
router = APIRouter(lifespan=lifespan)


# --- API Endpoints ---

# Define request model for community building
class BuildCommunitiesRequest(BaseModel):
    group_ids: Optional[List[str]] = None

@router.post('/communities/build_specific', status_code=status.HTTP_200_OK, dependencies=[Depends(get_api_key)])
async def trigger_build_specific_communities(
    request: BuildCommunitiesRequest,
    graphiti: ZepGraphitiDep,
):
    """Triggers community building for specified group IDs."""
    try:
        logger.info(f"Received request to build communities for group_ids: {request.group_ids}")
        # Ensure graphiti dependency is resolved correctly
        if not isinstance(graphiti, ZepGraphiti):
             raise TypeError("Graphiti dependency not resolved correctly")
        community_nodes = await graphiti.build_communities(group_ids=request.group_ids)
        return Result(message=f'Successfully built {len(community_nodes)} communities.', success=True)
    except Exception as e:
        logger.error(f"Error building communities: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to build communities: {str(e)}")

@router.post('/communities/build_all', status_code=status.HTTP_200_OK, response_model=Result, dependencies=[Depends(get_api_key)])
async def trigger_build_communities_all(
    graphiti: ZepGraphitiDep,
):
    """Triggers community building for all group IDs."""
    logger.info("Received request to build communities for all group IDs.")
    try:
        if not isinstance(graphiti, ZepGraphiti):
             raise TypeError("Graphiti dependency not resolved correctly")
        community_nodes = await graphiti.build_communities(group_ids=None)
        msg = f'Successfully built/updated {len(community_nodes)} communities across all group IDs.'
        logger.info(msg)
        return Result(message=msg, success=True)
    except Exception as e:
        logger.error(f"Error building communities: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to build communities: {str(e)}")


@router.post('/messages', status_code=status.HTTP_202_ACCEPTED, response_model=Result, dependencies=[Depends(get_api_key)])
async def add_messages(
    request: AddMessagesRequest,
    graphiti: ZepGraphitiDep,
):
    """
    Accepts a batch of messages. If the processing queue is full, returns 503.
    Otherwise, queues individual tasks for background processing by multiple workers.
    """
    if not request.messages:
         logger.info(f"Received empty message list for group_id: {request.group_id}")
         return Result(message='No messages provided in the batch', success=True) # Still success, just nothing to do

    # Check if queue is full BEFORE processing batch
    # Use qsize() for a more dynamic check than full()
    current_qsize = async_worker.queue.qsize()
    if current_qsize + len(request.messages) > QUEUE_MAX_SIZE:
        logger.warning(f"Processing queue is nearly full (size {current_qsize}/{QUEUE_MAX_SIZE}). Rejecting batch of {len(request.messages)} for group {request.group_id}.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Processing queue is full (size {current_qsize}). Please try again later."
        )

    queued_count = 0
    for m in request.messages:
        # Extract necessary info from Message DTO 'm'
        # These might need adjustment based on fields actually sent by client
        channel_id = getattr(m, 'source_channel_id', None) # Example: Use field if sent
        channel_name = getattr(m, 'source_channel_name', None) # Example: Use field if sent
        is_human = m.role_type == 'user' # Determine from required DTO field

        try:
            # Create the partial function for the background task
            task_func = partial(
                graphiti.add_episode, # Target method
                # Pass arguments explicitly by keyword for clarity
                uuid=m.uuid,
                group_id=request.group_id,
                name=m.name or f"SlackMsg_{m.timestamp.isoformat()}", # Use DTO name or generate
                episode_body=f"{m.role or m.role_type}: {m.content}", # Use DTO role or role_type
                reference_time=m.timestamp, # Use DTO timestamp
                source=EpisodeType.message, # Assuming message type
                source_description=m.source_description or f"Slack msg in {channel_name or channel_id or 'unknown'}", # Use DTO desc or generate
                source_channel_id=channel_id,
                source_channel_name=channel_name,
                is_human_message=is_human
                # entity_types=None, # Pass if needed
                # update_communities=False # Typically false during bulk ingest
            )

            # Put the task onto the shared queue
            async_worker.queue.put_nowait(task_func)
            queued_count += 1

        except asyncio.QueueFull:
             # Should be rare due to the check above, but handle defensively
             logger.error(f"Queue became full unexpectedly while adding tasks for group {request.group_id}. Processed {queued_count} messages from batch.")
             # Return 503, indicating partial acceptance is not supported here
             raise HTTPException(
                 status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                 detail=f"Processing queue filled up during request. Please try again later."
             )
        except Exception as e:
             # Log if queuing itself fails for other reasons
             logger.error(f"Failed to queue message for group_id {request.group_id} (TS: {m.timestamp}): {e}", exc_info=True)
             # Indicate internal error if queuing fails mid-batch
             raise HTTPException(status_code=500, detail="Internal error queuing message.")


    logger.info(f"Queued {queued_count}/{len(request.messages)} messages for group_id {request.group_id}.")
    # Return 202 Accepted as the tasks are queued, not completed
    return Result(message=f'{queued_count} messages accepted into processing queue', success=True)


# --- Other Management Endpoints ---
@router.post('/entity-node', status_code=status.HTTP_201_CREATED, response_model=EntityNode, dependencies=[Depends(get_api_key)])
async def add_entity_node(request: AddEntityNodeRequest, graphiti: ZepGraphitiDep):
    """Adds or updates a single entity node."""
    try:
        logger.info(f"Adding/Updating entity node: Name='{request.name}', UUID='{request.uuid}', GroupID='{request.group_id}'")
        # Ensure graphiti dependency is resolved correctly
        if not isinstance(graphiti, ZepGraphiti):
             raise TypeError("Graphiti dependency not resolved correctly")
        # Assuming save_entity_node handles created_at appropriately for manual adds
        node = await graphiti.save_entity_node(
            uuid=request.uuid,
            group_id=request.group_id,
            name=request.name,
            summary=request.summary,
        )
        return node # Return the created/updated node object
    except HTTPException:
         raise
    except Exception as e:
         logger.error(f"Unexpected error in add_entity_node: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Unexpected server error adding entity node")

@router.delete('/entity-edge/{uuid}', status_code=status.HTTP_200_OK, response_model=Result, dependencies=[Depends(get_api_key)])
async def delete_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    """Deletes a specific entity edge by UUID."""
    try:
        logger.info(f"Received request to delete entity edge: {uuid}")
        if not isinstance(graphiti, ZepGraphiti): raise TypeError("Graphiti dependency error")
        await graphiti.delete_entity_edge(uuid)
        return Result(message='Entity Edge deleted', success=True)
    except HTTPException:
         raise # Propagate 404 if not found
    except Exception as e:
         logger.error(f"Unexpected error in delete_entity_edge: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="Unexpected server error deleting entity edge")

@router.delete('/group/{group_id}', status_code=status.HTTP_200_OK, response_model=Result, dependencies=[Depends(get_api_key)])
async def delete_group(group_id: str, graphiti: ZepGraphitiDep):
    """Deletes all nodes and edges associated with a specific group ID."""
    try:
        logger.warning(f"Received request to delete ALL data for group: {group_id}")
        if not isinstance(graphiti, ZepGraphiti): raise TypeError("Graphiti dependency error")
        await graphiti.delete_group(group_id) # Ensure this method exists and works
        return Result(message=f'All data for group {group_id} deleted', success=True)
    except HTTPException:
         raise
    except Exception as e:
         logger.error(f"Unexpected error in delete_group for {group_id}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Unexpected server error deleting group {group_id}")

@router.delete('/episode/{uuid}', status_code=status.HTTP_200_OK, response_model=Result, dependencies=[Depends(get_api_key)])
async def delete_episode(uuid: str, graphiti: ZepGraphitiDep):
    """Deletes a specific episode node and attempts cleanup."""
    try:
        logger.info(f"Received request to delete episode: {uuid}")
        if not isinstance(graphiti, ZepGraphiti): raise TypeError("Graphiti dependency error")
        # Ensure delete_episodic_node maps to the correct graphiti method (e.g., remove_episode)
        await graphiti.delete_episodic_node(uuid)
        return Result(message='Episode deleted', success=True)
    except HTTPException:
        raise # Propagate 404
    except Exception as e:
        logger.error(f"Unexpected error in delete_episode for {uuid}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected server error deleting episode")

@router.post('/clear', status_code=status.HTTP_200_OK, response_model=Result, dependencies=[Depends(get_api_key)])
async def clear(graphiti: ZepGraphitiDep):
    """Clears ALL data from the graph and rebuilds indices."""
    try:
        logger.critical("Received request to clear ALL graph data!")
        if not isinstance(graphiti, ZepGraphiti): raise TypeError("Graphiti dependency error")
        await clear_data(graphiti.driver) # Call the function directly
        await graphiti.build_indices_and_constraints()
        logger.info("Graph clear operation completed successfully.")
        return Result(message='Graph cleared and indices rebuilt', success=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.critical(f"CRITICAL: Failed during graph clear operation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Critical error during graph clear operation")

