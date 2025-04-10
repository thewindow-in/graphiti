"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

MODIFIED TO HANDLE DYNAMIC/RECURRING EVENT NODES
"""

import logging
from time import time
from typing import List, Dict, Tuple, Optional, Set # Added Optional, Set, Tuple

import pydantic
from pydantic import BaseModel

# <<< CHANGE START >>>
# Import EntityEdge to create PREVIOUS_INSTANCE edges
from graphiti_core.edges import EntityEdge
# <<< CHANGE END >>>
from graphiti_core.helpers import MAX_REFLEXION_ITERATIONS, semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_nodes import NodeDuplicate
from graphiti_core.prompts.extract_nodes import (
    EntityClassification,
    ExtractedNodes,
    MissedEntities,
)
from graphiti_core.prompts.summarize_nodes import Summary
from graphiti_core.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)

# <<< CHANGE START >>>
# Define labels that represent dynamic, potentially recurring events
# Customize this list based on the specific events you want to track as distinct instances
DYNAMIC_EVENT_LABELS: Set[str] = {
    "TechnicalIssue",
    "UserBlocker",
    "PayoutIssue",
    "PerformanceDip",
    "Outage", # Add more specific event types relevant to Wishlink
}
# Define the relationship type for linking recurring events
PREVIOUS_INSTANCE_RELATION_TYPE = "PREVIOUS_INSTANCE"
# <<< CHANGE END >>>


async def extract_message_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    custom_prompt='',
) -> list[str]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'episode_timestamp': episode.valid_at.isoformat(),
        'previous_episodes': [ep.content for ep in previous_episodes],
        'custom_prompt': custom_prompt,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_message(context), response_model=ExtractedNodes
    )
    extracted_node_names = llm_response.get('extracted_node_names', [])
    return extracted_node_names


async def extract_text_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    custom_prompt='',
) -> list[str]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'episode_timestamp': episode.valid_at.isoformat(),
        'previous_episodes': [ep.content for ep in previous_episodes],
        'custom_prompt': custom_prompt,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_text(context), ExtractedNodes
    )
    extracted_node_names = llm_response.get('extracted_node_names', [])
    return extracted_node_names


async def extract_json_nodes(
    llm_client: LLMClient, episode: EpisodicNode, custom_prompt=''
) -> list[str]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'episode_timestamp': episode.valid_at.isoformat(),
        'source_description': episode.source_description,
        'custom_prompt': custom_prompt,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_json(context), ExtractedNodes
    )
    extracted_node_names = llm_response.get('extracted_node_names', [])
    return extracted_node_names


async def extract_nodes_reflexion(
    llm_client: LLMClient,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    node_names: list[str],
) -> list[str]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'previous_episodes': [ep.content for ep in previous_episodes],
        'extracted_entities': node_names,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.reflexion(context), MissedEntities
    )
    missed_entities = llm_response.get('missed_entities', [])

    return missed_entities


async def extract_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode,
    previous_episodes: list[EpisodicNode],
    entity_types: dict[str, BaseModel] | None = None,
) -> list[EntityNode]:
    start = time()
    extracted_node_names: list[str] = []
    custom_prompt = ''
    entities_missed = True
    reflexion_iterations = 0
    while entities_missed and reflexion_iterations < MAX_REFLEXION_ITERATIONS:
        if episode.source == EpisodeType.message:
            extracted_node_names = await extract_message_nodes(
                llm_client, episode, previous_episodes, custom_prompt
            )
        elif episode.source == EpisodeType.text:
            extracted_node_names = await extract_text_nodes(
                llm_client, episode, previous_episodes, custom_prompt
            )
        elif episode.source == EpisodeType.json:
            extracted_node_names = await extract_json_nodes(llm_client, episode, custom_prompt)

        reflexion_iterations += 1
        if reflexion_iterations < MAX_REFLEXION_ITERATIONS:
            # Check for missed entities only if we haven't reached the max iterations
            missing_entities = await extract_nodes_reflexion(
                llm_client, episode, previous_episodes, extracted_node_names
            )
            entities_missed = len(missing_entities) != 0
            if entities_missed:
                custom_prompt = 'The following entities were missed in a previous extraction: '
                custom_prompt += ', '.join(missing_entities) # More concise joining
        else:
             entities_missed = False # Stop loop if max iterations reached


    node_classification_context = {
        'episode_content': episode.content,
        'previous_episodes': [ep.content for ep in previous_episodes],
        'extracted_entities': extracted_node_names,
        'entity_types': {
            type_name: values.model_json_schema().get('description', '') # Added default empty string
            for type_name, values in entity_types.items()
        }
        if entity_types is not None
        else {},
    }

    node_classifications: dict[str, str | None] = {}

    if entity_types is not None and extracted_node_names: # Check if there are nodes to classify
        try:
            llm_response = await llm_client.generate_response(
                prompt_library.extract_nodes.classify_nodes(node_classification_context),
                response_model=EntityClassification,
            )
            # Ensure llm_response is treated as a dict
            response_data = llm_response if isinstance(llm_response, dict) else {}
            entity_classifications = response_data.get('entity_classifications', [])
            # Ensure entity_classification is a dict before accessing keys
            node_classifications.update(
                {
                    entity_classification.get('name'): entity_classification.get('entity_type')
                    for entity_classification in entity_classifications
                    if isinstance(entity_classification, dict) and entity_classification.get('name') # Check name exists
                }
            )
        except Exception as e:
            logger.error(f"Error during node classification: {e}", exc_info=True)
            # Continue without classifications if LLM fails

    end = time()
    logger.debug(f'Extracted new nodes: {extracted_node_names} in {(end - start) * 1000} ms')

    # Convert the extracted data into EntityNode objects
    new_nodes = []
    current_time = utc_now() # Get time once
    for name in extracted_node_names:
        entity_type = node_classifications.get(name)
        # Normalize type check
        if entity_types is None or entity_type not in entity_types:
            entity_type = None # Set to None if not in provided types

        # Construct labels, ensuring 'Entity' is always present
        labels = ['Entity']
        if entity_type and entity_type not in ('None', 'null'):
             labels.append(entity_type)
        # Remove potential duplicates just in case
        labels = sorted(list(set(labels)))


        new_node = EntityNode(
            name=name,
            group_id=episode.group_id,
            labels=labels,
            summary='', # Summary is generated later in resolve step
            created_at=current_time, # Use consistent timestamp
        )
        new_nodes.append(new_node)
        logger.debug(f'Created new node: {new_node.name} (UUID: {new_node.uuid}, Labels: {labels})')

    return new_nodes


# <<< CHANGE START >>>
# Function signature changed to return an optional EntityEdge
async def resolve_extracted_node(
    llm_client: LLMClient,
    extracted_node: EntityNode,
    existing_nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, BaseModel] | None = None,
) -> tuple[EntityNode, dict[str, str], Optional[EntityEdge]]:
# <<< CHANGE END >>>
    start = time()
    previous_instance_edge: Optional[EntityEdge] = None # Initialize potential edge
    uuid_map: dict[str, str] = {} # Initialize uuid_map locally

    # Prepare context for LLM deduplication check
    existing_nodes_context = [
        {'uuid': node.uuid, 'name': node.name, 'labels': node.labels, 'attributes': node.attributes}
        for node in existing_nodes
    ]
    extracted_node_context = {
        'uuid': extracted_node.uuid,
        'name': extracted_node.name,
        'labels': extracted_node.labels,
        'summary': extracted_node.summary, # Summary is empty here, generated below
    }
    dedupe_context = {
        'existing_nodes': existing_nodes_context,
        'extracted_nodes': extracted_node_context, # Corrected key
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': [ep.content for ep in previous_episodes]
        if previous_episodes is not None
        else [],
    }

    # Prepare context for summary and attribute generation
    summary_context = {
        'node_name': extracted_node.name,
        'node_summary': extracted_node.summary, # Will be empty initially
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': [ep.content for ep in previous_episodes]
        if previous_episodes is not None
        else [],
        'attributes': [], # Populate based on entity_types
    }

    entity_type_classes: tuple[BaseModel, ...] = tuple()
    if entity_types is not None:
        entity_type_classes = entity_type_classes + tuple(
            filter(
                lambda x: x is not None,
                [entity_types.get(label) for label in extracted_node.labels if label != 'Entity'] # Get types for node's labels
            )
        )

    # Add attribute names from matched entity types to the summary context
    for entity_type_class in entity_type_classes:
        if hasattr(entity_type_class, 'model_fields'):
             summary_context['attributes'].extend(list(entity_type_class.model_fields.keys())) # type: ignore

    # Dynamically create the response model for summary/attributes
    # Ensure Summary is always included
    base_classes_for_model = entity_type_classes + (Summary,)
    # Filter out potential None values if entity_types lookup failed
    valid_base_classes = tuple(cls for cls in base_classes_for_model if cls is not None)

    entity_attributes_model: Type[BaseModel] = pydantic.create_model( # type: ignore
        f'Generated_{extracted_node.uuid[:8]}', # More unique name
        __base__=valid_base_classes,
    )


    # Run LLM calls concurrently
    dedupe_response, node_attributes_response = await semaphore_gather(
        llm_client.generate_response(
            prompt_library.dedupe_nodes.node(dedupe_context), response_model=NodeDuplicate
        ),
        llm_client.generate_response(
            prompt_library.summarize_nodes.summarize_context(summary_context),
            response_model=entity_attributes_model,
        ),
    )

    # Update extracted node with generated summary and attributes
    # Ensure responses are dicts before accessing
    attributes_data = node_attributes_response if isinstance(node_attributes_response, dict) else {}
    extracted_node.summary = attributes_data.get('summary', extracted_node.summary) # Keep original if summary fails
    # Filter out the 'summary' key before updating attributes
    attributes_to_update = {k: v for k, v in attributes_data.items() if k != 'summary'}
    extracted_node.attributes.update(attributes_to_update)


    # Process deduplication response
    dedupe_data = dedupe_response if isinstance(dedupe_response, dict) else {}
    is_duplicate: bool = dedupe_data.get('is_duplicate', False)
    duplicate_uuid: str | None = dedupe_data.get('uuid', None)
    resolved_name = dedupe_data.get('name', extracted_node.name) # Default to extracted name

    node_to_return = extracted_node # Default to returning the (updated) extracted node
    node_to_return.name = resolved_name # Update name based on dedupe response

    # <<< CHANGE START >>>
    # Check if it's a dynamic event AND flagged as a duplicate
    is_dynamic_event = any(label in DYNAMIC_EVENT_LABELS for label in extracted_node.labels)

    if is_duplicate and duplicate_uuid:
        # Find the existing node
        existing_node: Optional[EntityNode] = None
        for n in existing_nodes:
            if n.uuid == duplicate_uuid:
                existing_node = n
                break

        if existing_node:
            if is_dynamic_event:
                # It's a recurring dynamic event. Keep the new node, link to previous.
                logger.debug(f"Dynamic event recurrence detected: New node '{node_to_return.name}' ({node_to_return.uuid}) is recurrence of existing '{existing_node.name}' ({existing_node.uuid}). Creating PREVIOUS_INSTANCE edge.")
                # Create the edge linking new -> old
                previous_instance_edge = EntityEdge(
                    source_node_uuid=node_to_return.uuid, # New node is the source
                    target_node_uuid=existing_node.uuid, # Old node is the target
                    name=PREVIOUS_INSTANCE_RELATION_TYPE,
                    fact=f"{node_to_return.name} is a recurrence of a previous event {existing_node.name}",
                    group_id=node_to_return.group_id,
                    created_at=utc_now(),
                    episodes=episode.episodes if episode else [], # Link to current episode
                    # No embeddings needed for this structural edge by default
                )
                # We still return the NEW node, but also the edge to be saved.
                # uuid_map remains empty for this specific extracted_node UUID
            else:
                # It's a static entity duplicate. Merge summaries, update map, return existing.
                logger.debug(f"Static entity duplicate found: Merging '{extracted_node.name}' ({extracted_node.uuid}) into '{existing_node.name}' ({existing_node.uuid}).")
                # Merge summaries (optional, could use LLM again or simple concatenation)
                # For simplicity, let's prioritize the existing summary if it's longer/non-empty
                if not existing_node.summary or len(extracted_node.summary) > len(existing_node.summary):
                     existing_node.summary = extracted_node.summary # Or use LLM summarize_pair

                # Merge attributes (simple update, new values overwrite old)
                existing_node.attributes.update(extracted_node.attributes)

                existing_node.name = resolved_name # Use the name chosen by the LLM
                node_to_return = existing_node # Return the existing node
                uuid_map[extracted_node.uuid] = existing_node.uuid # Map old UUID to existing one
        else:
             logger.warning(f"Deduplication flagged node {extracted_node.uuid} as duplicate of {duplicate_uuid}, but existing node not found in provided list.")
             # Proceed as if not a duplicate if existing node isn't found

    # If not a duplicate, or if it was dynamic, uuid_map remains empty for this node's original UUID
    # and node_to_return is the (potentially updated) extracted_node.

    # <<< CHANGE END >>>

    end = time()
    log_msg = f"Resolved node: Extracted='{extracted_node.name}' ({extracted_node.uuid}) -> Final='{node_to_return.name}' ({node_to_return.uuid})"
    if is_duplicate and not is_dynamic_event:
        log_msg += f" (Merged into {duplicate_uuid})"
    elif previous_instance_edge:
         log_msg += f" (Dynamic Recurrence detected, created PREVIOUS_INSTANCE edge to {duplicate_uuid})"
    log_msg += f" in {(end - start) * 1000:.2f} ms"
    logger.debug(log_msg)

    # Return the final node (either new or existing), the map, and the optional new edge
    return node_to_return, uuid_map, previous_instance_edge


# <<< CHANGE START >>>
# Function signature changed to return list of new edges as well
async def resolve_extracted_nodes(
    llm_client: LLMClient,
    extracted_nodes: list[EntityNode],
    existing_nodes_lists: list[list[EntityNode]],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, BaseModel] | None = None,
) -> tuple[list[EntityNode], dict[str, str], list[EntityEdge]]:
# <<< CHANGE END >>>

    uuid_map: dict[str, str] = {}
    resolved_nodes: list[EntityNode] = []
    # <<< CHANGE START >>>
    # Collect newly created edges (e.g., PREVIOUS_INSTANCE)
    new_edges: list[EntityEdge] = []
    # <<< CHANGE END >>>

    # Run resolution for each node concurrently
    results: list[tuple[EntityNode, dict[str, str], Optional[EntityEdge]]] = list(
        await semaphore_gather(
            *[
                resolve_extracted_node(
                    llm_client,
                    extracted_node,
                    existing_nodes,
                    episode,
                    previous_episodes,
                    entity_types,
                )
                for extracted_node, existing_nodes in zip(extracted_nodes, existing_nodes_lists)
            ]
        )
    )

    # Process results
    for node, partial_uuid_map, potential_new_edge in results:
        uuid_map.update(partial_uuid_map)
        resolved_nodes.append(node)
        # <<< CHANGE START >>>
        # Add any newly created edge to the list
        if potential_new_edge:
            new_edges.append(potential_new_edge)
        # <<< CHANGE END >>>

    # <<< CHANGE START >>>
    # Return resolved nodes, the final map, and the list of new edges
    return resolved_nodes, uuid_map, new_edges
    # <<< CHANGE END >>>


async def dedupe_node_list(
    llm_client: LLMClient,
    nodes: list[EntityNode],
) -> tuple[list[EntityNode], dict[str, str]]:
    # This function likely operates on a larger scale for maintenance,
    # not typically within the single episode ingestion flow.
    # No changes needed here for dynamic event handling logic introduced above.
    start = time()

    # build node map
    node_map = {}
    for node in nodes:
        node_map[node.uuid] = node

    # Prepare context for LLM
    nodes_context = [
        {'uuid': node.uuid, 'name': node.name, 'summary': node.summary} for node in nodes
    ]

    context = {
        'nodes': nodes_context,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_nodes.node_list(context)
    )

    # Ensure llm_response is a dict
    response_data = llm_response if isinstance(llm_response, dict) else {}
    nodes_data = response_data.get('nodes', [])


    end = time()
    logger.debug(f'Deduplicated nodes list: {len(nodes_data)} groups found in {(end - start) * 1000} ms')

    # Get full node data
    unique_nodes = []
    uuid_map: dict[str, str] = {}
    for node_data in nodes_data:
         # Ensure node_data is a dict and has 'uuids'
         if not isinstance(node_data, dict) or 'uuids' not in node_data or not node_data['uuids']:
             logger.warning(f"Skipping invalid node group data: {node_data}")
             continue

         first_uuid = node_data['uuids'][0]
         node_instance: EntityNode | None = node_map.get(first_uuid)

         if node_instance is None:
            logger.warning(f'Node {first_uuid} from dedupe group not found in initial node map')
            continue

         # Update summary if provided in the response
         if 'summary' in node_data:
             node_instance.summary = node_data['summary']
         unique_nodes.append(node_instance)

         # Create mappings for the duplicates in this group
         for uuid_to_map in node_data['uuids'][1:]:
            if uuid_to_map != first_uuid: # Avoid self-mapping
                 uuid_map[uuid_to_map] = first_uuid

    return unique_nodes, uuid_map
