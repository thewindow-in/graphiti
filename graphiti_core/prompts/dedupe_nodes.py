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

MODIFIED TO CONSIDER LABELS/TYPES
"""

import json
from typing import Any, Optional, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class NodeDuplicate(BaseModel):
    is_duplicate: bool = Field(..., description='true if the NEW NODE represents the same real-world entity or event instance as one of the EXISTING NODES, otherwise false.')
    uuid: Optional[str] = Field(
        None,
        description="If is_duplicate is true, provide the UUID of the EXISTING NODE that it duplicates. Otherwise, null.",
    )
    name: str = Field(
        ...,
        description="The best, most complete name for the entity/event. If duplicating, choose the best name between the new and existing node, or synthesize one. If not duplicating, use the new node's name.",
    )


class Prompt(Protocol):
    node: PromptVersion
    node_list: PromptVersion # No changes needed for node_list prompt


class Versions(TypedDict):
    node: PromptFunction
    node_list: PromptFunction


def node(context: dict[str, Any]) -> list[Message]:
    # <<< CHANGE START >>>
    # Added consideration for node labels/types, especially for dynamic events
    return [
        Message(
            role='system',
            content='You are a helpful assistant that de-duplicates nodes based on conversation context and existing graph data. You must determine if a newly extracted node represents the *exact same* real-world entity or specific event instance as one already existing.',
        ),
        Message(
            role='user',
            content=f"""
        <PREVIOUS MESSAGES>
        {json.dumps([ep for ep in context['previous_episodes']], indent=2)}
        </PREVIOUS MESSAGES>
        <CURRENT MESSAGE>
        {context["episode_content"]}
        </CURRENT MESSAGE>

        <EXISTING NODES>
        {json.dumps(context['existing_nodes'], indent=2)}
        </EXISTING NODES>

        <NEW NODE>
        {json.dumps(context['extracted_nodes'], indent=2)}
        </NEW NODE>

        Task:
        Analyze the NEW NODE in the context of the CURRENT MESSAGE, PREVIOUS MESSAGES, and the list of potentially relevant EXISTING NODES. Determine if the NEW NODE represents the *exact same* real-world entity or *specific event instance* as one of the EXISTING NODES.

        Guidelines:
        1.  **Semantic Match:** Consider names, summaries/attributes, and the conversation context. Nodes can be duplicates even with slightly different names (e.g., "Priya Sharma" vs "Priya S.").
        2.  **Entity Type/Labels:** Pay attention to implicit or explicit types/labels. A "Campaign" node should generally only duplicate another "Campaign" node.
        3.  **Dynamic Events (e.g., TechnicalIssue, PayoutIssue):** Be cautious. A new mention of an "API Outage" might be a *new occurrence* of a similar issue, not a duplicate of a past outage node, unless the context strongly implies it's the *same ongoing* incident being discussed. If it seems like a distinct occurrence, mark `is_duplicate: false`.
        4.  **Name Update:** If `is_duplicate` is true, choose the most complete and accurate name from either the new or existing node, or synthesize the best possible name. If `is_duplicate` is false, use the NEW NODE's name.
        5.  **Output:** Respond with a JSON object containing `is_duplicate` (boolean), `uuid` (string UUID of the matched existing node if duplicate, else null), and `name` (string, the chosen/updated name).

        Respond with a JSON object in the specified format.
        """,
        ),
    ]
    # <<< CHANGE END >>>


def node_list(context: dict[str, Any]) -> list[Message]:
    # No changes needed for the node_list prompt in this step
    return [
        Message(
            role='system',
            content='You are a helpful assistant that de-duplicates nodes from node lists.',
        ),
        Message(
            role='user',
            content=f"""
        Given the following context, deduplicate a list of nodes:

        Nodes:
        {json.dumps(context['nodes'], indent=2)}

        Task:
        1. Group nodes together such that all duplicate nodes are in the same list of uuids
        2. All duplicate uuids should be grouped together in the same list
        3. Also return a new summary that synthesizes the summary into a new short summary

        Guidelines:
        1. Each uuid from the list of nodes should appear EXACTLY once in your response
        2. If a node has no duplicates, it should appear in the response in a list of only one uuid

        Respond with a JSON object in the following format:
        {{
            "nodes": [
                {{
                    "uuids": ["5d643020624c42fa9de13f97b1b3fa39", "node that is a duplicate of 5d643020624c42fa9de13f97b1b3fa39"],
                    "summary": "Brief summary of the node summaries that appear in the list of names."
                }}
            ]
        }}
        """,
        ),
    ]


versions: Versions = {'node': node, 'node_list': node_list}
