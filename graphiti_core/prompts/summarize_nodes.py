# graphiti_core/prompts/summarize_nodes.py
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

MODIFIED TO:
- Add BATCH COMMUNITY SUMMARIZATION.
- Enhance community summary prompt to include RECENT CONTEXT.
"""

import json
from typing import Any, Protocol, TypedDict, List

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class Summary(BaseModel):
    summary: str = Field(
        ...,
        description='Summary containing the important information about the entity. Under 500 words',
    )


class SummaryDescription(BaseModel):
    description: str = Field(..., description='One sentence description of the provided summary')


class CommunitySummaryAndName(BaseModel):
    community_name: str = Field(
        ...,
        description="A concise, descriptive name for the community (e.g., 'Creator Onboarding Issues', 'Q3 Campaign Performance', 'API Development Team'). Under 10 words."
    )
    # --- MODIFICATION START: Allow slightly longer summary ---
    community_summary: str = Field(
        ...,
        description="A brief summary (3-5 sentences) capturing the core theme based on members AND incorporating key highlights or status from recent context."
    )
    # --- MODIFICATION END ---


class Prompt(Protocol):
    summarize_pair: PromptVersion
    summarize_context: PromptVersion
    summary_description: PromptVersion
    summarize_community_batch: PromptVersion


class Versions(TypedDict):
    summarize_pair: PromptFunction
    summarize_context: PromptFunction
    summary_description: PromptFunction
    summarize_community_batch: PromptFunction


def summarize_pair(context: dict[str, Any]) -> list[Message]:
    # No changes needed
    return [
        Message(
            role='system',
            content='You are a helpful assistant that combines two summaries into one concise summary.',
        ),
        Message(
            role='user',
            content=f"""
        Synthesize the information from the following two summaries into a single succinct summary.
        The final summary must be under 500 words.

        Summary 1:
        {json.dumps(context['node_summaries'][0], indent=2)}

        Summary 2:
        {json.dumps(context['node_summaries'][1], indent=2)}
        """,
        ),
    ]


def summarize_context(context: dict[str, Any]) -> list[Message]:
    # No changes needed
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts entity properties and creates summaries from provided text.',
        ),
        Message(
            role='user',
            content=f"""

        <MESSAGES>
        {json.dumps(context.get('previous_episodes', []), indent=2)}
        {json.dumps(context.get('episode_content', ''), indent=2)}
        </MESSAGES>

        Given the above MESSAGES and the following ENTITY name and context, create a summary for the ENTITY.
        Your summary must only use information from the provided MESSAGES and ENTITY CONTEXT.
        Your summary should also only contain information relevant to the provided ENTITY.
        Summaries must be under 500 words.

        In addition, extract any values for the provided entity ATTRIBUTES based on their descriptions.
        If the value of an attribute cannot be found in the current context, set its value to None.
        Do not hallucinate attribute values.

        <ENTITY>
        {context.get('node_name', 'Unknown Entity')}
        </ENTITY>

        <ENTITY CONTEXT>
        {context.get('node_summary', 'No existing summary.')}
        </ENTITY CONTEXT>

        <ATTRIBUTES>
        {json.dumps(context.get('attributes', []), indent=2)}
        </ATTRIBUTES>
        """,
        ),
    ]


def summary_description(context: dict[str, Any]) -> list[Message]:
     # No changes needed
    return [
        Message(
            role='system',
            content='You are a helpful assistant that describes provided content in a single concise sentence.',
        ),
        Message(
            role='user',
            content=f"""
        Create a short one sentence description of the following summary that explains what kind of information is summarized.
        The description should be suitable as a label or title.

        Summary:
        {json.dumps(context.get('summary', ''), indent=2)}
        """,
        ),
    ]


def summarize_community_batch(context: dict[str, Any]) -> list[Message]:
    """
    Generates a prompt to summarize a community based on its members and recent context.
    Expects 'community_members' and 'recent_context' keys in the context dict.
    """
    # Prepare member info string
    members_info = ""
    for member in context.get('community_members', []):
        members_info += f"- Node Name: {member.get('name', 'N/A')}\n"
        if member.get('summary'):
            members_info += f"  Summary: {member['summary'][:200]}...\n"
        members_info += "\n"
    if not members_info:
        members_info = "No specific member information provided."

    # --- MODIFICATION START: Prepare recent context string ---
    recent_context_info = ""
    recent_ctx = context.get('recent_context')
    if isinstance(recent_ctx, list) and recent_ctx:
        # Format based on what you pass (e.g., list of facts, summaries, snippets)
        # Example: Assuming recent_context is a list of recent fact strings
        recent_context_info = "\n".join([f"- {fact}" for fact in recent_ctx])
    if not recent_context_info:
        recent_context_info = "No specific recent context provided."
    # --- MODIFICATION END ---

    return [
        Message(
            role='system',
            content="""You are an AI assistant skilled at analyzing groups of related entities (nodes in a knowledge graph) and synthesizing their collective theme and recent activity into a concise community name and summary.

You will be given:
1.  A list of nodes belonging to a detected community, potentially with their own summaries.
2.  A list of recent facts, events, or messages related to members of this community.

Your task is to:
1.  Determine the central topic, project, team, issue type, or common characteristic shared by the majority of the community members.
2.  Generate a short, descriptive `community_name` (under 10 words) that reflects this central theme.
3.  Generate a brief `community_summary` (3-5 sentences) that **first explains the core theme** based on the members, and **then incorporates key highlights, status updates, or significant recent events** mentioned in the recent context provided. Balance the static theme with the dynamic recent information."""
        ),
        Message(
            role='user',
            content=f"""
            Analyze the following community members and recent context. Based on this information, determine the central theme and recent highlights, then generate a concise community name and summary.

            <COMMUNITY MEMBERS>
            {members_info}
            </COMMUNITY MEMBERS>

            <RECENT CONTEXT (e.g., recent facts/events related to members)>
            {recent_context_info}
            </RECENT CONTEXT>

            Provide the output as a JSON object with keys "community_name" and "community_summary".
            Ensure the summary explains the core theme AND includes highlights from the recent context.
            """,
        ),
    ]

versions: Versions = {
    'summarize_pair': summarize_pair,
    'summarize_context': summarize_context,
    'summary_description': summary_description,
    'summarize_community_batch': summarize_community_batch,
}
