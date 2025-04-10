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

MODIFIED TO ADD BATCH COMMUNITY SUMMARIZATION
"""

import json
from typing import Any, Protocol, TypedDict, List # Added List

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class Summary(BaseModel):
    summary: str = Field(
        ...,
        description='Summary containing the important information about the entity. Under 500 words',
    )


class SummaryDescription(BaseModel):
    description: str = Field(..., description='One sentence description of the provided summary')

# <<< CHANGE START >>>
# New response model for batch community summarization and naming
class CommunitySummaryAndName(BaseModel):
    community_name: str = Field(
        ...,
        description="A concise, descriptive name for the community (e.g., 'Creator Onboarding Issues', 'Q3 Campaign Performance', 'API Development Team'). Under 10 words."
    )
    community_summary: str = Field(
        ...,
        description="A brief summary (2-3 sentences) capturing the core theme or purpose of the community based on its members."
    )
# <<< CHANGE END >>>


class Prompt(Protocol):
    summarize_pair: PromptVersion
    summarize_context: PromptVersion
    summary_description: PromptVersion
    # <<< CHANGE START >>>
    summarize_community_batch: PromptVersion # Add new prompt function
    # <<< CHANGE END >>>


class Versions(TypedDict):
    summarize_pair: PromptFunction
    summarize_context: PromptFunction
    summary_description: PromptFunction
    # <<< CHANGE START >>>
    summarize_community_batch: PromptFunction # Add new prompt function
    # <<< CHANGE END >>>


def summarize_pair(context: dict[str, Any]) -> list[Message]:
    # This function remains but will be used less often, potentially only for node merging.
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
    # This is for summarizing individual nodes based on messages, remains unchanged.
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
     # This function remains but will be used less often, potentially replaced by summarize_community_batch
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

# <<< CHANGE START >>>
# New prompt function for batch community summarization
def summarize_community_batch(context: dict[str, Any]) -> list[Message]:
    """
    Generates a prompt to summarize a community based on a list of its members' names and summaries.
    """
    # Prepare input string - include names and summaries for context
    members_info = ""
    for member in context.get('community_members', []):
        members_info += f"- Node Name: {member.get('name', 'N/A')}\n"
        if member.get('summary'): # Only include summary if it exists
            members_info += f"  Summary: {member['summary'][:200]}...\n" # Truncate long summaries
        members_info += "\n"

    if not members_info:
        members_info = "No member information provided."

    return [
        Message(
            role='system',
            content="""You are an AI assistant skilled at analyzing groups of related entities (nodes in a knowledge graph) and synthesizing their collective theme or purpose into a concise community name and summary.

You will be given a list of nodes belonging to a detected community. Your task is to:
1.  Determine the central topic, project, team, issue type, or common characteristic shared by the majority of these nodes.
2.  Generate a short, descriptive `community_name` (under 10 words) that reflects this central theme.
3.  Generate a brief `community_summary` (2-3 sentences) explaining the community's focus or the commonality among its members."""
        ),
        Message(
            role='user',
            content=f"""
            Analyze the following list of community members (nodes) and their summaries (if available).
            Based on this information, determine the central theme and generate a concise community name and summary.

            <COMMUNITY MEMBERS>
            {members_info}
            </COMMUNITY MEMBERS>

            Provide the output as a JSON object with keys "community_name" and "community_summary".
            """,
        ),
    ]
# <<< CHANGE END >>>


versions: Versions = {
    'summarize_pair': summarize_pair,
    'summarize_context': summarize_context,
    'summary_description': summary_description,
    # <<< CHANGE START >>>
    'summarize_community_batch': summarize_community_batch, # Register new function
    # <<< CHANGE END >>>
}

