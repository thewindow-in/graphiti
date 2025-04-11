# graphiti_core/llm_client/openai_client.py
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
- Use standard `create` with `json_object` response format instead of beta `parse`.
- Manually validate the JSON response against the Pydantic model.
- Refine how the JSON schema instructions are appended to the prompt.
"""

import json
import logging
import typing
from typing import ClassVar, Optional # Added Optional
import asyncio # Added asyncio for sleep

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

# Assuming Message model is defined correctly
from ..prompts.models import Message
from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig
from .errors import RateLimitError, RefusalError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gpt-4o-mini'


class OpenAIClient(LLMClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models
    (or compatible APIs like Google's via OpenAI library).

    Uses standard JSON mode and manual Pydantic validation.
    Refined prompt appendage for schema instructions.
    """

    MAX_RETRIES: ClassVar[int] = 2 # Keep retry count for recoverable errors

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
    ):
        """
        Initialize the OpenAIClient.
        """
        if cache:
            logger.warning('Caching is not implemented/enabled for OpenAIClient')
            cache = False

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        if client is None:
            # Ensure base_url is handled correctly, default might be OpenAI's
            api_base_url = config.base_url # Use configured base_url if provided
            logger.info(f"Initializing OpenAIClient with base_url: {api_base_url or 'Default OpenAI'}")
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=api_base_url)
        else:
            self.client = client

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, typing.Any]:
        """
        Internal method to generate response using standard JSON mode.
        """
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            # Ensure content is cleaned (happens in generate_response before call)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})
            elif m.role == 'assistant': # Include assistant messages if present
                 openai_messages.append({'role': 'assistant', 'content': m.content})

        effective_max_tokens = max_tokens or self.max_tokens

        try:
            logger.debug(f"Calling LLM. Model: {self.model or DEFAULT_MODEL}, Temp: {self.temperature}, MaxTokens: {effective_max_tokens}")
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=effective_max_tokens,
                response_format={'type': 'json_object'}, # Use standard JSON mode
            )

            response_content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            logger.debug(f"LLM Raw Response Finish Reason: {finish_reason}")
            logger.debug(f"LLM Raw Response Content Snippet: {response_content[:200] if response_content else 'None'}...")


            if finish_reason == 'content_filter':
                raise RefusalError("LLM refused to generate content due to content filter.")
            # Sometimes length causes partial/invalid JSON
            if finish_reason == 'length':
                 logger.warning("LLM stopped due to length, response might be incomplete/invalid JSON.")
                 # Attempt to parse anyway, might fail
            if not response_content:
                 raise ValueError("LLM returned empty content.")

            try:
                parsed_json = json.loads(response_content)
                if response_model:
                    try:
                        validated_data = response_model.model_validate(parsed_json)
                        return validated_data.model_dump()
                    except ValidationError as e:
                        logger.error(f"LLM response failed Pydantic validation: {e}. Response: {response_content[:500]}...")
                        # Raise specific error for retry logic
                        raise ValueError(f"LLM response validation failed: {e}") from e
                else:
                    if isinstance(parsed_json, dict):
                        return parsed_json
                    else:
                        raise ValueError(f"LLM returned non-dictionary JSON: {type(parsed_json)}")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode LLM JSON response: {e}. Response: {response_content[:500]}...")
                # Raise specific error for retry logic
                raise ValueError(f"Invalid JSON response from LLM: {e}") from e

        except openai.APITimeoutError as e:
             logger.warning(f"OpenAI API timeout: {e}")
             raise # Let retry handler deal with it
        except openai.APIConnectionError as e:
             logger.warning(f"OpenAI API connection error: {e}")
             raise # Let retry handler deal with it
        except openai.RateLimitError as e:
             logger.warning(f"OpenAI API rate limit exceeded: {e}")
             raise RateLimitError from e
        except openai.BadRequestError as e:
            logger.error(f"OpenAI API Bad Request (400): {e}")
            raise # Re-raise to indicate failure
        except RefusalError:
             raise # Propagate refusal errors directly
        except Exception as e:
            # Catch other potential errors (like ValueError from parsing/validation)
            logger.error(f"Unexpected error during LLM response generation: {e}", exc_info=True)
            raise # Re-raise for retry logic

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, typing.Any]:
        """
        Generates a response from the LLM, handling retries and Pydantic model validation.
        Appends refined JSON schema description to the prompt if response_model is provided.
        """
        # Create a copy to avoid modifying the original list passed by the caller
        messages_copy = [m.model_copy() for m in messages]

        # --- MODIFICATION START: Refined schema appending ---
        if response_model:
            try:
                schema_json = json.dumps(response_model.model_json_schema(), indent=2)

                # Refined instructions - place at the end of the *last message*
                # Ensure it's clearly separated and emphasizes the output format constraint.
                schema_prompt = (
                    f"\n\n---\n" # Clear separator
                    f"IMPORTANT: Your entire response must be ONLY the JSON object described below, "
                    f"conforming precisely to the provided schema. Do not include any introductory text, "
                    f"markdown formatting, or explanations outside the JSON structure.\n\n"
                    f"JSON Schema:\n```json\n{schema_json}\n```"
                )

                if messages_copy:
                    # Append to the content of the last message
                    messages_copy[-1].content += schema_prompt
                    logger.debug(f"Appended JSON schema instructions for {response_model.__name__} to the last message.")
                else:
                    # Should not happen if called correctly, but handle defensively
                    logger.error("Cannot append schema: messages list is empty.")
                    # Consider raising an error or proceeding without schema? Proceeding for now.

            except Exception as e:
                logger.error(f"Failed to generate or append JSON schema for {response_model.__name__}: {e}")
                # Proceed without schema instructions if generation fails

        # Clean input content *before* sending
        for message in messages_copy:
             message.content = self._clean_input(message.content)
        # --- MODIFICATION END ---

        # Retry Logic (Simplified - relies on exceptions from _generate_response)
        retry_count = 0
        last_exception: Optional[Exception] = None

        while retry_count <= self.MAX_RETRIES:
            try:
                # Call the internal method with the modified messages copy
                response_dict = await self._generate_response(messages_copy, response_model, max_tokens)
                return response_dict # Success

            except (RateLimitError, RefusalError, openai.BadRequestError) as e:
                # Don't retry these specific errors
                logger.error(f"Non-retryable error encountered: {e.__class__.__name__}")
                last_exception = e
                break # Exit retry loop

            except Exception as e:
                # Catch potentially retryable errors (Timeout, Connection, ValueError from parsing/validation)
                last_exception = e
                retry_count += 1
                logger.warning(
                    f"Attempt {retry_count}/{self.MAX_RETRIES} failed: {e.__class__.__name__}: {str(e)[:200]}. Retrying..."
                )
                if retry_count > self.MAX_RETRIES:
                    logger.error(f"Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}", exc_info=True)
                    break # Exit retry loop

                # Optional: Add context about the error to the prompt for the next attempt
                # Be careful with prompt length and sensitive info
                # error_context = f"\n\n[System Note: Previous attempt failed with error: {e.__class__.__name__}. Please ensure the response is valid JSON conforming strictly to the schema provided.]"
                # messages_copy[-1].content += error_context

                # Implement backoff
                wait_time = 2 ** retry_count # Simple exponential backoff
                logger.info(f"Waiting {wait_time}s before retry attempt {retry_count+1}...")
                await asyncio.sleep(wait_time)

        # If loop finished due to retries exceeded or non-retryable error
        if last_exception:
             raise last_exception
        else:
             raise Exception("LLM request failed after retries without specific exception.")

