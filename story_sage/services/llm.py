"""StorySageLLM: A wrapper for OpenAI's API to handle conversational document Q&A.

This class provides methods to generate responses, identify relevant document chunks,
and extract keywords from questions using OpenAI's chat completion API.

Example:
    config = StorySageConfig(
        openai_api_key="your-api-key",
        prompts={...},
        completion_model="gpt-4",
        completion_temperature=0.7,
        completion_max_tokens=1000
    )
    
    llm = StorySageLLM(config)
    
    # Generate a response
    context = [StorySageContext(
        chunk_id="b1_ch1_001",
        book_number=1,
        chapter_number=1,
        chunk="It was the best of times, it was the worst of times..."
    )]
    response, has_answer, follow_up = llm.generate_response(
        context=context,
        question="What is mentioned in the document?"
    )
    # Response: "The document mentions...", True, "Would you like to know more?"
    
    # Identify relevant chunks
    chunks, query = llm.identify_relevant_chunks(
        context={"chunk1": "Summary 1", "chunk2": "Summary 2"},
        question="Find relevant information"
    )
    # Result: ["chunk1"], "refined search query"
"""

from openai import OpenAI, BadRequestError
from typing import Dict, List, Tuple, TypeAlias, Optional, Union
from ..models import StorySageConfig, StorySageConversation, StorySageContext
from ..models.interfaces import (
    ResponseData, ChunkEvaluationResult, KeywordsResult, QueryResult,
    RefinedQuestionResult, TemporalQueryResult, SearchStrategyResult,
    SearchEvaluationResult
)
from pydantic import ValidationError
import httpx
import logging
from ..utils.junk_drawer import EmojiFormatter
import json
from dataclasses import dataclass
from .search import SearchResult, SearchStrategy

_UsageType: TypeAlias = Tuple[int, int]
"""Type alias for usage information tuple containing:
    completion_tokens: Number of tokens in the generated completion
    prompt_tokens: Number of tokens in the prompt
"""

class StorySageLLM:
    """A class to handle LLM operations for document Q&A using OpenAI's API.
    
    Args:
        config (StorySageConfig): Configuration object containing API keys, prompts, and model settings
        log_level (int, optional): Logging level. Defaults to logging.INFO
    
    Attributes:
        config (StorySageConfig): Stored configuration object
        prompts (dict): Prompt templates from config
        client (OpenAI): OpenAI client instance
        logger (Logger): Logger instance for the class
    """
    
    def __init__(self, config: StorySageConfig):
        self.config = config
        self.prompts = config.prompts
        self.client = OpenAI(api_key=config.openai_api_key, http_client=httpx.Client(verify=False))

        # Setup logger with emoji formatter
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initializing StorySageLLM with model %s", config.completion_model)
        self.logger.debug("API configuration complete")

    def generate_response(self, context: List[StorySageContext], question: str, 
                          conversation: StorySageConversation = None,
                          model: str = None, **kwargs) -> Tuple[ResponseData, _UsageType]:
        """Generates a response based on the provided context and question.

        Args:
            context (List[StorySageContext]): List of context objects containing relevant document content
            question (str): User's question to be answered
            conversation (StorySageConversation, optional): Previous conversation history. Defaults to None
            model (str, optional): OpenAI model to use. Defaults to config's model
            **kwargs: Additional arguments passed to OpenAI API

        Returns:
            Tuple[ResponseData, _UsageType]: Contains:
                - response: Generated answer text
                - has_answer: Boolean indicating if an answer was found
                - follow_up: Suggested follow-up question
                - tokens: Number of tokens used in the turn

        Raises:
            Exception: If OpenAI API call fails
        """
        
        self.logger.info("Generating response for question: '%s'", question[:100])
        self.logger.debug("Using %d context chunks", len(context))
        
        context = '"""\n' + '"""\n'.join([con.format_for_llm() for con in context]) + '"""'
        prompt_formatter = {'context': context, 'question': question}
        messages = self._set_up_prompt('generate_prompt', prompt_formatter, conversation)
        
        try:
            response = self.client.beta.chat.completions.parse(
                model=model or self.config.completion_model,
                messages=messages,
                temperature=self.config.completion_temperature,
                max_tokens=self.config.completion_max_tokens,
                response_format=ResponseData
            )
            result: ResponseData = response.choices[0].message.parsed
            tokens: _UsageType = (response.usage.completion_tokens, response.usage.prompt_tokens)
            self.logger.debug("API response received: tokens used (completion=%d, prompt=%d)", 
                              response.usage.completion_tokens, response.usage.prompt_tokens)
            return result, tokens
        except Exception as e:
            self.logger.error("Failed to generate response: %s", str(e), exc_info=True)
            raise

    def evaluate_chunks(self, context: Dict[str, str], question: str,
                   conversation: StorySageConversation = None,
                   model: str = None, **kwargs) -> Tuple[ChunkEvaluationResult, _UsageType]:
        """Combined chunk evaluation method.
        
        Returns:
            Tuple[ChunkEvaluationResult, _UsageType]: Evaluation results and token usage
        """

        self.logger.info("Evaluating chunks for question: '%s'", question[:100])
        
        summaries = '\n'.join([f"- {id}: {doc}" for id, doc in context.items()])
        prompt_formatter = {'chunks': summaries, 'question': question}
        messages = self._set_up_prompt('evaluate_chunks', prompt_formatter, conversation)
        response_format_object = {
            "type": "json_schema",
            "json_schema": {
                "name": "evaluate_chunks",
                "description": "Evaluates document chunks for relevancy to a user's question.",
                "schema": {
                    "type": "object",
                    "properties": {
                        "chunk_scores": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "chunk_id": {
                                            "type": "string",
                                            "description": "The ID of the document chunk"
                                        },
                                        "score": {
                                            "type": "integer",
                                            "description": "The relevance score of the chunk from 0-9"
                                        }
                                    },
                                    "required": ["chunk_id", "score"],
                                    "additionalProperties": False
                                }
                        },
                        "secondary_query": {
                            "type": ["string", "null"],
                            "description": "A refined search query based on the user's question"
                        }
                        
                    },
                    "required": ["chunk_scores", "secondary_query"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        
        try:
            response = self.client.beta.chat.completions.parse(
                model=model or self.config.completion_model,
                messages=messages,
                response_format=response_format_object
            )
            try:
                result = ChunkEvaluationResult.model_validate_json(response.choices[0].message.content)
            except ValidationError as e:
                self.logger.debug("Response content: %s", response.choices[0].message.content)
                self.logger.error("Failed to validate response: %s", str(e), exc_info=True)
                result = ChunkEvaluationResult.model_validate({'chunk_scores': [], 'secondary_query': None})
            tokens: _UsageType = (response.usage.completion_tokens, response.usage.prompt_tokens)
            return result, tokens
        except BadRequestError as e:
            self.logger.error(response)
            raise
        except Exception as e:
            self.logger.error("Failed to evaluate chunks: %s", str(e), exc_info=True)
            raise

    def get_keywords_from_question(self, question: str, conversation: StorySageConversation = None,
                                   model: str = None, **kwargs) -> Tuple[KeywordsResult, _UsageType]:
        """Extracts relevant keywords from a question for search purposes.

        Args:
            question (str): Question to extract keywords from
            conversation (StorySageConversation, optional): Previous conversation history. Defaults to None
            model (str, optional): OpenAI model to use. Defaults to config's model
            **kwargs: Additional arguments passed to OpenAI API

        Returns:
            Tuple[KeywordsResult, _UsageType]: Contains:
                - keywords: List of extracted keywords
                - tokens: Number of tokens used in the turn
            

        Raises:
            Exception: If OpenAI API call fails
        """
        
        self.logger.info("Extracting keywords from question: '%s'", question[:100])
        
        messages = self._set_up_prompt('generate_keywords_prompt', {'question': question}, conversation)

        try:
            response = self.client.beta.chat.completions.parse(
                model=model or self.config.completion_model,
                messages=messages,
                response_format=KeywordsResult
            )
            result: KeywordsResult = response.choices[0].message.parsed
            tokens: _UsageType = (response.usage.completion_tokens, response.usage.prompt_tokens)
            self.logger.debug("Extracted keywords: %s", result.keywords)
            return result, tokens
        except Exception as e:
            self.logger.error("Failed to extract keywords: %s", str(e), exc_info=True)
            raise

    def generate_query(self, question: str, conversation: StorySageConversation = None,
                      model: str = None, **kwargs) -> Tuple[QueryResult, _UsageType]:
        """Generates a query for vector store search based on the question.

        Args:
            question (str): Question to generate a search query from
            conversation (StorySageConversation, optional): Previous conversation history. Defaults to None
            model (str, optional): OpenAI model to use. Defaults to config's model
            **kwargs: Additional arguments passed to OpenAI API

        Returns:
            Tuple[QueryResult, _UsageType]: Contains:
                - query: Generated search query
                - tokens: Number of tokens used in the turn

        Raises:
            Exception: If OpenAI API call fails
        """
        
        self.logger.info("Generating search query for question: '%s'", question[:100])
        
        messages = self._set_up_prompt('generate_initial_query_prompt', {'question': question}, conversation)

        try:
            response = self.client.beta.chat.completions.parse(
                model=model or self.config.completion_model,
                messages=messages,
                response_format=QueryResult
            )
            result: QueryResult = response.choices[0].message.parsed
            tokens: _UsageType = (response.usage.completion_tokens, response.usage.prompt_tokens)
            self.logger.debug("Generated query: '%s', needs overview: %s", 
                              result.query, result.needs_overview)
            return result, tokens
        except Exception as e:
            self.logger.error("Failed to generate query: %s", str(e), exc_info=True)
            raise
        
    def generate_followup_query(self, question: str, conversation: StorySageConversation,
                              model: str = None, **kwargs) -> Tuple[QueryResult, _UsageType]:
        """Generates a followup query for vector store search based on conversation history and current question.
    
        Args:
            question (str): Current question to generate search query from
            conversation (StorySageConversation): Previous conversation history
            model (str, optional): OpenAI model to use. Defaults to config's model
            **kwargs: Additional arguments passed to OpenAI API
    
        Returns:
            Tuple[QueryResult, _UsageType]: Contains:
                - query: Generated search query
                - tokens: Number of tokens used in the turn
    
        Raises:
            Exception: If OpenAI API call fails
        """
        
        self.logger.info("Generating followup query for question: '%s'", question[:100])
        
        messages = self._set_up_prompt('generate_followup_query_prompt', 
                                       {'question': question},
                                       conversation=conversation)
    
        try:
            response = self.client.beta.chat.completions.parse(
                model=model or self.config.completion_model,
                messages=messages,
                response_format=QueryResult
            )
            result: QueryResult = response.choices[0].message.parsed
            tokens: _UsageType = (response.usage.completion_tokens, response.usage.prompt_tokens)
            return result, tokens
        except Exception as e:
            self.logger.error("Failed to generate followup query: %s", str(e), exc_info=True)
            raise

    def refine_followup_question(self, question: str, conversation: StorySageConversation,
                                 model: str = None, **kwargs) -> Tuple[RefinedQuestionResult, _UsageType]:
        """Refines the user's follow-up question based on the context of the conversation.

        Args:
            question (str): Current follow-up question to refine
            conversation (StorySageConversation): Previous conversation history
            model (str, optional): OpenAI model to use. Defaults to config's model
            **kwargs: Additional arguments passed to OpenAI API

        Returns:
            Tuple[RefinedQuestionResult, _UsageType]: Contains:
                - refined_question: Refined follow-up question
                - tokens: Number of tokens used in the turn

        Raises:
            Exception: If OpenAI API call fails
        """
        
        self.logger.info("Refining followup question: '%s'", question[:100])
        
        messages = self._set_up_prompt('refine_followup_question', 
                                       {'question': question},
                                       conversation=conversation)

        try:
            response = self.client.beta.chat.completions.parse(
                model=model or self.config.completion_model,
                messages=messages,
                response_format=RefinedQuestionResult
            )
            result: RefinedQuestionResult = response.choices[0].message.parsed
            tokens: _UsageType = (response.usage.completion_tokens, response.usage.prompt_tokens)
            return result, tokens
        except Exception as e:
            self.logger.error("Failed to refine followup question: %s", str(e), exc_info=True)
            raise

    def detect_temporal_query(self, question: str, 
                            conversation: StorySageConversation = None,
                            model: str = None, **kwargs) -> Tuple[TemporalQueryResult, _UsageType]:
        """Detects if a question is asking about a specific point in time.

        Args:
            question (str): The user's question
            conversation (StorySageConversation, optional): Conversation history
            model (str, optional): OpenAI model to use

        Returns:
            Tuple[TemporalQueryResult, _UsageType]: Contains temporal query information and token usage
        """
        self.logger.info("Detecting temporal aspects in question: '%s'", question[:100])
        
        messages = self._set_up_prompt('detect_temporal_query', {'question': question}, conversation)

        try:
            response = self.client.beta.chat.completions.parse(
                model=model or self.config.completion_model,
                messages=messages,
                response_format=TemporalQueryResult
            )
            result: TemporalQueryResult = response.choices[0].message.parsed
            tokens: _UsageType = (response.usage.completion_tokens, response.usage.prompt_tokens)
            return result, tokens
        except Exception as e:
            self.logger.error("Failed to detect temporal query: %s", str(e), exc_info=True)
            raise
    
    def determine_search_strategy(self, question: str, 
                                  model: str = None, 
                                  conversation: Optional[StorySageConversation] = None) -> Tuple[SearchStrategyResult, Tuple[int, int]]:
        """Determine the best search strategy for a given question."""
        messages = self._set_up_prompt("determine_search_strategy", {"question": question}, conversation)
        
        try:
            response = self.client.beta.chat.completions.parse(
                model=model or self.config.completion_model,
                messages=messages,
                response_format=SearchStrategyResult
            )
            result: SearchStrategyResult = response.choices[0].message.parsed
            tokens: _UsageType = (response.usage.completion_tokens, response.usage.prompt_tokens)
            return result, tokens
        except Exception as e:
            self.logger.error("Failed to determine search strategy: %s", str(e), exc_info=True)
            raise

    def evaluate_search_results(self, question: str, 
                              similarity_chunks: Dict[str, str],
                              keyword_chunks: List[SearchResult],
                              model: str = None,
                              conversation: Optional[StorySageConversation] = None) -> Tuple[SearchEvaluationResult, Tuple[int, int]]:
        """Evaluate which search results are more relevant for the question."""
        messages = self._set_up_prompt("evaluate_search_results", {
                                        "question": question,
                                        "similarity_chunks": similarity_chunks,
                                        "keyword_chunks": keyword_chunks
                                        }, conversation)
        
        try:
            response = self.client.beta.chat.completions.parse(
                model=model or self.config.completion_model,
                messages=messages,
                response_format=SearchEvaluationResult
            )
            result: SearchEvaluationResult = response.choices[0].message.parsed
            tokens: _UsageType = (response.usage.completion_tokens, response.usage.prompt_tokens)
            return result, tokens
        except Exception as e:
            self.logger.error("Failed to evaluate search results: %s", str(e), exc_info=True)
            raise
    
    def _get_conversation_turns(self, conversation: StorySageConversation) -> List[Dict[str, str]]:
        """Formats conversation history into OpenAI message format.

        Args:
            conversation (StorySageConversation): Conversation history to format

        Returns:
            List[Dict[str, str]]: List of message dictionaries with 'role' and 'content'
        """
        turns = []
        for turn in conversation.turns[:3]:
            turns.append({
                'role': 'user',
                'content': turn.question
            })
            turns.append({
                'role': 'assistant',
                'content': turn.response
            })
        return turns
    
    def _set_up_prompt(self, prompt_key: str, prompt_formatter: Dict[str, str], 
                       conversation: StorySageConversation = None) -> List[Dict[str, str]]:
        """Sets up the complete prompt with system message, conversation history, and user query.

        Args:
            prompt_key (str): Key to lookup prompt template in config
            prompt_formatter (Dict[str, str]): Values to format into prompt template
            conversation (StorySageConversation, optional): Conversation history. Defaults to None

        Returns:
            List[Dict[str, str]]: Formatted messages ready for OpenAI API

        Raises:
            KeyError: If prompt_key is not found in configured prompts
        """
        
        self.logger.debug("Setting up prompt with key: %s", prompt_key)
        
        if prompt_key not in self.prompts:
            self.logger.error("Invalid prompt key: %s", prompt_key)
            raise KeyError(f"Prompt key '{prompt_key}' not found in prompts")
        prompt_template = self.prompts[prompt_key]
        messages = []
        # Add initial system prompt
        messages.append({
            'role': prompt_template[0]['role'], 
            'content': prompt_template[0]['prompt']
        })
        # Add prior conversation turns
        if conversation:
            messages.extend(self._get_conversation_turns(conversation))
        # Add context and question
        for prompt in prompt_template[1:]:
            messages.append({'role': prompt['role'], 'content': prompt['prompt'].format_map(prompt_formatter)})

        self.logger.debug("Prompt setup complete with %d messages", len(messages))
        return messages

