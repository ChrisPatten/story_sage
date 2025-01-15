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

from openai import OpenAI
from typing import Dict, List, Tuple, TypeAlias
from ..models import StorySageConfig, StorySageConversation, StorySageContext
from pydantic import BaseModel
import httpx
import logging
from ..utils.junk_drawer import EmojiFormatter

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
    
    def __init__(self, config: StorySageConfig, log_level: int = logging.INFO):
        self.config = config
        self.prompts = config.prompts
        self.client = OpenAI(api_key=config.openai_api_key, http_client=httpx.Client(verify=False))

        # Setup logger with emoji formatter
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Create console handler if none exists
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            
            # Create and set formatter
            formatter = EmojiFormatter('%(emoji)s %(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)

    def generate_response(self, context: List[StorySageContext], question: str, 
                          conversation: StorySageConversation = None,
                          model: str = None, **kwargs) -> Tuple[str, bool, str, _UsageType]:
        """Generates a response based on the provided context and question.

        Args:
            context (List[StorySageContext]): List of context objects containing relevant document content
            question (str): User's question to be answered
            conversation (StorySageConversation, optional): Previous conversation history. Defaults to None
            model (str, optional): OpenAI model to use. Defaults to config's model
            **kwargs: Additional arguments passed to OpenAI API

        Returns:
            Tuple[str, bool, str, _UsageType]: Contains:
                - response: Generated answer text
                - has_answer: Boolean indicating if an answer was found
                - follow_up: Suggested follow-up question
                - tokens: Number of tokens used in the turn

        Raises:
            Exception: If OpenAI API call fails
        """
        
        class CompletionResult(BaseModel):
            response: str
            has_answer: bool
            follow_up: str
        
        context = '"""\n' + '"""\n'.join([con.format_for_llm() for con in context]) + '"""'
        prompt_formatter = {'context': context, 'question': question}
        messages = self._set_up_prompt('generate_prompt', prompt_formatter, conversation)
        
        try:
            response = self.client.beta.chat.completions.parse(
                model=model or self.config.completion_model,
                messages=messages,
                temperature=self.config.completion_temperature,
                max_tokens=self.config.completion_max_tokens,
                response_format=CompletionResult
            )
            result: CompletionResult = response.choices[0].message.parsed
            tokens: _UsageType = (response.usage.completion_tokens, response.usage.prompt_tokens)
            self.logger.debug(result)
            return (result.response, result.has_answer, result.follow_up, tokens)
        except Exception as e:
            raise Exception(f"Error getting completion from OpenAI: {str(e)}")

    def identify_relevant_chunks(self, context: Dict[str, str], question: str, 
                                 conversation: StorySageConversation = None, 
                                 model: str = None, **kwargs) -> Tuple[List[str], str, _UsageType]:
        """Identifies the most relevant document chunks for a given question.

        Args:
            context (Dict[str, str]): Dictionary mapping chunk IDs to their summaries
            question (str): User's question to find relevant chunks for
            conversation (StorySageConversation, optional): Previous conversation history. Defaults to None
            model (str, optional): OpenAI model to use. Defaults to config's model
            **kwargs: Additional arguments passed to OpenAI API

        Returns:
            Tuple[List[str], str, _UsageType]: Contains:
                - chunk_ids: List of relevant chunk IDs
                - secondary_query: Refined search query based on the question
                - tokens: Number of tokens used in the turn

        Raises:
            Exception: If OpenAI API call fails
        """
        
        class RelevantChunks(BaseModel):
            chunk_ids: List[str]
            secondary_query: str

        summaries = '\n'.join([f"- {id}: {doc}" for id, doc in context.items()])
        prompt_formatter = {'summaries': summaries, 'question': question}
        messages = self._set_up_prompt('relevant_chunks_prompt', prompt_formatter, conversation)
        
        try:
            response = self.client.beta.chat.completions.parse(
                model=model or self.config.completion_model,
                messages=messages,
                response_format=RelevantChunks
            )
            result: RelevantChunks = response.choices[0].message.parsed
            tokens: _UsageType = (response.usage.completion_tokens, response.usage.prompt_tokens)
            return (result.chunk_ids, result.secondary_query, tokens)
        except Exception as e:
            raise Exception(f"Error getting completion from OpenAI: {str(e)}")
        
    def get_keywords_from_question(self, question: str, conversation: StorySageConversation = None,
                                   model: str = None, **kwargs) -> Tuple[List[str], _UsageType]:
        """Extracts relevant keywords from a question for search purposes.

        Args:
            question (str): Question to extract keywords from
            conversation (StorySageConversation, optional): Previous conversation history. Defaults to None
            model (str, optional): OpenAI model to use. Defaults to config's model
            **kwargs: Additional arguments passed to OpenAI API

        Returns:
            Tuple[List[str], _UsageType]: Contains:
                - keywords: List of extracted keywords
                - tokens: Number of tokens used in the turn
            

        Raises:
            Exception: If OpenAI API call fails
        """
        
        class KeywordsResult(BaseModel):
            keywords: List[str]

        messages = self._set_up_prompt('generate_keywords_prompt', {'question': question}, conversation)

        try:
            response = self.client.beta.chat.completions.parse(
                model=model or self.config.completion_model,
                messages=messages,
                response_format=KeywordsResult
            )
            result: KeywordsResult = response.choices[0].message.parsed
            tokens: _UsageType = (response.usage.completion_tokens, response.usage.prompt_tokens)
            return result.keywords, tokens
        except Exception as e:
            raise Exception(f"Error getting completion from OpenAI: {str(e)}")

    def generate_query(self, question: str, conversation: StorySageConversation = None,
                      model: str = None, **kwargs) -> Tuple[str, _UsageType]:
        """Generates a query for vector store search based on the question.

        Args:
            question (str): Question to generate a search query from
            conversation (StorySageConversation, optional): Previous conversation history. Defaults to None
            model (str, optional): OpenAI model to use. Defaults to config's model
            **kwargs: Additional arguments passed to OpenAI API

        Returns:
            Tuple[str, _UsageType]: Contains:
                - query: Generated search query
                - tokens: Number of tokens used in the turn

        Raises:
            Exception: If OpenAI API call fails
        """
        
        class QueryResult(BaseModel):
            query: str

        messages = self._set_up_prompt('generate_initial_query_prompt', {'question': question}, conversation)

        try:
            response = self.client.beta.chat.completions.parse(
                model=model or self.config.completion_model,
                messages=messages,
                response_format=QueryResult
            )
            result: QueryResult = response.choices[0].message.parsed
            tokens: _UsageType = (response.usage.completion_tokens, response.usage.prompt_tokens)
            return result.query, tokens
        except Exception as e:
            raise Exception(f"Error getting completion from OpenAI: {str(e)}")
        
    def generate_followup_query(self, question: str, conversation: StorySageConversation,
                              model: str = None, **kwargs) -> Tuple[str, _UsageType]:
        """Generates a followup query for vector store search based on conversation history and current question.
    
        Args:
            question (str): Current question to generate search query from
            conversation (StorySageConversation): Previous conversation history
            model (str, optional): OpenAI model to use. Defaults to config's model
            **kwargs: Additional arguments passed to OpenAI API
    
        Returns:
            Tuple[str, _UsageType]: Contains:
                - query: Generated search query
                - tokens: Number of tokens used in the turn
    
        Raises:
            Exception: If OpenAI API call fails
        """
        
        class QueryResult(BaseModel):
            query: str
    
        messages = self._set_up_prompt('generate_followup_query_prompt', {
            'conversation': conversation,
            'question': question
        })
    
        try:
            response = self.client.beta.chat.completions.parse(
                model=model or self.config.completion_model,
                messages=messages,
                response_format=QueryResult
            )
            result: QueryResult = response.choices[0].message.parsed
            tokens: _UsageType = (response.usage.completion_tokens, response.usage.prompt_tokens)
            return result.query, tokens
        except Exception as e:
            raise Exception(f"Error getting completion from OpenAI: {str(e)}")

    def refine_followup_question(self, question: str, conversation: StorySageConversation,
                                 model: str = None, **kwargs) -> Tuple[str, _UsageType]:
        """Refines the user's follow-up question based on the context of the conversation.

        Args:
            question (str): Current follow-up question to refine
            conversation (StorySageConversation): Previous conversation history
            model (str, optional): OpenAI model to use. Defaults to config's model
            **kwargs: Additional arguments passed to OpenAI API

        Returns:
            Tuple[str, _UsageType]: Contains:
                - refined_question: Refined follow-up question
                - tokens: Number of tokens used in the turn

        Raises:
            Exception: If OpenAI API call fails
        """
        
        class RefinedQuestionResult(BaseModel):
            refined_question: str

        messages = self._set_up_prompt('refine_followup_question', {
            'history': conversation,
            'question': question
        })

        try:
            response = self.client.beta.chat.completions.parse(
                model=model or self.config.completion_model,
                messages=messages,
                response_format=RefinedQuestionResult
            )
            result: RefinedQuestionResult = response.choices[0].message.parsed
            tokens: _UsageType = (response.usage.completion_tokens, response.usage.prompt_tokens)
            return result.refined_question, tokens
        except Exception as e:
            raise Exception(f"Error getting completion from OpenAI: {str(e)}")

    
    def _get_conversation_turns(self, conversation: StorySageConversation) -> List[Dict[str, str]]:
        """Formats conversation history into OpenAI message format.

        Args:
            conversation (StorySageConversation): Conversation history to format

        Returns:
            List[Dict[str, str]]: List of message dictionaries with 'role' and 'content'
        """
        turns = []
        for turn in conversation.turns:
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
        
        if prompt_key not in self.prompts:
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

        return messages

