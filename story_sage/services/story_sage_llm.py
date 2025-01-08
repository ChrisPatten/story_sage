from openai import OpenAI
from typing import Dict, List, Any, Tuple
from ..models import StorySageConfig, StorySageConversation, StorySageContext
from pydantic import BaseModel
import httpx
import logging



class StorySageLLM:
    def __init__(self, config: StorySageConfig, log_level: int = logging.INFO):
        self.config = config
        self.prompts = config.prompts
        self.client = OpenAI(api_key=config.openai_api_key, http_client=httpx.Client(verify=False))
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

    def generate_response(self, context: List[StorySageContext], question: str, 
                          conversation: StorySageConversation = None,
                          model: str = None, **kwargs) -> Tuple[str, bool, str]:
        
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
            self.logger.debug(result)
            return (result.response, result.has_answer, result.follow_up)
        except Exception as e:
            raise Exception(f"Error getting completion from OpenAI: {str(e)}")

    def identify_relevant_chunks(self, context: Dict[str, str], question: str, 
                                 conversation: StorySageConversation = None, 
                                 model: str = None, **kwargs) -> Tuple[List[str], str]:

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
            return (result.chunk_ids, result.secondary_query)
        except Exception as e:
            raise Exception(f"Error getting completion from OpenAI: {str(e)}")
        
    def get_keywords_from_question(self, question: str, conversation: StorySageConversation = None,
                                   model: str = None, **kwargs) -> List[str]:
        
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
            return result.keywords
        except Exception as e:
            raise Exception(f"Error getting completion from OpenAI: {str(e)}")

    
    def _get_conversation_turns(self, conversation: StorySageConversation) -> List[Dict[str, str]]:
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

