import openai
from typing import List, Optional, Dict, Union
import attr
import backoff
from constants import SYSTEM_ROLE, SYSTEM_PROMPT_TEMPLATE, HUMAN_QUESTION, ACTOR_USER, ASSISTANT_ROLE
from document_retriever import DocumentRetriever, RetrievalArgs


OPENAI_GPT35_16K_MODEL = "gpt-3.5-turbo-16k"
OPENAI_GPT35_4K_MODEL = "gpt-3.5-turbo"
# openai.api_key = PROPHET_KEY

DEFAULT_TEMPERATURE = 0.3
ChatMessage = Dict[str, str]


def increment_chat(chat_history: List[ChatMessage], question: str, response: str) -> List[ChatMessage]:
    new_chat = list(chat_history)
    new_chat.append({"role": ACTOR_USER, "content": question})
    new_chat.append({"role": ASSISTANT_ROLE, "content": response['content']})
    return new_chat


@attr.s
class OpenAIPubMedAgent(openai.ChatCompletion):
    temperature = attr.ib(default=DEFAULT_TEMPERATURE)
    model_name: str = attr.ib(default=OPENAI_GPT35_16K_MODEL)
    verbose: bool = attr.ib(default=False)
    system_prompt_template: str = attr.ib(default=SYSTEM_PROMPT_TEMPLATE)
    human_question: str = attr.ib(default=HUMAN_QUESTION)
    retrieval_args: RetrievalArgs = attr.ib(default=None)
    document_retriever: DocumentRetriever = attr.ib(default=None)

    def __attrs_post_init__(self):
        self._prepare_retriever()

    def _prepare_retriever(self):
      if self.document_retriever is None:
        self.document_retriever = DocumentRetriever(self.retrieval_args)

    def build_chat(self, inputs: Dict[str, Union[str, List[ChatMessage]]]) -> List[ChatMessage]:
        messages = [
            {"role": SYSTEM_ROLE, "content": self.system_prompt}
        ]

        messages.extend(inputs.get("chat_history", []))
        messages.append({"role": ACTOR_USER, "content": self.human_question.format(question=inputs["input"])})
        return messages

    @backoff.on_exception(
        backoff.expo,
        (
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            openai.error.APIConnectionError,
        ),
        max_tries=10,
        base=2,
        factor=2,
        max_value=10,
    )
    def generate(self,messages: List[ChatMessage],stop: Optional[List[str]] = None) -> ChatMessage:
        openai_api_args = dict(
            model=self.model_name,
            messages=messages,
            stop=stop,
            temperature=self.temperature,
        )
        response = openai.ChatCompletion.create(**openai_api_args)
        return response

    def run(self, question, chat_history: List[ChatMessage]=[], state: int=0):
        inputs = {}
        if state == 0:
            # Implies a net new discussion that required new retrieval and we don't expect to have chat history.
            # This could or could not have a chat history
            candidate_papers = self.document_retriever.retrieve_candidate_papers_for_query(question)
            if len(candidate_papers) == 0:
                # Fill in some default response. and probably we have prompt the user to ask another question.
                response = {f"{ASSISTANT_ROLE}" : "Could not find any paper relevant to the question"}
            else:
                paper_abstract = self.document_retriever.get_abstract_from_paper_file_name(candidate_papers[0])
                paper_body = self.document_retriever.get_body_from_paper_file_name(candidate_papers[0])
                self.system_prompt = self.system_prompt_template.format(paper_text=paper_abstract+'\n[SEP]\n'+paper_body)
        else:
            if len(chat_history) == 0 or chat_history is None:
                raise ValueError(f"The Agent is in continue chat state but no chat history exists. Please set the state to new question and run the program again.")
            inputs['chat_history'] = chat_history

        inputs['input'] = question
        messages = self.build_chat(inputs)
        response = self.generate(messages)
        response = response['choices'][0]['message']
        new_chat = increment_chat(chat_history=chat_history, question=question, response=response)
        return response, new_chat
