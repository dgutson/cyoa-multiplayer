#from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import os
import random
import re
import sys
from typing import Final, Any, NamedTuple

from huggingface_hub import InferenceClient
import yaml

StoryLine = list[str]
Participant = str
ChatMessages = list[dict[str, Any]]

@dataclass
class Config:
    participants: list[Participant] = field(default_factory=list) # pyright: ignore
    initial_participant_id: int = 0
    plot: str = ''
    max_depth: int = 0
    max_options: int = 2

    @property
    def initial_participant(self) -> Participant:
        return self.participant_name(self.initial_participant_id)

    def participant_name(self, participant_id: int) -> Participant:
        return self.participants[participant_id]

class LlmStoryManager:
    MODEL_ID = "HuggingFaceTB/SmolLM3-3B"

    def __init__(self) -> None:
        api_key = os.environ.get("HF_TOKEN")
        if not api_key:
            print("HF_TOKEN env var not set.")
            sys.exit(1)

        self._client = InferenceClient(
            provider="hf-inference",
            api_key=api_key)

    @staticmethod
    def _add_chat_message(chat_messages: ChatMessages, message: str) -> ChatMessages:
        chat_messages.append({"role": "user", "content": message})
        return chat_messages

    @staticmethod
    def _story_line_to_chat(story_line: StoryLine) -> ChatMessages:
        chat_messages = [{"role": "system", "content": "/no_think"}]
        for story_item in story_line:
            logging.debug(f"STORY ITEM: {story_item}")
            LlmStoryManager._add_chat_message(chat_messages, story_item)

        return chat_messages

    def prompt(self, story_so_far: StoryLine, question: str) -> str | None:
        chat_messages = self._story_line_to_chat(story_so_far)
        logging.debug(f"QUESTION: {question}.")
        chat_messages = self._add_chat_message(chat_messages, question)

        completion = self._client.chat.completions.create( # pyright: ignore
            model = self.MODEL_ID,
            messages = chat_messages
        )

        answer = completion.choices[0].message.content
        assert isinstance(answer, str)
        logging.debug("RESPONSE -> " + answer)
        return answer

_JSON_FENCE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)
def json_cleanup(input_str: str) -> str:
    matches = _JSON_FENCE.findall(input_str)
    markup_count = len(matches)

    if markup_count == 0:
        json_str = input_str
    elif markup_count == 1:
        json_str = matches[0]
        logging.debug(f"json-sanitized: {json_str}.")
    else:
        raise ValueError(f"Expected exactly one fenced JSON block, found {len(matches)}")

    json_str = json_str.strip()

    if not json_str.endswith('}'):
        json_str += '}'

    json_str = json_str.replace("]]", "]")

    return json_str

def split_on_blank_lines(text: str) -> list[str]:
    # Split on two or more line breaks with optional spaces/tabs, ignore multiples
    sections = re.split(r'\n\s*\n+', text.strip())
    # Strip leading/trailing whitespace from each section
    return [section.strip() for section in sections if section.strip()]

class StoryState(NamedTuple):
    participant: Participant
    delta_story: str
    options: list[str]

class StoryCreator:
    @dataclass
    class StoryNode:
        participant_id: int
        delta_story: str
        options: list[str]
        chosen_option: int | None # None means that there was no option chosen yet

        def chosen_option_str(self) -> str | None:
            if self.chosen_option is not None:
                return self.options[self.chosen_option]
            return None

    @staticmethod
    def _load_config() -> Config:
        config = Config()

        with open('story.yml', 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        config.participants = data['participants']
        init_participant_name = data.get('initial-participant', 'random')
        if init_participant_name == 'random':
            config.initial_participant_id = random.randrange(0, len(config.participants))
        else:
            config.initial_participant_id = config.participants.index(init_participant_name)

        config.plot = data['plot']
        config.max_depth = data.get('max-depth', 10)
        config.max_options = data.get('max-options', 2)
        return config

    def __init__(self) -> None:
        self._config: Final = self._load_config()
        self._llm_manager: Final = LlmStoryManager()
        init_story_node = StoryCreator.StoryNode(
            participant_id=self._config.initial_participant_id,
            delta_story=self._create_starting_delta(),
            options=[],
            chosen_option=None
        )
        self._story: list[StoryCreator.StoryNode] = [init_story_node]
        self._append_new_delta_and_options(self._config.initial_participant_id)

    def _create_starting_delta(self) -> str:
        preamble = f"""
            Vamos a crear un relato estilo 'Elige Tu Propia Aventura',
            pero multijugador por turnos, es decir, cada elección la hace
            cada jugador por turnos.
            Los jugadores son: {",".join(self._config.participants)}.
            La cantidad de opciones por turno es {self._config.max_options}.
            El argumento es este:
            {self._config.plot}
        """
        return preamble

    def _append_new_delta_and_options(self, participant_id: int) -> None:
        participant = self._config.participant_name(participant_id)
        prompt = f"""
            Ahora elige {participant}.
            Describe la proxima situación (como un incremento a la historia hasta el momento),
            antes de enumerar las opciones.
            """
        story_line = self._get_story_line()
        delta_story = self._llm_manager.prompt(story_line, prompt)
        assert delta_story is not None
        story_line.append(delta_story)
        prompt = f"""
            Ahora enumera las opciones que tiene {participant}.
            Escribe cada opción en un párrafo, y sepáralos por una línea en blanco. 
        """
        options_str = self._llm_manager.prompt(story_line, prompt)
        assert options_str is not None
        options = split_on_blank_lines(options_str)

        if options:
            new_node = StoryCreator.StoryNode(
                delta_story=delta_story,
                options=options,
                participant_id=participant_id,
                chosen_option=None
            )
            self._story.append(new_node)

    def _get_node_participant(self, node: 'StoryCreator.StoryNode') -> str:
        return self._config.participant_name(node.participant_id)

    def _get_story_line(self) -> StoryLine:
        story_line: StoryLine = []
        for story_node in self._story:
            text = story_node.delta_story
            chosen_option = story_node.chosen_option_str()
            if chosen_option:
                text += f"""
                    \n
                    {self._get_node_participant(story_node)} eligió:\n
                    <{chosen_option}>
                """
            story_line.append(text)
        return story_line

    def get_current(self) -> StoryState:
        current_node = self._story[-1]
        return StoryState(
            participant=self._get_node_participant(current_node),
            delta_story=current_node.delta_story,
            options=current_node.options)


    def _next_participant(self, participant_id: int) -> int:
        return (participant_id + 1) % len(self._config.participants)

    def choose(self, option: int) -> None:
        next_participant_id = self._next_participant(self._story[-1].participant_id)
        self._story[-1].chosen_option = option
        logging.debug(f"chose option: {self._story[-1].chosen_option}, string: {self._story[-1].chosen_option_str()}")
        self._append_new_delta_and_options(next_participant_id)

    def pop_option(self) -> None:
        self._story.pop()

def play(sm: StoryCreator, level: int) -> None:
    participant, delta_context, options = sm.get_current()
    indent: Final = '--' * level
    print(f"{indent}Delta-context: {delta_context}.")
    print(f"{indent}{participant} tiene que elegir:")
    print('\n'.join(options))
    for opt_index in range(len(options)):
        print(f"{indent} - eligiendo opcion {opt_index}")
        sm.choose(opt_index)
        play(sm, level + 1)
        sm.pop_option()



def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    sm = StoryCreator()
    play(sm, 0)

main()
