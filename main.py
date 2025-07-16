from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import random
from typing import Final

from huggingface_hub import InferenceClient
import yaml

StoryLine = list[str]
Participant = str
ChatMessages = list[dict]

@dataclass
class Config:
    participants: list[Participant] = []
    initial_participant_id: int = 0
    plot: str = ''
    max_depth: int = 0
    max_options: int = 2

    @property
    def initial_participant(self) -> Participant:
        return self.participants[self.initial_participant_id]


class LlmStoryManager:
    MODEL_ID = "HuggingFaceTB/SmolLM3-3B"

    def __init__(self) -> None:
        self._client = InferenceClient(
            provider="hf-inference",
            api_key=os.environ["HF_TOKEN"])

    @staticmethod
    def _add_chat_message(chat_messages: ChatMessages, message: str) -> ChatMessages:
        chat_messages.append({"role": "user", "content": message})
        return chat_messages

    @staticmethod
    def _story_line_to_chat(story_line: StoryLine) -> ChatMessages:
        chat_messages = [{"role": "system", "content": "/no_think"}]
        for story_item in story_line:
            LlmStoryManager._add_chat_message(chat_messages, story_item)

        return chat_messages

    def get_options(self, story_so_far: StoryLine, question: str) -> list[str]:
        chat_messages = self._story_line_to_chat(story_so_far)
        chat_messages = self._add_chat_message(chat_messages, question)

        completion = self._client.chat.completions.create(
            model = self.MODEL_ID,
            messages = chat_messages
        )

        answer = completion.choices[0].message.content
        if answer:
            story_so_far.append(answer)
            return story_so_far
        return []


class StoryTree:
    class StoryNode:
        def __init__(self, participant_id: int, story_so_far: StoryLine) -> None:
            self._participant_id: Final = participant_id
            self._story_so_far = story_so_far
            self.children: list['StoryTree.StoryNode'] = []

    def __init__(self, starting_participant: int, starting_story: StoryLine) -> None:
        self._root: Final = StoryTree.StoryNode(starting_participant, starting_story)

class PlayerInterface(ABC):
    @abstractmethod
    def decide(self, participant: int, context: str, options: list[str]) -> int:
        pass

class StoryCreator:
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

        config.max_depth = data.get('max-depth', 10)
        config.max_options = data.get('max-options', 2)
        return config

    def __init__(self, player_interface: PlayerInterface) -> None:
        self._player_interface: Final = player_interface
        self._config = self._load_config()
        starting_story = self._create_starting_story()
        self._story_tree = StoryTree(self._config.initial_participant_id, starting_story)

    def _create_starting_story(self) -> StoryLine:
        preamble = f"""
            Vamos a crear un relato estilo 'Elige Tu Propia Aventura',
            pero multijugador por turnos, es decir, cada elección la hace
            cada jugador por turnos.
            Los jugadores son: {",".join(self._config.participants)}.
            La cantidad de opciones por turno es {self._config.max_options}.
            El argumento es este:
            {self._config.plot}

            El primero en elegir es {self._config.initial_participant}.
        """
        return StoryLine(preamble)




"""
completion = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM3-3B",
    messages=[
        {"role": "system", "content": "/no_think"},
        {
            "role": "user",
            "content": "Hagamos una historia de Elige Tu Propia Aventura. Qué nombre elegirías?"
        },
        {
            "role": "user",
            "content": "De qué se trataría?"
        }
    ],
)

print(completion.choices[0].message.content)
"""
