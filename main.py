from dataclasses import dataclass, field
import logging
import os
import random
import re
from string import Template
import sys
from typing import Final, Any, NamedTuple, NewType

from huggingface_hub import InferenceClient
import yaml

StoryLine = list[str]
Participant = str
HFChatMessage = dict[str, Any]
HFChatMessages = list[HFChatMessage]  # structure required by the HFace API
Prompt = NewType('Prompt', str)
Response = NewType('Response', str)


@dataclass
class Config:
    participants: list[Participant] = field(
        default_factory=list)  # pyright: ignore
    initial_participant_id: int = 0
    plot: str = ''
    max_depth: int = 0
    max_options: int = 2
    lang: str = ''

    @property
    def initial_participant(self) -> Participant:
        return self.participant_name(self.initial_participant_id)

    def participant_name(self, participant_id: int) -> Participant:
        return self.participants[participant_id]


class LlmStoryManager:
    MODEL_ID = "HuggingFaceTB/SmolLM3-3B"  # TODO: get this from the config

    def __init__(self) -> None:
        api_key = os.environ.get("HF_TOKEN")
        if not api_key:
            print("HF_TOKEN env var not set.")
            sys.exit(1)

        self._client: Final = InferenceClient(provider="hf-inference",
                                              api_key=api_key)

        self._chat_messages: HFChatMessages = []

    def reset_exchanges(self) -> None:
        self._chat_messages = [{"role": "system", "content": "/no_think"}]

    def add_exchange(self, prompt: Prompt, response: Response) -> None:
        self._chat_messages.append({"role": "user", "content": str(prompt)})
        self._chat_messages.append({
            "role": "assistant",
            "content": str(response)
        })

    def prompt(self, question: Prompt) -> Response | None:
        logging.debug(f"QUESTION: {str(question)}.")
        self._chat_messages.append({"role": "user", "content": str(question)})

        completion = self._client.chat.completions.create(  # pyright: ignore
            model=self.MODEL_ID,
            messages=self._chat_messages)

        answer = completion.choices[0].message.content
        assert isinstance(answer, str)
        logging.debug("RESPONSE -> " + answer)
        self.add_exchange(question, Response(answer))
        return Response(answer)


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
        raise ValueError(
            f"Expected exactly one fenced JSON block, found {len(matches)}")

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


class Templates:

    def __init__(self, lang: str) -> None:
        filepath = os.path.join("lang", f"{lang}.yml")
        with open(filepath, 'r', encoding='utf-8') as file:
            template_strings = yaml.safe_load(file)
        self._templates: Final = {
            k: Template(v)
            for k, v in template_strings.items()
        }
        self._values: dict[str, str] = {}

    def set_key(self, key: str, value: str) -> None:
        self._values[key] = value

    def get_str(self, key: str) -> str:
        return self._templates.get(key, Template('')).substitute(self._values)


class StoryState(NamedTuple):
    participant: Participant
    delta_story: str
    options: list[str]


class StoryCreator:

    @dataclass
    class StoryNode:
        participant_id: int
        delta_prompt: Prompt
        delta_story: Response
        options_prompt: Prompt
        options: list[str]
        chosen_option: int | None  # None means that there was no option chosen yet

        @property
        def options_response(self) -> Response:
            return Response("\n\n".join(self.options))

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
            config.initial_participant_id = random.randrange(
                0, len(config.participants))
        else:
            config.initial_participant_id = config.participants.index(
                init_participant_name)

        config.plot = data['plot']
        config.max_depth = data.get('max-depth', 10)
        config.max_options = data.get('max-options', 2)
        config.lang = data.get('language', 'es_AR')
        return config

    def __init__(self) -> None:
        self._config: Final = self._load_config()
        self._llm_manager: Final = LlmStoryManager()
        self._templates = Templates(self._config.lang)
        self._fill_templates()

        self._story: list[StoryCreator.StoryNode] = []
        self._append_new_delta_and_options(self._config.initial_participant_id,
                                           'starting-delta')

    def _fill_templates(self) -> None:
        t = self._templates
        config = self._config
        t.set_key('players', ",".join(config.participants))
        t.set_key('max_options', str(config.max_options))
        t.set_key('plot', config.plot)

    def _prepare_story_line(self) -> None:
        self._llm_manager.reset_exchanges()
        for story_node in self._story:
            self._llm_manager.add_exchange(story_node.delta_prompt,
                                           story_node.delta_story)
            self._llm_manager.add_exchange(story_node.options_prompt,
                                           story_node.options_response)

    def _get_last_option_chosen(self) -> Prompt:

        if self._story:
            story_node = self._story[-1]
            last_chosen_str = story_node.chosen_option_str()
            if last_chosen_str:
                self._templates.set_key(
                    'participant',
                    self._config.participant_name(story_node.participant_id))
                self._templates.set_key('chosen_option', last_chosen_str)
                return Prompt(self._templates.get_str('chose'))

        return Prompt('')

    def _append_new_delta_and_options(self,
                                      participant_id: int,
                                      delta_key: str = 'delta') -> None:
        last_chosen_option = self._get_last_option_chosen()
        self._prepare_story_line()

        participant = self._config.participant_name(participant_id)
        self._templates.set_key('participant', participant)

        # add delta-key
        delta_prompt = Prompt(last_chosen_option +
                              self._templates.get_str(delta_key))
        delta_story = self._llm_manager.prompt(delta_prompt)
        assert delta_story is not None

        # add options
        options_prompt = Prompt(self._templates.get_str('options'))
        options_response = self._llm_manager.prompt(options_prompt)
        assert options_response is not None
        options = split_on_blank_lines(str(options_response))

        if options:
            new_node = StoryCreator.StoryNode(participant_id=participant_id,
                                              delta_prompt=delta_prompt,
                                              delta_story=delta_story,
                                              options_prompt=options_prompt,
                                              options=options,
                                              chosen_option=None)
            self._story.append(new_node)

    def _get_node_participant(self, node: 'StoryCreator.StoryNode') -> str:
        return self._config.participant_name(node.participant_id)

    def get_current(self) -> StoryState:
        current_node = self._story[-1]
        return StoryState(participant=self._get_node_participant(current_node),
                          delta_story=current_node.delta_story,
                          options=current_node.options)

    def _next_participant(self, participant_id: int) -> int:
        return (participant_id + 1) % len(self._config.participants)

    def choose(self, option: int) -> None:
        next_participant_id = self._next_participant(
            self._story[-1].participant_id)
        self._story[-1].chosen_option = option

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


def play_console(sm: StoryCreator) -> None:
    participant, delta_context, options = sm.get_current()
    print(f"""
        ------------------------------
        {delta_context}.
        Elige {participant}.
        Opciones:
    """)
    for i, opt in enumerate(options):
        print(f"{i} - {opt}\n---")

    selected: None | int = None
    while selected is None:
        try:
            selected = int(input("Tu elección?: "))
            sm.choose(selected)
        except ValueError:
            print("Elección equiocada.")
        except KeyboardInterrupt:
            sys.exit(0)

    play_console(sm)


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)

    sm = StoryCreator()

    play_console(sm)


main()
