import argparse
from dataclasses import dataclass, field
import json
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
    participants: list[Participant] = field(  # pyright: ignore
        default_factory=list)
    participants_data: str = ''
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

    def __init__(self, model: str, debug_mode: bool = False) -> None:
        api_key = os.environ.get("HF_TOKEN")
        if not api_key:
            print("HF_TOKEN env var not set.")
            sys.exit(1)

        self._model: Final = model
        self._client: Final = InferenceClient(provider="hf-inference",
                                              api_key=api_key)

        self._chat_messages: HFChatMessages = []
        self._prompts = 0
        self._debug_mode: Final = debug_mode

    def reset_exchanges(self) -> None:
        self._chat_messages = [{"role": "system", "content": "/no_think"}]

    def _add_prompt(self, prompt: Prompt) -> None:
        self._chat_messages.append({"role": "user", "content": str(prompt)})

    def _add_response(self, response: Response) -> None:
        self._chat_messages.append({
            "role": "assistant",
            "content": str(response)
        })

    def add_exchange(self, prompt: Prompt, response: Response) -> None:
        self._add_prompt(prompt)
        self._add_response(response)

    def _log_messages(self) -> None:
        if not self._debug_mode:
            return
        with open(f"msg{self._prompts}.json", 'w', encoding='utf-8') as f:
            json.dump(self._chat_messages, f, indent=2)

    def prompt(self, question: Prompt) -> Response | None:
        self._prompts += 1
        logging.debug(f"QUESTION: {str(question)}.")
        self._add_prompt(question)
        logging.debug(f"Sending {len(self._chat_messages)} message.")

        completion = self._client.chat.completions.create(  # pyright: ignore
            model=self._model,
            messages=self._chat_messages)

        answer = completion.choices[0].message.content
        assert isinstance(answer, str)
        logging.debug("RESPONSE -> " + answer)
        response = Response(answer)
        self._add_response(response)
        self._log_messages()
        return response


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
    def _load_config(story_file: str, participants_file: str) -> Config:
        config = Config()

        with open(story_file, 'r', encoding='utf-8') as file:
            story_data = yaml.safe_load(file)

        with open(participants_file, 'r', encoding='utf-8') as file:
            participants_data = yaml.safe_load(file)

        data = {**story_data, **participants_data}

        config.participants = data['participants']
        config.participants_data = data.get('participants-data', '')
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

    def __init__(self, story_file: str, participants_file: str, model: str,
                 debug_mode: bool) -> None:
        self._config: Final = self._load_config(
            story_file=story_file, participants_file=participants_file)
        self._llm_manager: Final = LlmStoryManager(model,
                                                   debug_mode=debug_mode)
        self._templates = Templates(self._config.lang)
        self._fill_templates()

        self._story: list[StoryCreator.StoryNode] = []
        self._append_new_delta_and_options(self._config.initial_participant_id,
                                           'starting-delta')

    def _fill_templates(self) -> None:
        t = self._templates
        config = self._config
        players_str = ",".join(config.participants)
        if config.participants_data:
            players_str += '.' + config.participants_data
        t.set_key('players', players_str)
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


def read_int(prompt: str) -> int | None:
    try:
        return int(input(prompt))
    except ValueError:
        return None
    except KeyboardInterrupt:
        sys.exit(0)


def input_option(max_opt: int) -> int:
    prompt = f"Tu elección? (0-{max_opt-1}): "
    while True:
        selected = read_int(prompt)
        if selected:
            if 0 <= selected < max_opt:
                return selected

            print("Elección fuera de rango.")
        else:
            print("Elección invalida.")


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

    sm.choose(input_option(len(options)))

    play_console(sm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="""
Choose Your Own Story multiplayer.\n\n
NOTE: You must define the environment variable HF_TOKEN with your Hugging Face token.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-s',
        '--story',
        type=str,
        default='story.yml',
        help='The filename of the story yaml file (default: story.yml)')

    parser.add_argument(
        '-p',
        '--participants',
        type=str,
        default='participants.yml',
        help=
        'The filename of the participants yaml file (default: participants.yml)'
    )

    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='HuggingFaceTB/SmolLM3-3B',
        help='The AI model to use (default: HuggingFaceTB/SmolLM3-3B)')

    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        help='Enable debug mode')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    sm = StoryCreator(story_file=args.story,
                      participants_file=args.participants,
                      model=args.model,
                      debug_mode=args.debug)

    play_console(sm)


main()
