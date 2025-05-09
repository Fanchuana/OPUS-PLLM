import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Optional
from transformers import PreTrainedTokenizer


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()
    Qwen_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[dict[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "### "
    sep2: str = None
    version: str = "Unknown"
    skip_next: bool = False
    tokenizer: PreTrainedTokenizer = None

    def get_prompt(self):
        messages = self.messages
        if type(messages) is not List[dict]:
            assert 'Messages List Should be List[dict]'
        if self.tokenizer is None or not hasattr(self.tokenizer, "apply_chat_template"):
            if self.sep_style == SeparatorStyle.SINGLE:
                ret = self.system + self.sep
                for message in messages:
                    role, content = message['role'], message['content']
                    if content and len(content)!=0:
                        ret += role + ": " + content + self.sep
                    else:
                        ret += role + ":"
            elif self.sep_style == SeparatorStyle.TWO:
                seps = [self.sep, self.sep2]
                ret = self.system + seps[0]
                for i, message in enumerate(messages):
                    role, content = message['role'], message['content']
                    if content and len(content)!=0:
                        ret += role + ": " + content + seps[i % 2]
                    else:
                        ret += role + ":"
            elif self.sep_style == SeparatorStyle.MPT:
                ret = self.system + self.sep
                for message in messages:
                    role, content = message['role'], message['content']
                    if content and len(content)!=0:
                        ret += role + content + self.sep
                    else:
                        ret += role
            elif self.sep_style == SeparatorStyle.LLAMA_2:
                wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
                wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
                ret = ""

                for i, message in enumerate(messages):
                    role, content = message['role'], message['content']
                    if i == 0:
                        assert content, "first message should not be none"
                        assert role == self.roles[0], "first message should come from user"
                    if content:
                        if i == 0: content = wrap_sys(self.system) + content
                        if i % 2 == 0:
                            content = wrap_inst(content)
                            ret += self.sep + content
                        else:
                            ret += " " + content + " " + self.sep2
                    else:
                        ret += ""
                ret = ret.lstrip(self.sep)
            elif self.sep_style == SeparatorStyle.LLAMA_3:
                raise NotImplementedError
            elif self.sep_style == SeparatorStyle.Qwen_2:
                raise NotImplementedError
            elif self.sep_style == SeparatorStyle.PLAIN:
                seps = [self.sep, self.sep2]
                ret = self.system
                for i, message in enumerate(messages):
                    role, content = message['role'], message['content']
                    if content:
                        ret += content + seps[i % 2]
                    else:
                        ret += ""
            else:
                raise ValueError(f"Invalid style: {self.sep_style}")
        else:
            ret = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        return ret
    def get_prompt_eval(self):
        if self.tokenizer is None:
            raise NotImplementedError
        return self.tokenizer.apply_chat_template(
            self.messages,
            tokenize = False,
            add_generation_prompt=True
        )

    def append_message(self, role, message):
        self.messages.append({'role':role, 'content':message})

    def _get_current_date(self, fallback: Optional[str] = None):
        from datetime import datetime
        try:
            # 格式说明: %d=日, %b=月份缩写, %Y=年
            return datetime.now().strftime("%d %b %Y")
        except Exception as e:
            return fallback if fallback is not None else "26 Jul 2024"

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[{'role':x,'content':y} for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
            tokenizer=self.tokenizer)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

default_chat_template = """
{% for message in messages %}
    {% if message['role'] == 'system' %}
        <|im_start|>system\n{{ message['content'] }}<|im_end|>\n
    {% elif message['role'] == 'user' %}
        <|im_start|>user\n{{ message['content'] }}<|im_end|>\n
    {% elif message['role'] == 'assistant' %}
        <|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}
"""

conv_vicuna_v0 = Conversation(
    system="A chat between a curious student and a biological professor who is familiar with protein properties. "
           "The biological professor gives helpful, detailed, and professional answers to student's questions.",
    roles=["Student", "Professor"],
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
conv_vicuna_v1 = Conversation(
    system="You are an automated protein annotation system that provides precise, database-validated identifiers in required formats. "
           "Responses are strictly concise and correct.",
    roles=["Student", "Professor"],
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v3 = Conversation(
    system="A chat between a curious user and a biological assistant who is familiar with protein properties. "
           "The biological assistant gives helpful, detailed, and professional answers to user's questions.",
    roles=["user", "assistant"],
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v2 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=["USER", "ASSISTANT"],
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

