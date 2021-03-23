"""Mini-numpyro

http://pyro.ai/examples/minipyro.html
"""

from typing import Any, List, Dict, Generic, TypeVar


NUMPYRO_STACK = []
PARAM_STORE = {}


def get_param_store():
    return PARAM_STORE


class Messanger:
    def __init__(self) -> None:
        NUMPYRO_STACK.append(self)

    def __exit__(self, *args, **kwargs):
        assert NUMPYRO_STACK[-1] is self
        NUMPYRO_STACK.pop()

    def process_message(self, msg):
        pass

    def postprocess_message(self, msg):
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self:
            return self.fn(*args, **kwargs)




