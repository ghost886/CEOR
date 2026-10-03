import os
import hashlib
from tenacity import retry, wait_random_exponential, retry_if_not_exception_type

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional unless GPT metrics are used
    OpenAI = None


class KeyError(Exception):
    """OpenAIKey not provided in environment variable."""
    pass


CLIENT = None


def _get_client():
    """Create the optional OpenAI client only when a GPT metric is used."""
    global CLIENT  # pylint: disable=global-statement
    if OpenAI is None:
        raise ImportError("`openai` is required for GPT-based metrics.")
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise KeyError(
            'Need to provide OpenAI API key in environment variable '
            '`OPENAI_API_KEY`.'
        )
    if CLIENT is None:
        CLIENT = OpenAI(api_key=api_key)
    return CLIENT


@retry(retry=retry_if_not_exception_type(KeyError), wait=wait_random_exponential(min=1, max=10))
def predict(prompt, temperature=1.0, model='gpt-4'):
    """Predict with GPT models."""
    client = _get_client()

    if isinstance(prompt, str):
        messages = [
            {'role': 'user', 'content': prompt},
        ]
    else:
        messages = prompt

    if model == 'gpt-4':
        model = 'gpt-4-0613'
    elif model == 'gpt-4-turbo':
        model = 'gpt-4-1106-preview'
    elif model == 'gpt-3.5':
        model = 'gpt-3.5-turbo-1106'

    output = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=200,
        temperature=temperature,
    )
    response = output.choices[0].message.content
    return response


def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)
