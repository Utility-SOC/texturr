"""Spotting responses that carry no content, so they do not pollute the groups."""

import re

# Exact (normalized) matches only. "no" is deliberately absent: it can be a real answer.
NON_ANSWERS = {
    'n/a', 'na', 'n.a.', 'none', 'nil', 'nothing', 'nope', 'idk', 'dk', 'test', 'asdf', 'x', 'xx', 'tbd',
    'no comment', 'no comments', 'no answer', 'no response', 'no opinion', 'not applicable',
    'nothing to add', 'nothing else', 'nothing to say', 'nothing really', 'i dont know', 'dont know',
}


def normalize(text):
    text = str(text).lower().strip().replace('’', "'")
    text = re.sub(r"[\s]+", ' ', text)
    return re.sub(r"^[\W_]+|[\W_]+$", '', text).replace("'", '') if re.search(r'\w', text) else text


def is_nonanswer(text):
    """True for empty, symbol-only, or boilerplate responses such as 'N/A', 'none', 'idk'."""
    raw = str(text).strip()
    if not re.search(r'[A-Za-z0-9]', raw):
        return True
    return normalize(raw) in NON_ANSWERS or raw.lower() in NON_ANSWERS
