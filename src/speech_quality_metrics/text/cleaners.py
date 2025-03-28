import re

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# Regular expression matching punctuation:
_punctuation_re = re.compile(r"[!\"#$%&\'\(\)\*\+,\-.\/:;<=>?@\[\\\]^_`{|}~]")


def lowercase(text):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)

def remove_punctuation(text):
    return re.sub(_punctuation_re, "", text)

def basic_cleaner(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)    
    text = remove_punctuation(text)
    text = collapse_whitespace(text)
    return text
