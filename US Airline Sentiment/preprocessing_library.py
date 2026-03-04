"""
preprocessing_library.py
Reusable functions for cleaning tweets, handling emojis & contractions, repeated chars, etc.
"""

import re
import html
from typing import Optional
import emoji
import contractions

# ---------------------------
# Cleaning utility functions
# ---------------------------

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_PATTERN = re.compile(r'#(\w+)')
NUMBER_PATTERN = re.compile(r'\d+')
MULTI_WHITESPACE = re.compile(r'\s+')
REPEATED_CHARS = re.compile(r'(.)\1{2,}', re.DOTALL)  # 3+ repeats

def lowercase(text: str) -> str:
    return text.lower()

def remove_html_entities(text: str) -> str:
    return html.unescape(text)

def remove_urls(text: str) -> str:
    return URL_PATTERN.sub('', text)

def remove_mentions(text: str) -> str:
    return MENTION_PATTERN.sub('', text)

def handle_hashtags(text: str, keep_hash_tag_text: bool = False) -> str:
    # Option: keep hashtag text without '#' (e.g., #happy -> happy)
    if keep_hash_tag_text:
        return HASHTAG_PATTERN.sub(r'\1', text)
    else:
        return HASHTAG_PATTERN.sub('', text)

def remove_numbers(text: str, keep_numbers: bool = False) -> str:
    if keep_numbers:
        return text
    return NUMBER_PATTERN.sub('', text)

def remove_punct_and_special_chars(text: str) -> str:
    # keep basic word characters and whitespace
    return re.sub(r'[^A-Za-z0-9\s]', ' ', text)

def normalize_whitespace(text: str) -> str:
    return MULTI_WHITESPACE.sub(' ', text).strip()

def reduce_repeated_characters(text: str) -> str:
    # Reduce sequences like soooo -> so
    def repl(match):
        ch = match.group(1)
        return ch*2  # keep two occurrences (so -> so, but sooooo -> soo)
    return REPEATED_CHARS.sub(repl, text)

def expand_contractions(text: str) -> str:
    # contractions.fix uses a dictionary to expand common english contractions
    return contractions.fix(text)

def emoji_remove(text: str) -> str:
    return emoji.get_emoji_regexp().sub('', text)

def emoji_to_text(text: str) -> str:
    # convert emojis to :name: and remove colons
    return emoji.demojize(text).replace(":", " ")

# ---------------------------
# Master clean_text function
# ---------------------------

def clean_text(text: str,
               lowercase_flag: bool = True,
               remove_urls_flag: bool = True,
               remove_mentions_flag: bool = True,
               remove_hashtags_flag: bool = False,
               keep_hashtag_text: bool = True,
               remove_numbers_flag: bool = False,
               remove_emoji_flag: bool = False,
               emoji_to_text_flag: bool = False,
               expand_contractions_flag: bool = True,
               reduce_repeats_flag: bool = True,
               remove_punct_flag: bool = True) -> str:
    """
    Apply a pipeline of cleaning steps to text.
    Order of operations chosen to preserve semantics:
    1. html unescape
    2. optionally expand contractions
    3. optionally convert emoji to text OR remove emoji
    4. remove urls, mentions, hashtags
    5. remove punctuation and numbers (optional)
    6. reduce repeated chars, normalize whitespace, lowercase
    """

    if text is None:
        return ""

    txt = str(text)

    # decode HTML entities (&amp; -> &)
    txt = remove_html_entities(txt)

    if expand_contractions_flag:
        txt = expand_contractions(txt)

    if emoji_to_text_flag:
        txt = emoji_to_text(txt)
    elif remove_emoji_flag:
        txt = emoji_remove(txt)

    if remove_urls_flag:
        txt = remove_urls(txt)

    if remove_mentions_flag:
        txt = remove_mentions(txt)

    if remove_hashtags_flag:
        txt = handle_hashtags(txt, keep_hash_tag_text=keep_hashtag_text)
    else:
        # if not removing hashtags, still optionally turn into plain word by stripping '#'
        if keep_hashtag_text:
            txt = handle_hashtags(txt, keep_hash_tag_text=True)

    if remove_numbers_flag:
        txt = remove_numbers(txt, keep_numbers=False)

    if reduce_repeats_flag:
        txt = reduce_repeated_characters(txt)

    if remove_punct_flag:
        txt = remove_punct_and_special_chars(txt)

    txt = normalize_whitespace(txt)

    if lowercase_flag:
        txt = lowercase(txt)

    return txt
