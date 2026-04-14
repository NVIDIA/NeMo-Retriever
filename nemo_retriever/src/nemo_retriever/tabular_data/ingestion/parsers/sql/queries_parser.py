import codecs

import logging

logger = logging.getLogger(__name__)


def pre_process(q: str) -> str:
    q = codecs.decode(q, "unicode-escape")
    q = q.replace('\"', '"')
    q = q.replace('INSERT OVERWRITE INTO', 'INSERT INTO')
    q = q.replace('MERGE INTO IDENTIFIER(', 'MERGE INTO (')
    q = q.replace("U&'", "'")
    return q
