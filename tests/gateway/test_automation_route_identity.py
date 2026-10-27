"""A background wake compares the restored route with the last human message's source.

Only the fields that name the chat may decide that comparison. A routing origin that kept
the chat but not the sender (blank user_id / user_name / chat_name) used to refuse every
wake for that chat as ``admission_conflict``."""
from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_automation import _route_identity

CHAT = 'group:vIMJfkdg0L8fjj9ZiqCWdrF+eCAtE189bO3GUvtMMME='


def _human():
    return SessionSource(platform=Platform.SIGNAL, chat_id=CHAT, chat_type='group',
                         chat_name='Bible Genius Dev', user_id='+15550000000', user_name='E X',
                         user_id_alt='c3b7', chat_id_alt='vIMJ', message_id='17929', profile='manager')


def test_a_chat_only_origin_matches_the_last_human_message():
    origin = SessionSource(platform=Platform.SIGNAL, chat_id=CHAT, chat_type='group', profile='manager')
    assert _route_identity(origin) == _route_identity(_human())


def test_a_different_chat_thread_or_profile_still_conflicts():
    for change in ({'chat_id': 'group:other'}, {'thread_id': 't1'}, {'profile': 'qa'},
                   {'chat_type': 'dm'}):
        other = SessionSource(**{**dict(platform=Platform.SIGNAL, chat_id=CHAT, chat_type='group',
                                        profile='manager'), **change})
        assert _route_identity(other) != _route_identity(_human()), change
