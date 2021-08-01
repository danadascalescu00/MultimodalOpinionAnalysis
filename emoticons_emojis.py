"""
    Emoticons data
"""
import re
from emoji import demojize

EMOTICONS = [
    ("Laughing", r':[‑,-]?\){2,}'),
    ("Rolling_on_the_floor_laughing", r'\=\){2,}|\=\]'),
    ("Heart", r'<3'),
    ("Broken_heart", r'<\\3'),
    ('Very_happy', r':[‑,-]?D'),
    ('Happy_face_or_smiley', r'[:,8,=][‑,-,o,O]?\)|\(\^[v,u,o,O]\^\)|:[‑,-]?3'),
    ('Happy', r'=]'),
    ('Mischievous_smile', r':[‑,-]?>'),
    ('Sticking_tongue_out_playfulness_or_cheekiness', r':P|:[‑,-]P|;P|:b|:-b'),
    ('Kiss', r':[‑,-]?[\*,X,x]'),
    ('Joy', r' uwu | UwU '),
    ('Surprised_or_shock', r':[‑,-]?[o|O|0]|o_O|o_0'),
    ('Sad_frown_andry_or_pouting', r':[‑,-]?\('),
    ('Very_sad', r':[(]{2,}'),
    ('Crying', r':[‑,-]?\'\('),
    ('Straight_face_no_expression_dissaproval_or_not_funny', r':[‑,-]?\|'),
    ('Annoyed_or_hesitant', r'>?[:][\\|\/]|\=\/|=\\'),
    ('Angel_saint_or_innocent', r'[0,O,o]:[‑,-]?[\),3]'),
    ('Embarrassed_or_blushing', r':\$'),
    ('Sad_or_crying', r';_;|\(;_;\)|\(\'_\'\)|Q_Q|\(;_:\)|\(:_;\)'),
    ('Evil_or_devilish', r'[>|}|3]:[‑,-]?\)'),
    ('Laughing_big_grin_or_laugh_with_glasses', r'[:,8,X,=][-,‑]?[D,3]|B\^D'),
    ('Tears_of_happiness', r':[\',\`][‑,-]?\)'),
    ('Horror', r'D[-,‑]\''),
    ('Great_dismay', r'D[8,;,=]'),
    ('Tongue_in_cheek', r':[-,‑]J'),
    ('Yawn', r'8[‑,-]0|>:O'),
    ('Sadness', r'D:'),
    ('Disgust', r'D:<'),
    ('Cool', r'\|;[‑,-]\)'),
    ('Drunk_or_confused', r'%[-,‑]?\)'),
    ('Sealed_lips_or_wearing_braces_or_tongue_tied', r':[-,‑]?[x,#,&]'),
    ('Skeptical_annoyed_undecided_uneasy_or_hesitant', r':[-,‑]?[.,/]|:[L,S]|=[/,L]'),
    ('Scepticism_disbelief_or_disapproval', r'\',:-\||\',:[-,-]'),
    ('Party_all_night', r'#‑\)'),
    ('Headphones_listening_to_music', r'\(\(d\[-_-\]b\)\)'),
    ('Bored', r'\|‑O'),
    ('Dump', r'<:‑\|'),
    ('Being_sick', r':-?#{2,3}..'),
    ('Amazed', r'\(\*_\*\)|\(\+_\+\)|\(\@_\@\)'),
    ('Confusion', r'\(\?_\?\)|\(\・\・?'),
    ('Wink_or_smirk', r';[-,‑]?[\),D,\]]|\*[-,‑]?\)|;\^\)|:‑,|;3'),
    ('Exciting', r'\\\(\^o\^\)\/|\\\(\^o\^\)\／|ヽ\(\^o\^\)丿|\(\*^0^\*\)|＼\(-o-\)／|＼\(~o~\)\／'),
    ('Giggling_with_hand_covering_mouth', r'\^m\^'),
    ('Joyful', r'\(\^_\^\)/|\(\^[O,o]\^\)／|\(°o°\)'),
    ('Tired', r'\(=_=\)'),
    ('Shame', r'\(-_-\)|\(一_一\)'),
    ('Surprised', r'\(o\.o\)'),
    ('Sleeping', r'\(-_-\)zzz'),
    ('Kowtow_as_a_sign_of_respect_or_dogeza_for_apology', r'\(__\)|_\(\._\.\)_|<\(_ _\)>|m\(_ _\)m|m\(__\)m|<m\(__\)m>|_\(_\^_\)_'),
    ('Troubled', r'\(>_<\)>?'),
    ('Nervous__Embarrassed_Troubled_Shy_Sweat_drop', r'\(-_-;\)|\(\^_\^;\)|\(-_-;\)|\(~_~;\)|\(・.・;\)|\(・_・;\)'),
    ('Wink', r'\(\^_-\)'),
    ('Normal_laugh', r'>\^_\^<|<\^!\^>|\(\^\.\^\)|\(\^J\^\)|\(\*\^[_,.]\^\*\)|\(\^<\^\)|\(\^\.\^\)|\(#\^\.\^#\)'),
    ('STH_ELSE', r'.')
]

emoticons_tokens = '|'.join('(?P<%s>%s)' % emoticon for emoticon in EMOTICONS)

def replace_emoticons(text):
    new_text = ""
    for match in re.finditer(emoticons_tokens, text):
        emoticon_name = match.lastgroup
        emoticon = match.group(emoticon_name)
        if emoticon_name == 'STH_ELSE':
            new_text += emoticon
        else:
            new_text += emoticon_name
    return new_text