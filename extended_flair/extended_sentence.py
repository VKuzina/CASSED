import flair.data
from flair.data import Token

class Sentence(flair.data.Sentence):
    def __init__(self, **kwargs):
        self.max_token = kwargs.pop('max_token')
        self.max_sentence_parts = kwargs.pop('max_sentence_parts')
        self.default_delimiter = kwargs.pop('default_delimiter')

        super().__init__(**kwargs)

    def add_tokens(self, tokens):
        for token in tokens:
            self.add_token(Token(token.text))

    def copy_part(self, sent):
        s = []
        for idx, token in enumerate(sent.tokens):
            if token.text == self.default_delimiter:
                break

        default_tokens = sent.tokens[:idx + 1]
        default_len = len(default_tokens)

        number_of_parts = min(len(sent.tokens) // self.max_token, self.max_sentence_parts)

        for part in range(number_of_parts + 1):
            cur = Sentence(max_token=self.max_token,
                           max_sentence_parts=self.max_sentence_parts,
                           default_delimiter=self.default_delimiter)
            cur.add_tokens(default_tokens)
            cur.add_tokens(sent.tokens[default_len + part * self.max_token: default_len + (part + 1) * self.max_token])

            s.append(cur)

        return s

