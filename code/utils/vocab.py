import os

PRE_DEFINE_TOKENS = ['<pad>', '<unk>', '<sos>', '<eos>', '<sep>']


class Vocab:
    def __init__(self, vocab_file, max_size=-1):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0
        for w in PRE_DEFINE_TOKENS:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
        if not os.path.exists(vocab_file):
            raise Exception('vocab_file not exits!!!')
        with open(vocab_file, 'r', encoding='utf8') as fr:
            for line in fr:
                word = line.strip()
                if word in PRE_DEFINE_TOKENS:
                    raise Exception('predefined word %s should not in the vocab_file' % word)
                self._word_to_id[word] = self._count
                self._id_to_word[self._count] = word
                self._count += 1
                if max_size > 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading."
                          % (max_size, self._count))
                    break
        self.unk_id = self._word_to_id['<unk>']

    def word2id(self, word):
        return self._word_to_id.get(word, self._word_to_id['<unk>'])

    def words2ids(self, words):
        ids = []
        for word in words:
            ids.append(self.word2id(word))
        return ids

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('word id %d not found in vocab!!!' % word_id)
        return self._id_to_word[word_id]

    def ids2words(self, ids):
        words = []
        for id in ids:
            words.append(self.id2word(id))
        return words

    def size(self):
        return self._count

    def history2ids(self, history_words, oovs=None):
        ids = []
        if oovs is None:
            oovs = []
        for w in history_words:
            i = self.word2id(w)
            if i == self.unk_id:  # If w is OOV
                if w not in oovs:  # Add to list of OOVs
                    oovs.append(w)
                oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
                ids.append(
                    self.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
            else:
                ids.append(i)
        return ids, oovs

    def kg2ids(self, kg_words, oovs):
        ids = []
        for w in kg_words:
            i = self.word2id(w)
            if i == self.unk_id:  # If w is OOV
                if w not in oovs:  # Add to list of OOVs
                    oovs.append(w)
                oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
                ids.append(self.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
            else:
                ids.append(i)
        return ids, oovs

    def response2ids(self, response_words, oovs):
        ids = []

        for w in response_words:
            i = self.word2id(w)
            if i == self.unk_id:  # If w is an OOV word
                if w in oovs:  # If w is an in-article OOV
                    vocab_idx = self.size() + oovs.index(w)  # Map to its temporary article OOV number
                    ids.append(vocab_idx)
                else:  # If w is an out-of-article OOV
                    ids.append(self.unk_id)  # Map to the UNK token id
            else:
                ids.append(i)
        return ids

    def outputids2words(self, output_ids, oovs):
        words = []
        for id in output_ids:
            if id < self.size():
                words.append(self.id2word(id))
            else:
                words.append(
                    oovs[
                        id - self.size()
                    ]
                )
        return words