import re

def get_sentences_splitters(txt):
    splitters =  ['? ', '! ', '. ', '.\n', '?\n', '!\n', ' ؟', '\n؟']
    all_sents = []
    last_sent_index= 0
    for i, (ch1, ch2) in enumerate(zip(txt, txt[1:])):
        if ch1 + ch2 in splitters:
            all_sents.append((txt[last_sent_index:i+len(ch1)],ch2))
            last_sent_index = i+ len(ch1+ch2)
    all_sents.append((txt[last_sent_index:], None))
    return [item[0] for item in all_sents], [item[1] for item in all_sents[:-1]]

def space_special_chars(txt):
    return re.sub('([.:،<>,!?()])', r' \1 ', txt)

def de_space_special_chars(txt):
    txt = re.sub('( ([.:،<>,!?()]) )', r'\2', txt)
    txt = re.sub('( ([.:،<>,!?()]))', r'\2', txt)
    return re.sub('(([.:،<>,!?()]) )', r'\2', txt)
