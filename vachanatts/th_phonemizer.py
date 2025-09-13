from .TH_G2P import g2p
from .text_cleaner import clean_text
from .phoneme_ids import DEFAULT_PHONEME_ID_MAP
from pythainlp import word_tokenize

def phonemizer(text):

    cleaned_text = clean_text(text)
    cleaned_text = word_tokenize(cleaned_text)
    ipa = g2p(cleaned_text, "ipa")
    ipa = [p for p in ipa if p in DEFAULT_PHONEME_ID_MAP]

    sentence_endings = set([".", "?", "!", "ฯ", "…"])
    all_phonemes = []
    sentence_phonemes = []

    for p in ipa:
        sentence_phonemes.append(p)
        if p in sentence_endings:
            all_phonemes.append(sentence_phonemes)
            sentence_phonemes = []

    if sentence_phonemes:
        all_phonemes.append(sentence_phonemes)

    return all_phonemes
