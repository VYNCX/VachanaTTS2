import re
import os

def eng_to_th_transliteration(english_text):
    """
    Transliterate English text into Thai phonetics.
    Supports large dictionaries, learning new words, and improved word splitting.
    """
    abs_dir = os.path.dirname(__file__)
    dict=abs_dir + "/data/dict_transliteration.txt"

    default_mappings = {
        "the": "เดอะ", "of": "ออฟ", "and": "แอนด์", "to": "ทู",
        "a": "เอ", "in": "อิน", "is": "อิส", "i": "ไอ", "you": "ยู",
        "he": "ฮี", "she": "ชี", "we": "วี", "they": "เดย์"
    }
    
    eng_to_thai = default_mappings.copy()
    try:
        with open(dict, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) == 2:
                        eng_key = parts[0].strip().lower()
                        thai_value = parts[1].strip()
                        eng_to_thai[eng_key] = thai_value
    except FileNotFoundError:
        print(f"Warning: Dataset '{dict}' not found. Using built-in mappings only.")
    
    text = english_text.lower().strip()
    
    words = re.findall(r"[a-zA-Z]+|[.,!?;'-]", text)
    
    result = []
    
    for word in words:
        if not word.strip():
            continue
        
        if re.match(r"[.,!?;'-]", word):
            result.append(word)
            continue
        
        if word in eng_to_thai:
            result.append(eng_to_thai[word])
            continue
        
        if word.isupper() and len(word) > 1:
            thai_word = "".join([eng_to_thai.get(c.lower(), c) for c in word])
            result.append(thai_word)
            continue
        
        thai_word = transliterate_compound_word(word, eng_to_thai)
        result.append(thai_word)
        
        if word not in eng_to_thai:
            eng_to_thai[word] = thai_word
            try:
                with open(dict, 'a', encoding='utf-8') as f:
                    f.write(f"{word},{thai_word}\n")
            except:
                pass

    return " ".join(result)

def transliterate_compound_word(word, mapping_dict):
    if word in mapping_dict:
        return mapping_dict[word]
    
    i = 0
    result = ""
    while i < len(word):
        matched = False
        for length in reversed(range(1, min(5, len(word) - i + 1))):  # try 4->1 chars
            substring = word[i:i+length]
            if substring in mapping_dict:
                result += mapping_dict[substring]
                i += length
                matched = True
                break
        if not matched:
            char = word[i]
            result += mapping_dict.get(char, char)
            i += 1
    return result

def transliterator(text):
    pattern = r'([ก-๙]+|[A-Za-z0-9&\'" ]+|[^\sA-Za-z0-9ก-๙]+|\n)'
    tokens = re.findall(pattern, text)
    tokens = [t for t in tokens if t.strip() != ""]

    processed = []
    prev_token = None
    for token in tokens:
        if re.match(r'^[ก-๙]+$', token):
            processed.append(token)
        elif re.match(r'^[A-Za-z0-9&\'" ]+$', token.strip()):
            if token.strip() and (prev_token is None or token.strip() != prev_token):
                translit = eng_to_th_transliteration(token)
                processed.append(translit)
                prev_token = token.strip()
            else:
                continue
        else:
            processed.append(token)
            prev_token = None
    return " ".join(processed)

if __name__ == "__main__":
    text = "This is thai example for you thank you."
    output = transliterator(text)
    print("Output:", output)
