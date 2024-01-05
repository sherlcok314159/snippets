import re
from pathlib import Path

def extract_keys(text):
    matches = re.findall(r'\\cite\{(.*?)\}', text)
    seperate_matches = []
    for match in matches:
        match = [m.strip() for m in match.split(',')]
        seperate_matches.extend(match)
    return seperate_matches

def extract_key_from_bib(text):
    return re.findall(r'\{(.*?),', text)[0]

def load_keys_from_ref(file):
    keys = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('@'):
                keys.append(extract_key_from_bib(line))
    return keys

def load_dict_from_ref(file):
    ref_dict = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('@'):
                cache = []
                ref_key = extract_key_from_bib(line)
            cache.append(line)
            ref_dict[ref_key] = cache
    return ref_dict

files = ['abstract.tex', 'exp.tex', 'intro.tex', 'method.tex', 'motivation.tex', 'related.tex']

# 1. 加载已有的 references 以及 acl bib
ref_keys = load_keys_from_ref('references.bib')
ref_entry = load_dict_from_ref('anthology.bib')

# 2. 读取论文中所有需要的 citation key
all_extra_keys = []
for file in files:
    complete_path = Path('main', file)
    file_keys = []
    with open(complete_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'cite' in line:
                file_keys.extend(extract_keys(line))
    extra_keys = list(set(file_keys) - set(ref_keys))
    all_extra_keys.extend(extra_keys)

# 3. 将缺少的添加进最后的 bib 中
with open('references.bib', 'a', encoding='utf-8') as a:
    for key in set(all_extra_keys):
        entry = ref_entry[key]
        for e in entry:
            a.write(e)