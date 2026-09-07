import re, sys
path = sys.argv[1]
t = open(path).read()
t = re.sub(r'(?m)^%.*$', '', t)
def words(s):
    s = re.sub(r'\\cite\{[^}]*\}', '', s)
    s = re.sub(r'\\(ref|label|eqref)\{[^}]*\}', 'X', s)
    s = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', s)
    s = re.sub(r'\$[^$]*\$', 'X', s)
    s = re.sub(r'[{}~]', ' ', s)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-\.%/]*", s))
abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', t, re.S).group(1)
impact = re.search(r'Impact Statement---\}(.*?)\}\\\\', t, re.S)
print("abstract words:", words(abstract))
if impact: print("impact statement words:", words(impact.group(1)))
body = t.split(r'\section{Introduction}')[1].split(r'\section*{Supplementary Materials}')[0]
floats = re.findall(r'\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}', body, re.S)
nofloat = re.sub(r'\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}', '', body, flags=re.S)
print("BODY prose words (Intro..Conclusion, no floats):", words(nofloat))
for name, chunk in re.findall(r'\\section\{([^}]*)\}(.*?)(?=\\section\{|\Z)', nofloat, re.S):
    print(f"   {name}: {words(chunk)}")
caps = re.findall(r'\\caption\{(.*?)\}\s*\\label', body, re.S)
notes = re.findall(r'\\begin\{flushleft\}(.*?)\\end\{flushleft\}', body, re.S)
print("captions:", len(caps), "words:", sum(words(c) for c in caps), "| table notes words:", sum(words(n) for n in notes))
supp = t.split(r'\section*{Supplementary Materials}')[1].split(r'\bibliographystyle')[0]
print("supp pointer + acknowledgment words:", words(supp))
print("display items: figures", len(re.findall(r'\\begin\{figure', body)), "tables", len(re.findall(r'\\begin\{table', body)))
keys = set(k.strip() for c in re.findall(r'\\cite\{([^}]*)\}', t) for k in c.split(','))
print("distinct citation keys:", len(keys))
