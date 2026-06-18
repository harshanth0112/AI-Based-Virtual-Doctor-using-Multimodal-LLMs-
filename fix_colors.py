import glob, re
import os

path = r"c:\Users\acer\Downloads\medipredict_multimodal\medipredict2\templates\*.html"
files = glob.glob(path)

reps = {
    # Grays / Whites -> text vars
    re.compile(r'(?i)#F1F5F9'): 'var(--text)',
    re.compile(r'(?i)#E2E8F0'): 'var(--text)',
    re.compile(r'(?i)#CBD5E1'): 'var(--t2)',
    re.compile(r'(?i)#94A3B8'): 'var(--t2)',
    re.compile(r'(?i)#64748B'): 'var(--t3)',
    re.compile(r'(?i)#475569'): 'var(--t4)',
    
    # Accents (Light versions from dark mode -> Darker/Primary version for light mode)
    re.compile(r'(?i)#6EE7B7'): 'var(--green)',
    re.compile(r'(?i)#38BDF8'): 'var(--blue-d)',
    re.compile(r'(?i)#7DD3FC'): 'var(--blue)',
    re.compile(r'(?i)#0EA5E9'): 'var(--blue)',
    re.compile(r'(?i)#FCA5A5'): 'var(--red-d)',
    re.compile(r'(?i)#FCD34D'): 'var(--yellow)',
    
    # White alpha backgrounds/borders -> Black alpha 
    re.compile(r'rgba\(255,\s*255,\s*255,\s*\.04\)'): 'rgba(0,0,0,.02)',
    re.compile(r'rgba\(255,\s*255,\s*255,\s*\.08\)'): 'rgba(0,0,0,.06)',
    re.compile(r'rgba\(255,\s*255,\s*255,\s*\.1\)'): 'rgba(0,0,0,.15)',
    re.compile(r'rgba\(255,\s*255,\s*255,\s*\.14\)'): 'var(--border2)',
    re.compile(r'rgba\(255,\s*255,\s*255,\s*\.03\)'): 'var(--bg3)',
    re.compile(r'rgba\(255,\s*255,\s*255,\s*\.05\)'): 'rgba(0,0,0,.04)',
    re.compile(r'rgba\(255,\s*255,\s*255,\s*\.06\)'): 'rgba(0,0,0,.05)',
    re.compile(r'rgba\(255,\s*255,\s*255,\s*\.2\)'): 'rgba(0,0,0,.2)',
    re.compile(r'rgba\(255,\s*255,\s*255,\s*\.25\)'): 'rgba(0,0,0,.25)',
}

count = 0
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    original = content
    for pattern, rep in reps.items():
        content = pattern.sub(rep, content)
            
    if content != original:
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)
        count += 1
print(f"Updated {count} HTML templates.")
