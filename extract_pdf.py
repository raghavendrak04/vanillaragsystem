from PyPDF2 import PdfReader

r = PdfReader('source/RegsNavyIV.pdf')
print(f'Total Pages: {len(r.pages)}')

# Get all text
full_text = ""
for i in range(len(r.pages)):
    text = r.pages[i].extract_text()
    if text:
        full_text += text + "\n"

# Print summary stats
print(f'Total characters: {len(full_text)}')
print(f'Total words: {len(full_text.split())}')

# Print table of contents / structure
print("\n=== DOCUMENT STRUCTURE (first 100 lines with chapter/section markers) ===")
lines = full_text.split('\n')
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped and (stripped[0].isdigit() or 'CHAPTER' in stripped.upper() or 'PART' in stripped.upper() or 'SECTION' in stripped.upper()):
        if len(stripped) < 200:
            print(f'  Line {i}: {stripped}')
