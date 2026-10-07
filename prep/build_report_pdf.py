"""REPORT.md → REPORT.pdf (python-markdown 으로 HTML, headless Chrome 으로 인쇄)
Chrome: 환경변수 CHROME 또는 VS Code markdown-pdf 확장에 들어 있는 Chrome
"""
import os, glob, subprocess, tempfile
import markdown

K = '/data/knhyun/KAMP'
chrome = os.environ.get('CHROME') or sorted(glob.glob(os.path.expanduser(
    '~/.vscode-server/data/User/globalStorage/yzane.markdown-pdf/chrome/*/chrome-linux64/chrome')))[-1]
md = open(f'{K}/REPORT.md', encoding='utf-8').read()


def blank_before_blocks(text):
    """GitHub 은 문단 바로 다음 줄의 표·목록을 인식하지만 python-markdown 은 빈 줄이 필요 → 코드 블록 밖에서 빈 줄 삽입"""
    out, fence = [], False
    is_tbl = lambda l: l.lstrip().startswith('|')
    is_lst = lambda l: l.startswith(('- ', '* ')) or (l.split('. ')[0].isdigit() and '. ' in l)
    for l in text.split('\n'):
        if l.startswith('```'):
            fence = not fence
        if not fence and out and out[-1].strip() and not out[-1].startswith('```'):
            prev = out[-1]
            if (is_tbl(l) and not is_tbl(prev)) or (is_lst(l) and not is_lst(prev) and not prev.startswith((' ', '\t'))):
                out.append('')
        if not fence and l.startswith('  ') and l.lstrip().startswith(('- ', '* ')):   # 하위 목록: GitHub 2칸 → python-markdown 4칸
            n = len(l) - len(l.lstrip()); l = ' ' * (n * 2) + l.lstrip()
        out.append(l)
    return '\n'.join(out)


body = markdown.markdown(blank_before_blocks(md), extensions=['tables', 'fenced_code', 'sane_lists'])
css = '''
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'Noto Sans CJK KR', 'Noto Sans CJK JP', sans-serif; font-size: 9.5pt; line-height: 1.55; color: #111; }
h1 { font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 14pt; border-bottom: 1px solid #aaa; padding-bottom: 2px; margin-top: 22px; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 16px; page-break-after: avoid; }
h4 { font-size: 10.5pt; page-break-after: avoid; }
table { border-collapse: collapse; margin: 8px 0; font-size: 8.3pt; width: 100%; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #c8c8c8; padding: 3px 5px; vertical-align: top; }
th { background: #f0f0ee; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.3pt; background: #f4f4f2; padding: 0 2px; }
pre { background: #f4f4f2; padding: 6px 8px; overflow-wrap: anywhere; white-space: pre-wrap; font-size: 8pt; }
pre code { background: none; padding: 0; }
img { max-width: 100%; display: block; margin: 6px auto; page-break-inside: avoid; }
blockquote { color: #555; border-left: 3px solid #ccc; margin: 6px 0; padding: 2px 10px; }
'''
html = f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>KAMP X-ray 이물 탐지 보고서</title><style>{css}</style></head><body>{body}</body></html>'
with tempfile.NamedTemporaryFile('w', suffix='.html', dir=K, delete=False, encoding='utf-8') as f:   # 상대 경로 그림을 위해 K 에 둠
    f.write(html); tmp = f.name
try:
    subprocess.run([chrome, '--headless', '--no-sandbox', '--disable-gpu', '--no-pdf-header-footer',
                    f'--print-to-pdf={K}/REPORT.pdf', f'file://{tmp}'], check=True, capture_output=True, timeout=300)
finally:
    os.remove(tmp)
print(f'{K}/REPORT.pdf')
