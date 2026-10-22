"""Inspect the publisher's public download page, without using credentials."""
import html,json,re,urllib.request
from pathlib import Path
WORK=Path('D:/huawei2026-experiments/room_support_20261021')
url='https://doi.org/10.21227/xbhv-f125'
with urllib.request.urlopen(url,timeout=60) as r:
    data=r.read().decode();final=r.url
(WORK/'dataport.html').write_text(data,encoding='utf-8')
text=html.unescape(re.sub('<[^>]+>',' ',data));text=re.sub(r'\s+',' ',text)
findings=[]
for pattern in ('download','login','log in','subscription','.zip','.mat','license','licence'):
    for m in list(re.finditer(re.escape(pattern),text,re.I))[:12]:findings.append(text[max(0,m.start()-150):m.end()+300])
links=re.findall(r'href=[\"\']([^\"\']+)[\"\']',data)
relevant=[html.unescape(x) for x in links if any(w in x.lower() for w in ('download','login','.zip','.mat','amazonaws','license'))]
result=dict(url=final,findings=findings,links=relevant)
(WORK/'dataport_access.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
