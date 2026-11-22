"""Parse Mermaid in existing local Chrome; no packages or browsers are installed.

Loads a pinned Mermaid browser script from jsDelivr, never uploads diagram content.
All network requests are blocked after loading the renderer. Writes SVG and local previews.
The statechart is syntax-checked, not model-checked or rendered by a replacement pstack renderer.
"""
import ast
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'docs'/'research_validation'/'experiment_04'/'diagrams'


def verify_ref(ref):
    if ref.startswith('conn_exp.'):
        parts=ref.split('.')[1:]
        file=ROOT/'conn_experiment_01'/'src'/'conn_exp'/(parts[0]+'.py')
    else:
        parts=ref.split('.')
        file=ROOT/'experiments'/'experiment_04'/'src'/'conn_exp04'/(parts[0]+'.py')
    if not file.exists(): raise AssertionError(ref)
    tree=ast.parse(file.read_text(encoding='utf-8'))
    for part in parts[1:]:
        tree=next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==part)


def main():
    records=json.loads((OUT/'catalog.json').read_text())
    results=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True)
        page=browser.new_page(viewport={'width':1800,'height':1200},device_scale_factor=1)
        page.goto('about:blank')
        page.add_script_tag(url='https://cdn.jsdelivr.net/npm/mermaid@11.12.1/dist/mermaid.min.js')
        page.evaluate("mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'default',fontFamily:'Arial',flowchart:{htmlLabels:false}})")
        page.route('**/*',lambda route:route.abort())
        for i,r in enumerate(records):
            for ref in r['references']: verify_ref(ref)
            code=(OUT/(r['name']+'.mmd')).read_text()
            page.evaluate('(code)=>mermaid.parse(code)',code)
            result={'name':r['name'],'parsed':True,'source_references_verified':True}
            if 'stateDiagram' not in code:
                svg=page.evaluate('async ({code,id})=>(await mermaid.render(id,code)).svg',{'code':code,'id':f'exp04_{i}'})
                (OUT/(r['name']+'.svg')).write_text(svg,encoding='utf-8')
                page.set_content(svg)
                box=page.locator('svg').bounding_box()
                page.locator('svg').screenshot(path=str(OUT/(r['name']+'.png')))
                result.update(rendered=True,width=box['width'],height=box['height'])
            else: result.update(rendered=False,note='Standard Mermaid syntax parsed; pstack runtime unavailable; no formal checking.')
            results.append(result)
        browser.close()
    (OUT/'validation.json').write_text(json.dumps({'mermaid':'11.12.1','formal_model_checking':False,'results':results},indent=2)+'\n')
    print('Validated',len(results),'diagrams; source references verified; seven rendered, statechart parsed.')


if __name__=='__main__': main()
