from playwright.sync_api import sync_playwright
from pathlib import Path
import json
root=Path('/home/viplab/contest/final_project')
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1440,'height':1000}); errors=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://127.0.0.1:8765/inspector.html')
 page.locator('#pid').fill('SOFTWARE_QA');page.locator('#start').click()
 page.wait_for_function("document.getElementById('canvas').width>300")
 page.locator('#mode').select_option('regions')
 page.screenshot(path=str(root/'analysis/review_study_admin/ui_preview.png'),full_page=True)
 page.goto('http://127.0.0.1:8765/study.html')
 page.locator('#pid').fill('SOFTWARE_QA');page.locator('#start').click()
 page.wait_for_function("document.getElementById('canvas').width>300")
 assert page.locator('#next').is_disabled()
 page.locator('#canvas').click(position={'x':40,'y':40})
 assert page.locator('#next').is_enabled()
 page.locator('#undo').click();assert page.locator('#next').is_disabled()
 page.locator('#none').check();page.locator('#next').click()
 page.on('dialog',lambda d:d.accept())
 page.reload();page.locator('#pid').fill('SOFTWARE_QA');page.locator('#start').click()
 assert '2 / 66' in page.locator('#progressLabel').inner_text()
 for i in range(65):
  page.locator('#none').check();page.locator('#next').click()
 assert page.locator('#done').is_visible()
 with page.expect_download() as dl: page.locator('#exportDone').click()
 dest=root/'outputs/review_ui_qa/browser_export.json';dl.value.save_as(str(dest))
 data=json.loads(dest.read_text());assert len(data['results'])==66
 assert len({r['case_id'] for r in data['results']})==66
 assert not errors,errors
 data['software_test']=True;dest.write_text(json.dumps(data,ensure_ascii=False,indent=2))
 (root/'outputs/review_ui_qa/browser_checks.json').write_text(json.dumps({'passed':True,'cases':66,'errors':errors,'checks':['render','mode switching','click','undo','next validation','resume','completion','download'],'human_results':False},indent=2))
 b.close()
