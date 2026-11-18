/* Optional check after browser.cjs. All solver outputs below are fixtures. */
const path=require('node:path');
const {execFileSync}=require('node:child_process');
const {chromium}=require(process.env.TOMOGUI_PLAYWRIGHT||'playwright');
const project=path.resolve(__dirname,'../.test-output/browser-workspace-v020');
execFileSync(process.env.TOMOGUI_PYTHON||'python',['-c',`
import json, shutil, sys
from pathlib import Path
import yaml
from tomogui.differential import complete_baseline
root=Path(sys.argv[1]);assert root.name=='browser-workspace-v020'
state=json.loads((root/'.tomogui/project.json').read_text())
manifest=Path(state['run']['manifest']);data=json.loads(manifest.read_text())
folder=Path(data['root'])/'inversion_P'
for step in (1,2,3):
    params=yaml.safe_load((folder/f'3_input_params/input_params_joint_step{step}.yaml').read_text())
    out=folder/params['output_setting']['output_dir'];out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(data['root'])/'dataset/reference_P.dat',out/'src_rec_file_reloc_0100_obs.dat')
    if step==2:
        shutil.copyfile(next((folder/'2_models').glob('model_init*.h5')),out/'final_model.h5')
        kd=params['model_update']['optim_method_0']['Kdensity_coe']
complete_baseline(folder)
data['selected_kd']={'P':kd};data['validated']=True
manifest.write_text(json.dumps(data))
`,project],{stdio:'inherit'});
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.TOMOGUI_BROWSER?{executablePath:process.env.TOMOGUI_BROWSER}:{})});
 const page=await browser.newPage({viewport:{width:1140,height:1000}}),errors=[];
 page.on('pageerror',error=>errors.push(error.message));
 page.setDefaultTimeout(30000);
 await page.goto('http://127.0.0.1:8767');
 await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Ready'));
 await page.locator('[data-tab="run"]').click();
 for(const[id,value]of Object.entries({'dd-azimuth':'30','dd-distance':'10','dd-shift':'3'})){
  if(await page.locator('#'+id).inputValue()!==value)throw Error('Incorrect default '+id);
 }
 await page.locator('#dd-enabled').check();
 await page.locator('#dd-azimuth').fill('60');await page.locator('#dd-quality').fill('0.8');
 await page.locator('#dd-preview').click();
 await page.waitForFunction(()=>document.querySelector('#notice').textContent==='dd_preview completed.');
 const report=await page.evaluate(async()=>{const jobs=await(await fetch('/api/jobs')).json();return await(await fetch('/api/dd-report?id='+jobs.find(j=>j.operation==='dd_preview'&&j.status==='completed').id)).json();});
 if(!(await page.locator('#dd-status').textContent()).includes('CR'))throw Error('Missing readable pair counts');
 if(!report.ready||!report.phases.P.cr||!report.phases.P.cs)throw Error('Missing pair preview');
 await page.locator('#dd-prepare').click();
 await page.waitForFunction(()=>document.querySelector('#notice').textContent==='dd_prepare completed.',{},{timeout:120000});
 if(await page.locator('#dd-resolution').isDisabled())throw Error('Prepared resolution unavailable');
 await page.locator('#dd-azimuth').fill('50');
 if(!(await page.locator('#dd-resolution').isDisabled()))throw Error('Changed settings were not invalidated');
 await page.locator('#dd-azimuth').fill('60');
 await page.reload();await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Ready'));
 await page.locator('[data-tab="run"]').click();
 if(await page.locator('#dd-azimuth').inputValue()!=='60'||!(await page.locator('#dd-enabled').isChecked()))throw Error('Staged settings were not restored');
 await page.locator('#dd-run').click();
 await page.waitForFunction(()=>document.querySelector('#notice').textContent.includes('Score and accept'));
 await page.locator('#toggle-jobs').click();
 await page.screenshot({path:path.resolve(__dirname,'../.test-output/differential-live.png'),fullPage:true});
 console.log(JSON.stringify({errors,preview:report.phases.P,settingsRestored:true,unvalidatedRunBlocked:true},null,2));
 await browser.close();if(errors.length)process.exit(1);
})().catch(error=>{console.error(error);process.exit(1)});
