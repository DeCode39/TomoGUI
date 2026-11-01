/* Read-only acceptance check against the existing eqall event catalog. */
const {chromium}=require(process.env.TOMOGUI_PLAYWRIGHT||'playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.TOMOGUI_BROWSER?{executablePath:process.env.TOMOGUI_BROWSER}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1140,height:1000}});
  page.setDefaultTimeout(15000);
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(process.env.TOMOGUI_URL||'http://127.0.0.1:8766');
  await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Ready'));
  await page.locator('#data-path').fill('EQcatalog/eqall.20000101-20250101.out');
  await page.locator('#detect').click();
  await page.waitForFunction(()=>document.querySelector('#detected').textContent==='Event catalog (no arrival picks)');
  await page.locator('#quality-column').selectOption('quality');
  await page.locator('#preview-quality').click();
  await page.waitForFunction(()=>document.querySelector('#quality-preview').textContent.includes('B -> 0.75'));
  if(!await page.locator('#prepare').isDisabled())throw Error('Event-only catalog preparation was enabled');
  const weights=await page.evaluate(async()=>{
   boot.token='expired';return api('quality',{spec:{column:'quality',mode:'grades',grades:[{grade:'a',weight:.8}]},values:['A']});
  });
  if(weights.weights[0]!==.8)throw Error('Session recovery failed');
  await page.screenshot({path:require('node:path').resolve(__dirname,'../.test-output/data-grades-live.png'),fullPage:true});
  console.log(JSON.stringify({gradePreview:await page.locator('#quality-preview').innerText(),eventCatalogBlocked:true,sessionRecovery:true,errors}));
  if(errors.length)throw Error(errors.join('\n'));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
