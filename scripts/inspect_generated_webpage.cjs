// Browser observations of the original generated page; never rewrites its source.
const {spawn}=require('node:child_process');
const fs=require('node:fs/promises');
const path=require('node:path');
const os=require('node:os');
const {randomUUID,createHash}=require('node:crypto');
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const root=path.resolve(__dirname,'..');
const arm=process.argv[2]||'base';
if(!['base','ui'].includes(arm))throw Error('Choose base or ui');
(async()=>{
  const directory=path.join(root,'results','webpage_design_v1');
  const record=JSON.parse(await fs.readFile(path.join(directory,`${arm}_result.json`),'utf8'));
  const sourceFileSha256=createHash('sha256').update(await fs.readFile(path.join(directory,`${arm}.html`))).digest('hex');
  if(record.html_file_sha256&&record.html_file_sha256!==sourceFileSha256)throw Error('Generated source file hash differs from its inference record');
  const browser=spawn('C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    ['--headless','--disable-gpu','--no-first-run','--no-default-browser-check',
     '--remote-debugging-port=9228',`--user-data-dir=${path.join(os.tmpdir(),'webpage-design-qa-'+randomUUID())}`,'about:blank'],
    {windowsHide:true,stdio:'ignore'});
  let socket;
  try{
    let pages;
    for(let i=0;i<100;i++){try{pages=await(await fetch('http://127.0.0.1:9228/json')).json();if(pages.some(p=>p.type==='page'))break;}catch{}await sleep(150);}
    const page=pages?.find(p=>p.type==='page');if(!page)throw Error('Browser did not start');
    socket=new WebSocket(page.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
    let sequence=0;const pending=new Map();const exceptions=[];const dialogs=[];
    socket.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')exceptions.push(m.params.exceptionDetails.text);if(m.method==='Page.javascriptDialogOpening'){dialogs.push(m.params.message);command('Page.handleJavaScriptDialog',{accept:true}).catch(()=>{});}if(m.id){const p=pending.get(m.id);if(p){pending.delete(m.id);m.error?p.reject(Error(m.error.message)):p.resolve(m.result);}}};
    function command(method,params={}){return new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));setTimeout(()=>{if(pending.has(id)){pending.delete(id);reject(Error('Timeout: '+method));}},15000).unref();});}
    async function evaluate(expression){const r=await command('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
    async function screenshot(name){const r=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});await fs.writeFile(path.join(directory,name),Buffer.from(r.data,'base64'));}
    await command('Page.enable');await command('Runtime.enable');
    await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await command('Page.navigate',{url:'http://127.0.0.1:8765/webpage-test/'+arm});
    for(let i=0;i<100;i++){if(await evaluate("document.readyState==='complete' && !!document.body"))break;await sleep(100);}
    await sleep(300);
    const desktop=await evaluate(`({title:document.title,width:innerWidth,scrollWidth:document.documentElement.scrollWidth,horizontalOverflow:document.documentElement.scrollWidth>innerWidth,headings:[...document.querySelectorAll('h1,h2,h3')].map(n=>n.textContent.trim()),navigation:!!document.querySelector('nav'),footer:!!document.querySelector('footer'),emailInputs:document.querySelectorAll('input[type=email]').length,externalAssets:[...document.querySelectorAll('[src],link[href]')].map(n=>n.getAttribute('src')||n.getAttribute('href')).filter(s=>s.startsWith('http:')||s.startsWith('https:')||s.startsWith('//'))})`);
    await screenshot(`${arm}_desktop.png`);
    await command('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
    await sleep(250);
    const mobile=await evaluate(`({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,horizontalOverflow:document.documentElement.scrollWidth>innerWidth,viewportMeta:document.querySelector('meta[name=viewport]')?.content||null})`);
    await screenshot(`${arm}_phone.png`);
    const interaction=await evaluate(`(async()=>{const input=document.querySelector('input[type=email]');const cta=[...document.querySelectorAll('a,button')].find(n=>/start planning/i.test(n.textContent));const beforeScroll=scrollY;if(cta)cta.click();await new Promise(r=>setTimeout(r,800));const afterScroll=scrollY;const result={startPlanningFound:!!cta,startPlanningHref:cta?.getAttribute('href')||null,scrollChanged:afterScroll!==beforeScroll,emailInputPresent:!!input};if(!input)return result;result.associatedLabel=!!(input.labels?.length||input.getAttribute('aria-label'));const form=input.form;result.formPresent=!!form;if(!form)return result;input.value='invalid-email';result.invalidEmailRejected=!input.checkValidity();input.value='student@example.test';input.dispatchEvent(new Event('input',{bubbles:true}));input.dispatchEvent(new Event('change',{bubbles:true}));result.validEmailAccepted=input.checkValidity();const before=document.body.innerText;const button=form.querySelector('button[type=submit],input[type=submit],button:not([type])');if(button)button.click();else form.requestSubmit();await new Promise(r=>setTimeout(r,400));result.pageTextChanged=document.body.innerText!==before;result.textAfterSubmit=document.body.innerText;return result;})()`);
    await screenshot(`${arm}_phone_after_submit.png`);
    await fs.writeFile(path.join(directory,`${arm}_rendered_dom.html`),await evaluate('document.documentElement.outerHTML'));
    const result={arm,source_html_sha256:record.html_sha256,source_file_sha256:sourceFileSha256,source_edited:false,desktop,mobile,interaction,scriptExceptions:exceptions,dialogs,limitation:'One browser observation of one generated page; network and external submissions blocked by preview policy.'};
    await fs.writeFile(path.join(directory,`${arm}_browser_checks.json`),JSON.stringify(result,null,2));
    console.log(JSON.stringify(result));
    await command('Browser.close').catch(()=>{});
  }finally{socket?.close();if(browser.exitCode===null)browser.kill();}
})().catch(e=>{console.error(e);process.exitCode=1;});
