const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const ROOT=path.resolve(__dirname,'..');
const localModules=path.join(ROOT,'.publication_deps/node_modules');
const runtime=process.env.PAPER_NODE_MODULES || (fs.existsSync(path.join(localModules,'marked')) ? localModules : path.join(process.env.USERPROFILE||'', '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'));
const {chromium}=require(path.join(runtime,'playwright'));
const katex=require(path.join(ROOT,'.publication_deps/node_modules/katex'));
(async()=>{
 const {marked}=await import(pathToFileURL(path.join(runtime,'marked/lib/marked.esm.js')).href);
 marked.use({extensions:[
  {name:'displayMath',level:'block',start(s){return s.indexOf('$$')},tokenizer(s){const m=/^\$\$([\s\S]+?)\$\$(?:\n|$)/.exec(s);if(m)return {type:'displayMath',raw:m[0],text:m[1]};},renderer(t){return katex.renderToString(t.text.replace(/–/g,"-"),{displayMode:true,throwOnError:false});}},
  {name:'inlineMath',level:'inline',start(s){return s.indexOf('$')},tokenizer(s){const m=/^\$(?!\$)((?:\\.|[^$\n])+?)\$(?!\$)/.exec(s);if(m)return {type:'inlineMath',raw:m[0],text:m[1]};},renderer(t){return katex.renderToString(t.text.replace(/–/g,"-"),{throwOnError:false});}}
 ]});
 const markdown=fs.readFileSync(path.join(ROOT,'PAPER.md'),'utf8');
 const cssurl=pathToFileURL(path.join(ROOT,'.publication_deps/node_modules/katex/dist/katex.min.css')).href;
 const html=`<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="${cssurl}"><style>
 @page{size:A4;margin:17mm 18mm 20mm}body{font-family:Georgia,'Times New Roman',serif;font-size:10pt;line-height:1.43;color:#181818}h1{font-size:20pt;line-height:1.18;margin:0 0 10pt}h2{font-size:14pt;margin:19pt 0 7pt;break-after:avoid}h3{font-size:11.5pt;margin:14pt 0 6pt;break-after:avoid}h4{font-size:10.5pt;margin:10pt 0 5pt;break-after:avoid}p{margin:0 0 8pt;orphans:3;widows:3}li{margin:3pt 0}blockquote{border-left:3px solid #777;padding:4pt 8pt;margin:8pt 0;background:#f4f4f4}table{border-collapse:collapse;width:100%;font-size:8.3pt;margin:8pt 0 12pt;table-layout:auto;break-inside:auto}tr{break-inside:avoid}thead{display:table-header-group}th,td{border-bottom:0.5pt solid #aaa;padding:4pt 3pt;vertical-align:top;overflow-wrap:anywhere}th{border-top:1pt solid #333;border-bottom:1pt solid #333;text-align:left}a{color:#174778;text-decoration:none;overflow-wrap:anywhere}code{font:8.5pt Consolas,monospace;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:8pt}.katex-display{overflow-wrap:anywhere;margin:7pt 0;font-size:.9em}hr{border:0;border-top:0.5pt solid #aaa;margin:12pt 0}small{font-size:8pt}strong{font-weight:bold}
 </style></head><body>${marked.parse(markdown)}</body></html>`;
 fs.mkdirSync(path.join(ROOT,'publication','build'),{recursive:true});fs.writeFileSync(path.join(ROOT,'publication','build','paper.html'),html);
 const browser=await chromium.launch({executablePath:process.env.PAPER_BROWSER || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,args:['--no-first-run']});
 try{const page=await browser.newPage();await page.goto(pathToFileURL(path.join(ROOT,'publication','build','paper.html')).href,{waitUntil:'load'});await page.evaluate(()=>document.fonts.ready);fs.mkdirSync(path.join(ROOT,'output','pdf'),{recursive:true});await page.pdf({path:path.join(ROOT,'output','pdf','compositional_inference_paper.pdf'),format:'A4',printBackground:true,displayHeaderFooter:true,headerTemplate:'<span></span>',footerTemplate:'<div style="font:8px Georgia;width:100%;text-align:center;color:#777"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',margin:{top:'17mm',bottom:'20mm',left:'18mm',right:'18mm'}});console.log('PDF generated');}
 finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});