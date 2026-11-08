// render .drawio -> vector PDF (and PNG preview) with mxGraph in Chromium
const { chromium } = require('playwright'); const fs=require('fs'); const path=require('path');
(async()=>{
 const b=await chromium.launch(); const p=await b.newPage();
 const lib=fs.readFileSync(require.resolve('mxgraph/javascript/mxClient.min.js'),'utf8');
 for (const f of process.argv.slice(2)){
  const xml=fs.readFileSync(f,'utf8');
  const W=+xml.match(/pageWidth="(\d+)"/)[1], H=+xml.match(/pageHeight="(\d+)"/)[1];
  await p.setViewportSize({width:W,height:H});
  await p.setContent('<html><head><style>@page{margin:0}html,body{margin:0;background:#fff}</style></head><body><div id=g style="width:'+W+'px;height:'+H+'px;overflow:hidden"></div></body></html>');
  await p.addScriptTag({content:'window.mxBasePath="";window.mxLoadResources=false;window.mxLoadStylesheets=false;'+lib});
  await p.evaluate((x)=>{const doc=mxUtils.parseXml(x);const m=doc.getElementsByTagName('mxGraphModel')[0];
    const g=new mxGraph(document.getElementById('g'));g.setHtmlLabels(true);new mxCodec(m.ownerDocument).decode(m,g.getModel());},xml);
  await p.waitForTimeout(600);
  const out=f.replace(/\.drawio$/,'');
  await p.pdf({path:out+'.pdf',width:W+'px',height:H+'px',printBackground:true,pageRanges:'1'});
  await p.screenshot({path:out+'_preview.png'});
 }
 await b.close();})();
