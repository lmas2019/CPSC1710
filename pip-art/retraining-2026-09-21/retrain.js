function loadImage(src){return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error('Image failed: '+src));img.src=src;});}
function describe(img,size=24){
 const canvas=document.createElement('canvas');canvas.width=canvas.height=size;const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.fillStyle='#fff';ctx.fillRect(0,0,size,size);
 const scale=Math.min(size/img.naturalWidth,size/img.naturalHeight),w=img.naturalWidth*scale,h=img.naturalHeight*scale;ctx.drawImage(img,(size-w)/2,(size-h)/2,w,h);
 const rgba=ctx.getImageData(0,0,size,size).data,gray=new Float64Array(size*size),rgb=[],shade=[];
 for(let i=0;i<gray.length;i++)gray[i]=(rgba[i*4]*.299+rgba[i*4+1]*.587+rgba[i*4+2]*.114)/255;
 const mean=gray.reduce((a,b)=>a+b,0)/gray.length,sd=Math.sqrt(gray.reduce((a,b)=>a+(b-mean)**2,0)/gray.length)+.08,block=size/8;
 for(let y=0;y<size;y+=block)for(let x=0;x<size;x+=block){let sum=0,r=0,g=0,b=0;for(let j=0;j<block;j++)for(let i=0;i<block;i++){const p=(y+j)*size+x+i;sum+=gray[p];r+=rgba[p*4];g+=rgba[p*4+1];b+=rgba[p*4+2];}shade.push((sum/(block*block)-mean)/sd);rgb.push(r/(255*block*block),g/(255*block*block),b/(255*block*block));}
 const edges=new Float64Array(128),cellSize=size/4;for(let y=1;y<size-1;y++)for(let x=1;x<size-1;x++){const dx=gray[y*size+x+1]-gray[y*size+x-1],dy=gray[(y+1)*size+x]-gray[(y-1)*size+x];let angle=Math.atan2(dy,dx);if(angle<0)angle+=Math.PI;if(angle>=Math.PI)angle-=Math.PI;const cell=Math.floor(y/cellSize)*4+Math.floor(x/cellSize),bin=Math.min(7,Math.floor(angle/Math.PI*8));edges[cell*8+bin]+=Math.hypot(dx,dy);}
 const norm=Math.hypot(...edges)+1e-8;return {shade,rgb,edges:Array.from(edges,x=>x/norm)};
}
const vector=f=>[...f.shade.map(x=>x/8),...f.edges.map(x=>x*1.5),...f.rgb.map(x=>x/Math.sqrt(192)),1];
function probabilities(x,W,t=1){const z=W.map(w=>w.reduce((s,a,j)=>s+a*x[j],0)/t),m=Math.max(...z),e=z.map(v=>Math.exp(v-m)),sum=e.reduce((a,b)=>a+b,0);return e.map(v=>v/sum);}
function assess(samples,W,t=1){let correct=0,loss=0;const cm=Array.from({length:3},()=>[0,0,0]);for(const r of samples){const p=probabilities(r.x,W,t),got=p.indexOf(Math.max(...p));correct+=got===r.y;cm[r.y][got]++;loss-=Math.log(Math.max(1e-10,p[r.y]));}return {correct,total:samples.length,balanced:cm.reduce((s,row,i)=>s+row[i]/row.reduce((a,b)=>a+b,0),0)/3,loss:loss/samples.length,cm};}
function better(a,b){return !b||a.balanced>b.balanced+1e-9||(Math.abs(a.balanced-b.balanced)<1e-9&&a.loss<b.loss);}
function measure(epoch,W,train,valid,lambda,delta){const tr=assess(train,W),va=assess(valid,W);const counts=classes.map((_,i)=>train.filter(r=>r.y===i).length);let weightedLoss=0,l2=0;for(const r of train)weightedLoss-=Math.log(Math.max(1e-10,probabilities(r.x,W)[r.y]))/(3*counts[r.y]);for(const w of W)for(let j=0;j<w.length-1;j++)l2+=w[j]*w[j];return {epoch,trainingLoss:weightedLoss,objective:weightedLoss+lambda*l2/2,trainingCorrect:tr.correct,validationCorrect:va.correct,validationLoss:va.loss,weightChange:delta,weightNorm:Math.sqrt(W.reduce((s,w)=>s+w.reduce((v,a)=>v+a*a,0),0)),sampleWeights:[W[0][27],W[1][101],W[2][36]]};}

const classes=['bowls','cups','plates'];
async function main(){
 const rows=await (await fetch('training-data.json')).json();
 const config=await (await fetch('review-config.json')).json();
 const previous=await (await fetch('previous-learned-model.json')).json();
 const data=[];
 for(const row of rows){
  if(config.excludedHashes.includes(row.hash))continue;
  const label=config.labelOverrides[row.hash]||row.label;
  const img=await loadImage(row.image);
  data.push({...row,label,y:classes.indexOf(label),x:vector(describe(img,48))});
 }
 const originalTrain=data.filter(r=>r.split==='train'&&r.origin==='original');
 const train=data.filter(r=>r.split==='train'),valid=data.filter(r=>r.split==='valid');
 const fit=samples=>{
  const d=385,n=samples.length,counts=classes.map((_,i)=>samples.filter(r=>r.y===i).length);
  const W=Array.from({length:3},()=>new Float64Array(d));
  const log=[measure(0,W,samples,valid,.01,0)];
  for(let epoch=1;epoch<=600;epoch++){
   const grad=Array.from({length:3},()=>new Float64Array(d));
   for(const r of samples){const p=probabilities(r.x,W),balance=n/(3*counts[r.y]);for(let k=0;k<3;k++){const error=(p[k]-(r.y===k?1:0))*balance/n;for(let j=0;j<d;j++)grad[k][j]+=error*r.x[j];}}
   let delta=0;for(let k=0;k<3;k++)for(let j=0;j<d;j++){const step=.5*(grad[k][j]+(j===d-1?0:.01*W[k][j]));W[k][j]-=step;delta+=step*step;}
   if(epoch===1||epoch%25===0)log.push(measure(epoch,W,samples,valid,.01,Math.sqrt(delta)));
  }
  return {weights:W.map(w=>Array.from(w)),log,counts};
 };
 const reproduced=fit(originalTrain);
 const baselineDifference=Math.max(...reproduced.weights.flatMap((w,k)=>w.map((v,j)=>Math.abs(v-previous.weights[k][j]))));
 if(baselineDifference>1e-10)throw new Error('Baseline reproduction failed: '+baselineDifference);
 const candidate=fit(train);
 let temperature=1,bestLoss=Infinity;
 for(const t of [.5,.75,1,1.25,1.5,2,3,4]){const score=assess(valid,candidate.weights,t);if(score.loss<bestLoss){bestLoss=score.loss;temperature=t;}}
 const oldValidation=assess(valid,previous.weights,previous.temperature),newValidation=assess(valid,candidate.weights,temperature);
 // Decide using validation alone before evaluating either model on test photos.
 const accepted=better(newValidation,oldValidation);
 const test=data.filter(r=>r.split==='test');
 const report={createdAt:new Date().toISOString(),sourceExport:'pip-labeled-photos-2026-09-21.json',baselineDifference,settings:{size:48,lambda:.01,epochs:600,learningRate:.5,temperature},trainCount:train.length,counts:candidate.counts,addedCount:train.length-originalTrain.length,validation:{previous:oldValidation,candidate:newValidation},accepted,selectionRule:'Higher validation macro recall, breaking ties by calibrated validation loss; test results not used for selection.',test:{previous:assess(test,previous.weights,previous.temperature),candidate:assess(test,candidate.weights,temperature)},training:assess(train,candidate.weights,temperature),weights:candidate.weights,log:candidate.log,addedPhotos:train.filter(r=>r.origin==='visitor-export').map(r=>({source:r.source,label:r.label,hash:r.hash,previous:classes[probabilities(r.x,previous.weights,previous.temperature).indexOf(Math.max(...probabilities(r.x,previous.weights,previous.temperature)))],candidate:classes[probabilities(r.x,candidate.weights,temperature).indexOf(Math.max(...probabilities(r.x,candidate.weights,temperature)))]})),config};
 document.getElementById('report').textContent=JSON.stringify(report);document.body.dataset.done='true';
}
main().catch(e=>{document.getElementById('report').textContent=JSON.stringify({error:e.stack});document.body.dataset.done='error';});