var qn=Object.defineProperty;var jn=(i,e,t)=>e in i?qn(i,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):i[e]=t;var h=(i,e,t)=>jn(i,typeof e!="symbol"?e+"":e,t);class Jn{constructor(){h(this,"dims",[]);h(this,"paddedDims",[]);h(this,"layout","x");h(this,"dataType","Float32")}getByteSize(){let e=1;for(const t of this.paddedDims)e*=t;return this.dataType==="Float32"?e*=4:this.dataType==="Float16"&&(e*=2),e}}class Qn{constructor(e,t){h(this,"desc");h(this,"data");this.desc=e,this.data=t}}class Zn{constructor(e){h(this,"_view");h(this,"offset",0);this._view=e}read(e){const t=this._view,n=this.offset;switch(this.offset+=e,e){case 1:return t.getUint8(n);case 2:return t.getUint16(n,!0);case 4:return t.getUint32(n,!0);case 8:return Number(t.getBigUint64(n,!0));default:throw new Error("unsupported read size")}}}function ei(i){const e=new Uint8Array(i),t=new Zn(new DataView(i));if(t.read(2)!==16855)throw new Error("invalid or corrupted weights blob");const o=t.read(1);if(t.read(1),o!==2)throw new Error("unsupported weights blob version");const r=t.read(8);t.offset=r;const s=t.read(4),a=new Map;for(let u=0;u<s;++u){const l=new Jn,p=t.read(2),c=new TextDecoder().decode(e.subarray(t.offset,t.offset+p));t.offset+=p;const f=t.read(1);for(let x=0;x<f;++x)l.dims.push(t.read(4));l.paddedDims=[...l.dims],new TextDecoder().decode(e.subarray(t.offset,t.offset+f))==="oihw"&&(l.layout="oihw"),t.offset+=f;const g=String.fromCharCode(t.read(1));if(g==="f")l.dataType="Float32";else if(g==="h")l.dataType="Float16";else throw new Error("invalid tensor data type");const _=t.read(8),y=e.slice(_,_+l.getByteSize());a.set(c,new Qn(l,y))}return a}function ti(i,e){return i.channels===e.channels}const Me=8;class ze{constructor(e,t,n){h(this,"autoUpdateOutputBuffer",!0);h(this,"_label");h(this,"_device");h(this,"_outputBuffers",{});h(this,"_pipeline");h(this,"_bindGroups",[]);h(this,"_needsUpdatePipeline",!0);h(this,"_needsResizeBuffer",!0);h(this,"_inputs",[]);h(this,"_outputs",[]);h(this,"_uniforms",[]);h(this,"_uniformBuffers",{});h(this,"_width",10);h(this,"_height",10);h(this,"_execWidth");h(this,"_execHeight");h(this,"_csCode","");h(this,"_csMain");h(this,"_csDefine");h(this,"_groupOffsets",{inputs:0,uniforms:1,outputs:2});this._label=e,this._device=t,this._csMain=n.csMain,this._csDefine=n.csDefine,this._inputs=n.inputs,this._outputs=n.outputs,this._uniforms=n.uniforms,this.autoUpdateOutputBuffer=n.autoUpdateOutputBuffer??!0,n.uniforms.forEach(o=>{this._uniformBuffers[o.label]=t.createBuffer({label:this._label,size:o.data.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this._device.queue.writeBuffer(this._uniformBuffers[o.label],0,o.data)})}setCSCode({csDefine:e,csMain:t}){this._csDefine=e,this._csMain=t,this._needsUpdatePipeline=!0}setSize(e,t){e=Math.ceil(e),t=Math.ceil(t);const n=e!==this._width||t!==this._height;this._width=e,this._height=t,n&&(this._needsResizeBuffer=!0,this._needsUpdatePipeline=!0)}setExecuteSize(e,t){e=Math.ceil(e),t=Math.ceil(t),this._execWidth=e,this._execHeight=t}setOutputParams(e){this.autoUpdateOutputBuffer&&this._updateOutputBuffers(e),this._needsUpdatePipeline=!0}setOutputBuffers(e){this._outputBuffers=Object.keys(e).reduce((t,n)=>(t[n]={buffer:e[n],params:{channels:4}},t),{})}setUniform(e,t){const n=this._uniformBuffers[e];this._device.queue.writeBuffer(n,0,t)}getOutput(e){return this._needsResizeBuffer&&this.autoUpdateOutputBuffer&&(this._resizeOutputBuffers(),this._needsResizeBuffer=!1),this._outputBuffers[e].buffer}dispose(e=!0){Object.keys(this._uniformBuffers).forEach(t=>{this._uniformBuffers[t].destroy()}),e&&Object.keys(this._outputBuffers).forEach(t=>{this._outputBuffers[t].buffer.destroy()})}_createBuffer(e){const t=this._width*this._height*4*4;return this._device.createBuffer({label:this._label,size:Math.max(t,80),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC})}_resizeOutputBuffers(){const e=this._outputBuffers;for(const t in e){const{buffer:n,params:o}=e[t];n.destroy(),e[t].buffer=this._createBuffer(o)}}_updateOutputBuffers(e){var n,o;const t=this._outputBuffers;for(const r in e){const s=e[r];if(!ti(s,((n=t[r])==null?void 0:n.params)||{})){(o=t[r])==null||o.buffer.destroy();const a=this._createBuffer(s);t[r]={buffer:a,params:s}}}}_updatePipeline(e,t){if(!this._needsUpdatePipeline)return;this._needsUpdatePipeline=!1;const n=this._device,o=this._getFullCs(e,t);o!==this._csCode&&(this._csCode=o,this._pipeline=n.createComputePipeline({label:this._label,layout:"auto",compute:{module:n.createShaderModule({label:this._label,code:o}),entryPoint:"main"}}),this._updateBindGroups())}_getFullCs(e,t){const n=this._inputs,o=this._uniforms;let r=0;const s=this._groupOffsets={inputs:0,uniforms:0,outputs:0};return n.length>0&&r++,o.length>0&&(s.uniforms=r,r++),s.outputs=r,`
${n.sort().map((u,l)=>{const p=`@group(${s.inputs}) @binding(${l}) `,c=`in_${u}`;return t[u]==="texture"?`${p} var ${c}: texture_2d<f32>;`:`${p} var<storage, read> ${c}: array<vec${e[u].channels}f>;`}).join(`
`)}
${this._uniforms.map((u,l)=>`@group(${s.uniforms}) @binding(${l}) var<uniform> ${u.label}: ${u.type};`).join(`
`)}

${this._outputs.map((u,l)=>`@group(${s.outputs}) @binding(${l}) var<storage, read_write> out_${u}: array<vec${this._outputBuffers[u].params.channels}f>;`).join(`
`)}
${this._csDefine??""}
@compute @workgroup_size(${Me}, ${Me}, 1)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
${this._csMain}
}
`}_updateBindGroups(){const e=[],t=this._device,n=this._groupOffsets;this._uniforms.length>0&&(e[n.uniforms]=t.createBindGroup({label:this._label,layout:this._pipeline.getBindGroupLayout(n.uniforms),entries:this._uniforms.map((o,r)=>({binding:r,resource:{buffer:this._uniformBuffers[o.label]}}))})),this._bindGroups=e}createPass(e,t){this._needsResizeBuffer&&this.autoUpdateOutputBuffer&&(this._resizeOutputBuffers(),this._needsResizeBuffer=!1);const n=this._inputs.reduce((s,a)=>(s[a]=t[a].buffer?"buffer":"texture",s),{});this._updatePipeline(t,n);const o=this._groupOffsets;this._inputs.length>0&&(this._bindGroups[o.inputs]=this._device.createBindGroup({label:this._label,layout:this._pipeline.getBindGroupLayout(o.inputs),entries:this._inputs.map((s,a)=>({binding:a,resource:t[s].buffer?{buffer:t[s].buffer}:t[s].texture.createView()}))})),this._bindGroups[o.outputs]=this._device.createBindGroup({label:this._label,layout:this._pipeline.getBindGroupLayout(o.outputs),entries:this._outputs.map((s,a)=>({binding:a,resource:{buffer:this._outputBuffers[s].buffer}}))});const r=e.beginComputePass();r.setPipeline(this._pipeline),this._bindGroups.forEach((s,a)=>{r.setBindGroup(a,s)}),r.dispatchWorkgroups(Math.ceil((this._execWidth??this._width)/Me),Math.ceil((this._execHeight??this._height)/Me),1),r.end()}}const xt=1412.83765,bt=1.64593172,$t=.431384981,kt=-.00294139609,Bt=.192653254,It=.00626026094,Pt=.998620152,fn=15794576e-13,dn=.0322087631,hn=.00223151711,gn=.370974749;function _n(i){return i<=fn?i=xt*i:i<=dn?i=bt*Math.pow(i,$t)+kt:i=Bt*Math.log(i+It)+Pt,i}function ni(i){return i<=hn?i=i/xt:i<=gn?i=Math.pow((i-kt)/bt,1/$t):i=Math.exp((i-Pt)/Bt)-It,i}const ii=65504,yn=_n(ii),mn=1/yn,wn=yn;class rt{constructor(e,t,n,o){h(this,"x");h(this,"y");h(this,"width");h(this,"height");this.x=e,this.y=t,this.width=n,this.height=o}}function ri({data:i,channels:e}){let t=0;for(let s=0;s<i.length;s+=e){const a=i[s],u=i[s+1],l=i[s+2],p=.212671*a+.71516*u+.072169*l;t+=Math.log2(p+1e-4)}const n=i.length/e,o=t/n;return .18/Math.pow(2,o)}function oi({data:i,channels:e,inputScale:t}){const n=new Float32Array(i.length);n.set(i);for(let o=0;o<n.length;o+=e)for(let r=0;r<3;r++){let s=n[o+r]*t;n[o+r]=_n(s)*mn}return n}function si({data:i,channels:e,inputScale:t}){const n=new Float32Array(i.length);n.set(i);const o=1/t;for(let r=0;r<n.length;r+=e)for(let s=0;s<3;s++){let a=n[r+s]*wn;n[r+s]=ni(a)*o}return n}const zt=`
const a = ${xt};
const b = ${bt};
const c = ${$t};
const d = ${kt};
const e = ${Bt};
const f = ${It};
const g = ${Pt};
const y0 =${fn};
const y1 =${dn};
const x0 =${hn};
const x1 =${gn};

const normScale = ${mn};
const rcpNormScale = ${wn};
`;class ai{constructor(e,t){h(this,"_device");h(this,"_isHDR");h(this,"_inputPassAux");h(this,"_inputPassColor");h(this,"_outputPass");h(this,"_copyPass");h(this,"_isInputTexture");this._device=e,this._isHDR=t;const n=[{label:"inputScale",type:"f32",data:new Float32Array([1])},{label:"inputSize",type:"vec2i",data:new Int32Array(2)},{label:"outputSize",type:"vec2i",data:new Int32Array(2)},{label:"inputOffset",type:"vec2i",data:new Int32Array(2)}];this._inputPassAux=new ze("inputPassAux",this._device,{inputs:["color","albedo","normal"],outputs:["color","albedo","normal"],uniforms:n,csDefine:"",csMain:""}),this._inputPassColor=new ze("inputPassColor",this._device,{inputs:["color"],outputs:["color"],uniforms:n,csDefine:"",csMain:""}),this._outputPass=new ze("outputPass",this._device,{inputs:["color","raw"],outputs:["color"],uniforms:[{label:"inputScale",type:"f32",data:new Float32Array([1])},{label:"inputSize",type:"vec2i",data:new Int32Array(2)},{label:"outputSize",type:"vec2i",data:new Int32Array(2)},{label:"imageSize",type:"vec2i",data:new Int32Array(2)},{label:"inputOffset",type:"vec2i",data:new Int32Array(2)},{label:"outputOffset",type:"vec2i",data:new Int32Array(2)}],csDefine:"",csMain:""}),this._copyPass=new ze("copyPass",this._device,{inputs:["color"],outputs:["color"],autoUpdateOutputBuffer:!1,uniforms:[{label:"size",type:"vec2i",data:new Int32Array(2)}],csMain:`
let outIdx = i32(globalId.x + globalId.y * u32(size.x));
out_color[outIdx] = textureLoad(in_color, globalId.xy, 0);
`}),this._inputPassAux.setOutputParams({color:{channels:3},albedo:{channels:3},normal:{channels:3}}),this._inputPassColor.setOutputParams({color:{channels:3}}),this._outputPass.setOutputParams({color:{channels:4}})}_updatePasses(e,t=!1){if(this._isInputTexture!=null&&this._isInputTexture===e)return;this._isInputTexture=e;const n=this._isHDR,o=`
${zt}
fn PUForward(y: f32) -> f32 {
  if (y <= y0) {
    return a * y;
  } else if (y <= y1) {
    return b * pow(y, c) + d;
  } else {
    return e * log(y + f) + g;
  }
}`;function r(a){return e?`textureLoad(in_${a}, globalId.xy + vec2u(inputOffset), 0)`:`in_${a}[inIdx]`}const s=`
let x = i32(globalId.x);
let y = i32(globalId.y);
let inIdx = (y + inputOffset.y) * inputSize.x + (x + inputOffset.x);
let col = ${r("color")};

let outIdx = y * outputSize.x + x;

if (${t}) {
  // Denoise the inversed alpha. Or the anti aliased edge will be too dark after denoised
  out_color[outIdx] = vec3f(1.0 - col.a);
}
else if (${n}) {
  out_color[outIdx] = vec3f(PUForward(col.r * inputScale), PUForward(col.g * inputScale), PUForward(col.b * inputScale)) * normScale;
}
else {
  out_color[outIdx] = col.rgb;
}
`;this._inputPassAux.setCSCode({csDefine:o,csMain:`
${s}
let alb = ${r("albedo")};
let nor = ${r("normal")};
out_normal[outIdx] = nor.rgb;
out_albedo[outIdx] = alb.rgb;
  `}),this._inputPassColor.setCSCode({csDefine:o,csMain:`
${s}
`}),this._outputPass.setCSCode({csDefine:`
${zt}
fn PUInverse(y: f32) -> f32 {
  if (y <= x0) {
    return y / a;
  } else if (y <= x1) {
    return pow((y - d) / b, 1 / c);
  } else {
    return exp((y - g) / e) - f;
  }
}
`,csMain:`
let x = i32(globalId.x);
let y = i32(globalId.y);
if (x >= outputSize.x || y >= outputSize.y) {
  return;
}
let inIdx = (y + inputOffset.y) * inputSize.x + x + inputOffset.x;
let outIdx = (y + outputOffset.y) * imageSize.x + x + outputOffset.x;
let col = in_color[inIdx];
let raw = ${e?"textureLoad(in_raw, globalId.xy + vec2u(outputOffset), 0)":"in_raw[outIdx]"};

if (${t}) {
  out_color[outIdx] = vec4f(raw.rgb, 1.0 - col.r);
}
else if (${n}) {
  out_color[outIdx] = vec4f(
    vec3f(PUInverse(col.r * rcpNormScale), PUInverse(col.g * rcpNormScale), PUInverse(col.b * rcpNormScale)) / inputScale,
    // Pick the alpha
    raw.a
  );
}
else {
  out_color[outIdx] = vec4f(col.rgb, raw.a);
}
`})}setImageSize(e,t){this._inputPassAux.setUniform("inputSize",new Int32Array([e,t])),this._inputPassColor.setUniform("inputSize",new Int32Array([e,t])),this._outputPass.setUniform("imageSize",new Int32Array([e,t])),this._outputPass.setSize(e,t),this._copyPass.setSize(e,t),this._copyPass.setUniform("size",new Int32Array([e,t]))}setInputTile(e){const t=new Int32Array([e.width,e.height]);[this._inputPassAux,this._inputPassColor].forEach(n=>{n.setUniform("inputOffset",new Int32Array([e.x,e.y])),n.setUniform("outputSize",t),n.setSize(t[0],t[1])}),this._outputPass.setUniform("inputSize",t)}setOutputTile(e,t){const n=this._outputPass,o=new Int32Array([e.width,e.height]),r=e.x-t.x,s=e.y-t.y;n.setUniform("outputSize",o),n.setUniform("inputOffset",new Int32Array([r,s])),n.setUniform("outputOffset",new Int32Array([e.x,e.y])),n.setExecuteSize(o[0],o[1])}forward(e,t,n,o){const r=e instanceof GPUTexture;this._updatePasses(r,o);const s=this._inputPassAux,a=this._inputPassColor,u=this._device.createCommandEncoder();function l(p){return p instanceof GPUTexture?{texture:p,channels:4}:{buffer:p,channels:4}}return t&&n?s.createPass(u,{color:l(e),albedo:l(t),normal:l(n)}):a.createPass(u,{color:l(e)}),this._device.queue.submit([u.finish()]),t&&n?{color:s.getOutput("color"),albedo:s.getOutput("albedo"),normal:s.getOutput("normal")}:{color:a.getOutput("color")}}inverse(e,t){const o=this._device.createCommandEncoder(),r=this._outputPass;return r.createPass(o,{color:{buffer:e,channels:4},raw:t instanceof GPUBuffer?{buffer:t,channels:4}:{texture:t,channels:4}}),this._device.queue.submit([o.finish()]),r.getOutput("color")}copyInputDataToOutput(e){const t=this._device.createCommandEncoder(),o=this._outputPass.getOutput("color"),r=this._copyPass;e instanceof GPUTexture?(r.setOutputBuffers({color:o}),r.createPass(t,{color:{texture:e,channels:4}})):t.copyBufferToBuffer(e,0,o,0,o.size),this._device.queue.submit([t.finish()])}dispose(){this._outputPass.dispose(),this._inputPassAux.dispose(),this._inputPassColor.dispose(),this._copyPass.dispose(!1)}}const ui=256,ci=384,li=16,pi=128,ne=16;function Te(i,e){return Math.ceil(i/e)*e}function fi(i,e){return Math.floor(i/e)*e}function ot(i,e,t){return Math.min(Math.max(i,e),t)}function di(i){const e=[...i].sort((n,o)=>n-o),t=Math.floor(e.length/2);return e.length%2?e[t]:(e[t-1]+e[t])/2}function Dt(i,e){return i<=e?Math.min(Te(i,ne),e):e}class hi{constructor(e,t=!0){h(this,"enabled");h(this,"maxTileSize");h(this,"minTileSize");h(this,"targetTileTimeMs");h(this,"_tileSize");h(this,"_adjustmentStep");const n=typeof t=="object"?t:{};this.enabled=t!==!1,this.maxTileSize=Math.max(ne,fi(e,ne)),this.minTileSize=ot(Te(n.minTileSize??ui,ne),ne,this.maxTileSize),this.targetTileTimeMs=Math.max(1,n.targetTileTimeMs??li),this._adjustmentStep=Math.max(ne,Te(n.adjustmentStep??pi,ne)),this._tileSize=this.enabled?ot(Te(n.initialTileSize??ci,ne),this.minTileSize,this.maxTileSize):this.maxTileSize}get tileSize(){return this._tileSize}observe(e){if(!this.enabled||e.length===0)return!1;const t=e.filter(r=>Number.isFinite(r)&&r>=0);if(t.length===0)return!1;const n=di(t);let o=this._tileSize;return n>this.targetTileTimeMs*1.25?o-=this._adjustmentStep:n<this.targetTileTimeMs*.65&&(o+=this._adjustmentStep),o=ot(Te(o,ne),this.minTileSize,this.maxTileSize),o===this._tileSize?!1:(this._tileSize=o,!0)}}async function gi(i){try{await i.onSubmittedWorkDone()}catch{}}function v(i,e){return{op:"conv2d",id:i,input:e,weight:`${i}.weight`,bias:`${i}.bias`,activation:"relu",padding:"same"}}function pe(i,e){return{op:"maxPool2d",id:i,input:e,size:2,stride:2,padding:"same"}}function fe(i,e){return{op:"upsample2d",id:i,input:e,scale:2,mode:"nearest"}}function de(i,e,t){return{op:"concat",id:i,inputs:[e,t],axis:"channels"}}const _i={schemaVersion:1,id:"oidn-unet-small-v1",family:"oidn-unet-small",input:"input",output:"dec_conv0",receptiveField:174,nodes:[v("enc_conv0","input"),v("enc_conv1","enc_conv0"),pe("pool1","enc_conv1"),v("enc_conv2","pool1"),pe("pool2","enc_conv2"),v("enc_conv3","pool2"),pe("pool3","enc_conv3"),v("enc_conv4","pool3"),pe("pool4","enc_conv4"),v("enc_conv5a","pool4"),v("enc_conv5b","enc_conv5a"),fe("up4","enc_conv5b"),de("concat4","up4","pool3"),v("dec_conv4a","concat4"),v("dec_conv4b","dec_conv4a"),fe("up3","dec_conv4b"),de("concat3","up3","pool2"),v("dec_conv3a","concat3"),v("dec_conv3b","dec_conv3a"),fe("up2","dec_conv3b"),de("concat2","up2","pool1"),v("dec_conv2a","concat2"),v("dec_conv2b","dec_conv2a"),fe("up1","dec_conv2b"),de("concat1","up1","input"),v("dec_conv1a","concat1"),v("dec_conv1b","dec_conv1a"),v("dec_conv0","dec_conv1b")]},yi={schemaVersion:1,id:"oidn-unet-large-v1",family:"oidn-unet-large",input:"input",output:"dec_conv1c",receptiveField:202,nodes:[v("enc_conv1a","input"),v("enc_conv1b","enc_conv1a"),pe("pool1","enc_conv1b"),v("enc_conv2a","pool1"),v("enc_conv2b","enc_conv2a"),pe("pool2","enc_conv2b"),v("enc_conv3a","pool2"),v("enc_conv3b","enc_conv3a"),pe("pool3","enc_conv3b"),v("enc_conv4a","pool3"),v("enc_conv4b","enc_conv4a"),pe("pool4","enc_conv4b"),v("enc_conv5a","pool4"),v("enc_conv5b","enc_conv5a"),fe("up4","enc_conv5b"),de("concat4","up4","pool3"),v("dec_conv4a","concat4"),v("dec_conv4b","dec_conv4a"),fe("up3","dec_conv4b"),de("concat3","up3","pool2"),v("dec_conv3a","concat3"),v("dec_conv3b","dec_conv3a"),fe("up2","dec_conv3b"),de("concat2","up2","pool1"),v("dec_conv2a","concat2"),v("dec_conv2b","dec_conv2a"),fe("up1","dec_conv2b"),de("concat1","up1","input"),v("dec_conv1a","concat1"),v("dec_conv1b","dec_conv1a"),v("dec_conv1c","dec_conv1b")]},mi=[_i,yi];function vn(i){const e=new Set;for(const t of i.nodes)t.op==="conv2d"&&(e.add(t.weight),e.add(t.bias));return e}function Wt(i){return i.desc.getByteSize()}function xn(i){return[...i].sort().join(", ")}function bn(i,e=mi){const t=e.filter(n=>{const o=vn(n);return[...o].some(r=>!i.has(r))?!1:n.allowAdditionalTensors===!0||[...i.keys()].every(r=>o.has(r))});if(t.length===1)return t[0];throw t.length>1?new Error(`Ambiguous OIDN model topology: ${t.map(n=>n.id).join(", ")}`):new Error(`Unsupported OIDN model topology. TZA tensors: ${xn(i.keys())}`)}function Rt(i,e,t){const n=i.get(e);if(!n)throw new Error(`Model ${t} is missing tensor ${e}`);if(n.data.byteLength!==Wt(n))throw new Error(`Tensor ${e} has ${n.data.byteLength} bytes, expected ${Wt(n)}`);return n}function wi(i,e=bn(i)){if(e.schemaVersion!==1)throw new Error(`Unsupported model descriptor schema ${e.schemaVersion}`);const t=vn(e);if(!e.allowAdditionalTensors){const c=[...i.keys()].filter(f=>!t.has(f));if(c.length>0)throw new Error(`Model ${e.id} has unexpected tensors: ${xn(c)}`)}const n=new Map,o=new Map,r=new Map,s=new Set([e.input]);let a,u;const l=(c,f)=>{const d=n.get(c);if(d===void 0)throw new Error(`Model ${e.id} node ${f} reads unknown or forward value ${c}`);return d};for(const c of e.nodes){if(s.has(c.id))throw new Error(`Model ${e.id} produces duplicate value ${c.id}`);if(c.op==="conv2d"){const f=Rt(i,c.weight,e.id),d=Rt(i,c.bias,e.id),g=f.desc.dims;if(f.desc.layout!=="oihw"||g.length!==4)throw new Error(`Tensor ${c.weight} must use OIHW layout`);if(g[2]!==3||g[3]!==3)throw new Error(`Tensor ${c.weight} must use a 3x3 kernel`);if(d.desc.layout!=="x"||d.desc.dims.length!==1)throw new Error(`Tensor ${c.bias} must be a one-dimensional bias`);if(d.desc.dims[0]!==g[0])throw new Error(`Tensor ${c.bias} has ${d.desc.dims[0]} channels, expected ${g[0]}`);if(f.desc.dataType!==d.desc.dataType)throw new Error(`Weight and bias dtype differ for ${c.id}`);if(u&&u!==f.desc.dataType)throw new Error(`Mixed tensor dtypes are not supported by model ${e.id}`);u=f.desc.dataType,c.input===e.input&&a===void 0&&(a=g[1],n.set(e.input,a));const _=l(c.input,c.id);if(_!==g[1])throw new Error(`Tensor ${c.weight} expects ${g[1]} input channels, but ${c.input} provides ${_}`);n.set(c.id,g[0]),o.set(c.id,{weight:f,bias:d,inputChannels:g[1],outputChannels:g[0],kernelHeight:g[2],kernelWidth:g[3]}),r.set(c.id,{inputChannels:g[1],outputChannels:g[0]})}else if(c.op==="concat"){if(c.inputs.length<2)throw new Error(`Concat ${c.id} requires at least two inputs`);const f=c.inputs.reduce((d,g)=>d+l(g,c.id),0);n.set(c.id,f)}else n.set(c.id,l(c.input,c.id));s.add(c.id)}if(a===void 0||u===void 0)throw new Error(`Model ${e.id} has no convolution reading its input`);const p=n.get(e.output);if(p===void 0)throw new Error(`Model ${e.id} output ${e.output} is not produced`);if(p!==3)throw new Error(`Model ${e.id} must produce 3 channels, got ${p}`);return{spec:e,inputChannels:a,outputChannels:p,tensorDataType:u,channelsByValue:n,convChannels:r,convTensors:o}}const vi="This is not an object",xi="This is not a Float16Array object",Lt="This constructor is not a subclass of Float16Array",$n="The constructor property value is not an object",bi="Species constructor didn't return TypedArray object",$i="Derived constructor created TypedArray object which was too small length",Ee="Attempting to access detached ArrayBuffer",ft="Cannot convert undefined or null to object",dt="Cannot mix BigInt and other types, use explicit conversions",Gt="@@iterator property is not callable",Yt="Reduce of empty array with no initial value",ki="The comparison function must be either a function or undefined",st="Offset is out of bounds";function C(i){return(e,...t)=>K(i,e,t)}function ke(i,e){return C(xe(i,e).get)}const{apply:K,construct:Se,defineProperty:Ft,get:at,getOwnPropertyDescriptor:xe,getPrototypeOf:Oe,has:ht,ownKeys:kn,set:Xt,setPrototypeOf:Bn}=Reflect,Bi=Proxy,{EPSILON:Ii,MAX_SAFE_INTEGER:Ht,isFinite:In,isNaN:be}=Number,{iterator:re,species:Pi,toStringTag:Ct,for:Ci}=Symbol,$e=Object,{create:Je,defineProperty:Ue,freeze:Ti,is:Vt}=$e,gt=$e.prototype,Si=gt.__lookupGetter__?C(gt.__lookupGetter__):(i,e)=>{if(i==null)throw T(ft);let t=$e(i);do{const n=xe(t,e);if(n!==void 0)return ce(n,"get")?n.get:void 0}while((t=Oe(t))!==null)},ce=$e.hasOwn||C(gt.hasOwnProperty),Pn=Array,Cn=Pn.isArray,Qe=Pn.prototype,Ei=C(Qe.join),Ai=C(Qe.push),Oi=C(Qe.toLocaleString),Tt=Qe[re],Ui=C(Tt),{abs:Ni,trunc:Tn}=Math,Ze=ArrayBuffer,Mi=Ze.isView,Sn=Ze.prototype,zi=C(Sn.slice),Di=ke(Sn,"byteLength"),_t=typeof SharedArrayBuffer<"u"?SharedArrayBuffer:null,Wi=_t&&ke(_t.prototype,"byteLength"),St=Oe(Uint8Array),Ri=St.from,D=St.prototype,Li=D[re],Gi=C(D.keys),Yi=C(D.values),Fi=C(D.entries),Xi=C(D.set),Kt=C(D.reverse),Hi=C(D.fill),Vi=C(D.copyWithin),qt=C(D.sort),Pe=C(D.slice),Ki=C(D.subarray),z=ke(D,"buffer"),_e=ke(D,"byteOffset"),k=ke(D,"length"),En=ke(D,Ct),qi=Uint8Array,j=Uint16Array,jt=(...i)=>K(Ri,j,i),Et=Uint32Array,ji=Float32Array,me=Oe([][re]()),et=C(me.next),Ji=C(function*(){}().next),Qi=Oe(me),T=TypeError,ut=RangeError,An=WeakSet,On=An.prototype,Zi=C(On.add),er=C(On.has),tt=WeakMap,At=tt.prototype,Fe=C(At.get),tr=C(At.has),Ot=C(At.set),Un=new tt,nr=Je(null,{next:{value:function(){const e=Fe(Un,this);return et(e)}},[re]:{value:function(){return this}}});function De(i){if(i[re]===Tt&&me.next===et)return i;const e=Je(nr);return Ot(Un,e,Ui(i)),e}const Nn=new tt,Mn=Je(Qi,{next:{value:function(){const e=Fe(Nn,this);return Ji(e)},writable:!0,configurable:!0}});for(const i of kn(me))i!=="next"&&Ue(Mn,i,xe(me,i));function Jt(i){const e=Je(Mn);return Ot(Nn,e,i),e}function Xe(i){return i!==null&&typeof i=="object"||typeof i=="function"}function Qt(i){return i!==null&&typeof i=="object"}function He(i){return En(i)!==void 0}function yt(i){const e=En(i);return e==="BigInt64Array"||e==="BigUint64Array"}function ir(i){try{return Cn(i)?!1:(Di(i),!0)}catch{return!1}}function zn(i){if(_t===null)return!1;try{return Wi(i),!0}catch{return!1}}function rr(i){return ir(i)||zn(i)}function Zt(i){return Cn(i)?i[re]===Tt&&me.next===et:!1}function or(i){return He(i)?i[re]===Li&&me.next===et:!1}function We(i){if(typeof i!="string")return!1;const e=+i;return i!==e+""||!In(e)?!1:e===Tn(e)}const Ve=Ci("__Float16Array__");function sr(i){if(!Qt(i))return!1;const e=Oe(i);if(!Qt(e))return!1;const t=e.constructor;if(t===void 0)return!1;if(!Xe(t))throw T($n);return ht(t,Ve)}const mt=1/Ii;function ar(i){return i+mt-mt}const Dn=6103515625e-14,ur=65504,Wn=.0009765625,en=Wn*Dn,cr=Wn*mt;function lr(i){const e=+i;if(!In(e)||e===0)return e;const t=e>0?1:-1,n=Ni(e);if(n<Dn)return t*ar(n/en)*en;const o=(1+cr)*n,r=o-(o-n);return r>ur||be(r)?t*(1/0):t*r}const Rn=new Ze(4),Ln=new ji(Rn),Gn=new Et(Rn),ee=new j(512),te=new qi(512);for(let i=0;i<256;++i){const e=i-127;e<-24?(ee[i]=0,ee[i|256]=32768,te[i]=24,te[i|256]=24):e<-14?(ee[i]=1024>>-e-14,ee[i|256]=1024>>-e-14|32768,te[i]=-e-1,te[i|256]=-e-1):e<=15?(ee[i]=e+15<<10,ee[i|256]=e+15<<10|32768,te[i]=13,te[i|256]=13):e<128?(ee[i]=31744,ee[i|256]=64512,te[i]=24,te[i|256]=24):(ee[i]=31744,ee[i|256]=64512,te[i]=13,te[i|256]=13)}function ie(i){Ln[0]=lr(i);const e=Gn[0],t=e>>23&511;return ee[t]+((e&8388607)>>te[t])}const Ut=new Et(2048);for(let i=1;i<1024;++i){let e=i<<13,t=0;for(;!(e&8388608);)e<<=1,t-=8388608;e&=-8388609,t+=947912704,Ut[i]=e|t}for(let i=1024;i<2048;++i)Ut[i]=939524096+(i-1024<<13);const Be=new Et(64);for(let i=1;i<31;++i)Be[i]=i<<23;Be[31]=1199570944;Be[32]=2147483648;for(let i=33;i<63;++i)Be[i]=2147483648+(i-32<<23);Be[63]=3347054592;const Yn=new j(64);for(let i=1;i<64;++i)i!==32&&(Yn[i]=1024);function I(i){const e=i>>10;return Gn[0]=Ut[Yn[e]+(i&1023)]+Be[e],Ln[0]}function ue(i){const e=+i;return be(e)||e===0?0:Tn(e)}function ct(i){const e=ue(i);return e<0?0:e<Ht?e:Ht}function Re(i,e){if(!Xe(i))throw T(vi);const t=i.constructor;if(t===void 0)return e;if(!Xe(t))throw T($n);const n=t[Pi];return n??e}function Ae(i){if(zn(i))return!1;try{return zi(i,0,0),!1}catch{}return!0}function tn(i,e){const t=be(i),n=be(e);if(t&&n)return 0;if(t)return 1;if(n||i<e)return-1;if(i>e)return 1;if(i===0&&e===0){const o=Vt(i,0),r=Vt(e,0);if(!o&&r)return-1;if(o&&!r)return 1}return 0}const Nt=2,Ke=new tt;function ve(i){return tr(Ke,i)||!Mi(i)&&sr(i)}function $(i){if(!ve(i))throw T(xi)}function Le(i,e){const t=ve(i),n=He(i);if(!t&&!n)throw T(bi);if(typeof e=="number"){let o;if(t){const r=w(i);o=k(r)}else o=k(i);if(o<e)throw T($i)}if(yt(i))throw T(dt)}function w(i){const e=Fe(Ke,i);if(e!==void 0){const o=z(e);if(Ae(o))throw T(Ee);return e}const t=i.buffer;if(Ae(t))throw T(Ee);const n=Se(P,[t,i.byteOffset,i.length],i.constructor);return Fe(Ke,n)}function nn(i){const e=k(i),t=[];for(let n=0;n<e;++n)t[n]=I(i[n]);return t}const Fn=new An;for(const i of kn(D)){if(i===Ct)continue;const e=xe(D,i);ce(e,"get")&&typeof e.get=="function"&&Zi(Fn,e.get)}const pr=Ti({get(i,e,t){return We(e)&&ce(i,e)?I(at(i,e)):er(Fn,Si(i,e))?at(i,e):at(i,e,t)},set(i,e,t,n){return We(e)&&ce(i,e)?Xt(i,e,ie(t)):Xt(i,e,t,n)},getOwnPropertyDescriptor(i,e){if(We(e)&&ce(i,e)){const t=xe(i,e);return t.value=I(t.value),t}return xe(i,e)},defineProperty(i,e,t){return We(e)&&ce(i,e)&&ce(t,"value")&&(t.value=ie(t.value)),Ft(i,e,t)}});class P{constructor(e,t,n){let o;if(ve(e))o=Se(j,[w(e)],new.target);else if(Xe(e)&&!rr(e)){let s,a;if(He(e)){s=e,a=k(e);const u=z(e);if(Ae(u))throw T(Ee);if(yt(e))throw T(dt);const l=new Ze(a*Nt);o=Se(j,[l],new.target)}else{const u=e[re];if(u!=null&&typeof u!="function")throw T(Gt);u!=null?Zt(e)?(s=e,a=e.length):(s=[...e],a=s.length):(s=e,a=ct(s.length)),o=Se(j,[a],new.target)}for(let u=0;u<a;++u)o[u]=ie(s[u])}else o=Se(j,arguments,new.target);const r=new Bi(o,pr);return Ot(Ke,r,o),r}static from(e,...t){const n=this;if(!ht(n,Ve))throw T(Lt);if(n===P){if(ve(e)&&t.length===0){const p=w(e),c=new j(z(p),_e(p),k(p));return new P(z(Pe(c)))}if(t.length===0)return new P(z(jt(e,ie)));const u=t[0],l=t[1];return new P(z(jt(e,function(p,...c){return ie(K(u,this,[p,...De(c)]))},l)))}let o,r;const s=e[re];if(s!=null&&typeof s!="function")throw T(Gt);if(s!=null)Zt(e)?(o=e,r=e.length):or(e)?(o=e,r=k(e)):(o=[...e],r=o.length);else{if(e==null)throw T(ft);o=$e(e),r=ct(o.length)}const a=new n(r);if(t.length===0)for(let u=0;u<r;++u)a[u]=o[u];else{const u=t[0],l=t[1];for(let p=0;p<r;++p)a[p]=K(u,l,[o[p],p])}return a}static of(...e){const t=this;if(!ht(t,Ve))throw T(Lt);const n=e.length;if(t===P){const r=new P(n),s=w(r);for(let a=0;a<n;++a)s[a]=ie(e[a]);return r}const o=new t(n);for(let r=0;r<n;++r)o[r]=e[r];return o}keys(){$(this);const e=w(this);return Gi(e)}values(){$(this);const e=w(this);return Jt(function*(){for(const t of Yi(e))yield I(t)}())}entries(){$(this);const e=w(this);return Jt(function*(){for(const[t,n]of Fi(e))yield[t,I(n)]}())}at(e){$(this);const t=w(this),n=k(t),o=ue(e),r=o>=0?o:n+o;if(!(r<0||r>=n))return I(t[r])}with(e,t){$(this);const n=w(this),o=k(n),r=ue(e),s=r>=0?r:o+r,a=+t;if(s<0||s>=o)throw ut(st);const u=new j(z(n),_e(n),k(n)),l=new P(z(Pe(u))),p=w(l);return p[s]=ie(a),l}map(e,...t){$(this);const n=w(this),o=k(n),r=t[0],s=Re(n,P);if(s===P){const u=new P(o),l=w(u);for(let p=0;p<o;++p){const c=I(n[p]);l[p]=ie(K(e,r,[c,p,this]))}return u}const a=new s(o);Le(a,o);for(let u=0;u<o;++u){const l=I(n[u]);a[u]=K(e,r,[l,u,this])}return a}filter(e,...t){$(this);const n=w(this),o=k(n),r=t[0],s=[];for(let l=0;l<o;++l){const p=I(n[l]);K(e,r,[p,l,this])&&Ai(s,p)}const a=Re(n,P),u=new a(s);return Le(u),u}reduce(e,...t){$(this);const n=w(this),o=k(n);if(o===0&&t.length===0)throw T(Yt);let r,s;t.length===0?(r=I(n[0]),s=1):(r=t[0],s=0);for(let a=s;a<o;++a)r=e(r,I(n[a]),a,this);return r}reduceRight(e,...t){$(this);const n=w(this),o=k(n);if(o===0&&t.length===0)throw T(Yt);let r,s;t.length===0?(r=I(n[o-1]),s=o-2):(r=t[0],s=o-1);for(let a=s;a>=0;--a)r=e(r,I(n[a]),a,this);return r}forEach(e,...t){$(this);const n=w(this),o=k(n),r=t[0];for(let s=0;s<o;++s)K(e,r,[I(n[s]),s,this])}find(e,...t){$(this);const n=w(this),o=k(n),r=t[0];for(let s=0;s<o;++s){const a=I(n[s]);if(K(e,r,[a,s,this]))return a}}findIndex(e,...t){$(this);const n=w(this),o=k(n),r=t[0];for(let s=0;s<o;++s){const a=I(n[s]);if(K(e,r,[a,s,this]))return s}return-1}findLast(e,...t){$(this);const n=w(this),o=k(n),r=t[0];for(let s=o-1;s>=0;--s){const a=I(n[s]);if(K(e,r,[a,s,this]))return a}}findLastIndex(e,...t){$(this);const n=w(this),o=k(n),r=t[0];for(let s=o-1;s>=0;--s){const a=I(n[s]);if(K(e,r,[a,s,this]))return s}return-1}every(e,...t){$(this);const n=w(this),o=k(n),r=t[0];for(let s=0;s<o;++s)if(!K(e,r,[I(n[s]),s,this]))return!1;return!0}some(e,...t){$(this);const n=w(this),o=k(n),r=t[0];for(let s=0;s<o;++s)if(K(e,r,[I(n[s]),s,this]))return!0;return!1}set(e,...t){$(this);const n=w(this),o=ue(t[0]);if(o<0)throw ut(st);if(e==null)throw T(ft);if(yt(e))throw T(dt);if(ve(e))return Xi(w(this),w(e),o);if(He(e)){const u=z(e);if(Ae(u))throw T(Ee)}const r=k(n),s=$e(e),a=ct(s.length);if(o===1/0||a+o>r)throw ut(st);for(let u=0;u<a;++u)n[u+o]=ie(s[u])}reverse(){$(this);const e=w(this);return Kt(e),this}toReversed(){$(this);const e=w(this),t=new j(z(e),_e(e),k(e)),n=new P(z(Pe(t))),o=w(n);return Kt(o),n}fill(e,...t){$(this);const n=w(this);return Hi(n,ie(e),...De(t)),this}copyWithin(e,t,...n){$(this);const o=w(this);return Vi(o,e,t,...De(n)),this}sort(e){$(this);const t=w(this),n=e!==void 0?e:tn;return qt(t,(o,r)=>n(I(o),I(r))),this}toSorted(e){$(this);const t=w(this);if(e!==void 0&&typeof e!="function")throw new T(ki);const n=e!==void 0?e:tn,o=new j(z(t),_e(t),k(t)),r=new P(z(Pe(o))),s=w(r);return qt(s,(a,u)=>n(I(a),I(u))),r}slice(e,t){$(this);const n=w(this),o=Re(n,P);if(o===P){const g=new j(z(n),_e(n),k(n));return new P(z(Pe(g,e,t)))}const r=k(n),s=ue(e),a=t===void 0?r:ue(t);let u;s===-1/0?u=0:s<0?u=r+s>0?r+s:0:u=r<s?r:s;let l;a===-1/0?l=0:a<0?l=r+a>0?r+a:0:l=r<a?r:a;const p=l-u>0?l-u:0,c=new o(p);if(Le(c,p),p===0)return c;const f=z(n);if(Ae(f))throw T(Ee);let d=0;for(;u<l;)c[d]=I(n[u]),++u,++d;return c}subarray(e,t){$(this);const n=w(this),o=Re(n,P),r=new j(z(n),_e(n),k(n)),s=Ki(r,e,t),a=new o(z(s),_e(s),k(s));return Le(a),a}indexOf(e,...t){$(this);const n=w(this),o=k(n);let r=ue(t[0]);if(r===1/0)return-1;r<0&&(r+=o,r<0&&(r=0));for(let s=r;s<o;++s)if(ce(n,s)&&I(n[s])===e)return s;return-1}lastIndexOf(e,...t){$(this);const n=w(this),o=k(n);let r=t.length>=1?ue(t[0]):o-1;if(r===-1/0)return-1;r>=0?r=r<o-1?r:o-1:r+=o;for(let s=r;s>=0;--s)if(ce(n,s)&&I(n[s])===e)return s;return-1}includes(e,...t){$(this);const n=w(this),o=k(n);let r=ue(t[0]);if(r===1/0)return!1;r<0&&(r+=o,r<0&&(r=0));const s=be(e);for(let a=r;a<o;++a){const u=I(n[a]);if(s&&be(u)||u===e)return!0}return!1}join(e){$(this);const t=w(this),n=nn(t);return Ei(n,e)}toLocaleString(...e){$(this);const t=w(this),n=nn(t);return Oi(n,...De(e))}get[Ct](){if(ve(this))return"Float16Array"}}Ue(P,"BYTES_PER_ELEMENT",{value:Nt});Ue(P,Ve,{});Bn(P,St);const qe=P.prototype;Ue(qe,"BYTES_PER_ELEMENT",{value:Nt});Ue(qe,re,{value:qe.values,writable:!0,configurable:!0});Bn(qe,D);function fr(i){return i.op==="concat"?i.inputs:[i.input]}function dr(i){const e=new Map;for(const t of i.nodes)for(const n of fr(t)){const o=e.get(n)??[];o.push(t),e.set(n,o)}return e}function lt(i,e,t){const n=i.get(e);if(!((n==null?void 0:n.length)!==1||n[0].op!==t))return n[0]}function wt(i,e={}){const t=i.spec,n=dr(t),o=new Map(t.nodes.map(p=>[p.id,p])),r=new Set,s=new Map;let a=0,u=0;if(e.fuseConvPool!==!1)for(const p of t.nodes){if(p.op!=="conv2d"||p.activation!=="relu")continue;const c=lt(n,p.id,"maxPool2d");!c||c.size!==2||c.stride!==2||(r.add(p.id),s.set(c.id,{op:"fusedConvReluMaxPool2d",id:c.id,input:p.input,conv:p,pool:c}),a++)}if(e.fuseUpsampleConcatConv!==!1)for(const p of t.nodes){if(p.op!=="conv2d")continue;const c=o.get(p.input);if((c==null?void 0:c.op)!=="concat"||c.inputs.length!==2||lt(n,c.id,"conv2d")!==p)continue;const f=i.channelsByValue.get(c.inputs[0]);if(f===void 0||f%4!==0)continue;const d=c.inputs.map(_=>{const y=o.get(_);return(y==null?void 0:y.op)==="upsample2d"&&y.scale===2&&y.mode==="nearest"&&lt(n,y.id,"concat")===c?{value:y.input,upsample:y}:{value:_}});if(d.filter(_=>_.upsample).length===1){r.add(c.id);for(const _ of c.inputs){const y=o.get(_);(y==null?void 0:y.op)==="upsample2d"&&r.add(y.id)}s.set(p.id,{op:"fusedUpsampleConcatConv2d",id:p.id,inputs:d,conv:p}),u++}}const l=[];for(const p of t.nodes){const c=s.get(p.id);c?l.push(c):r.has(p.id)||l.push(p)}return{spec:t,nodes:l,fusions:{convPool:a,upsampleConcatConv:u}}}function hr(i){return i.op==="concat"?i.inputs:i.op==="fusedUpsampleConcatConv2d"?i.inputs.map(e=>e.value):[i.input]}function rn(i,e){return i.width===e.width&&i.height===e.height}function gr(i,e,t,n){if(!Number.isInteger(e)||e<=0||!Number.isInteger(t)||t<=0)throw new Error(`Invalid model input size ${e}x${t}`);const o=wt(i,n),r={width:e,height:t,channels:i.inputChannels},s=new Map([[i.spec.input,r]]),a=[],u=(c,f)=>{const d=s.get(c);if(!d)throw new Error(`Planned node ${f} reads missing value ${c}`);return d};for(const c of o.nodes){let f;if(c.op==="conv2d"){const d=u(c.input,c.id);f={width:d.width,height:d.height,channels:i.convChannels.get(c.id).outputChannels}}else if(c.op==="maxPool2d"){const d=u(c.input,c.id);f={width:Math.ceil(d.width/2),height:Math.ceil(d.height/2),channels:d.channels}}else if(c.op==="upsample2d"){const d=u(c.input,c.id);f={width:d.width*2,height:d.height*2,channels:d.channels}}else if(c.op==="concat"){const d=c.inputs.map(g=>u(g,c.id));if(d.some(g=>!rn(g,d[0])))throw new Error(`Concat ${c.id} has mismatched spatial shapes`);f={width:d[0].width,height:d[0].height,channels:d.reduce((g,_)=>g+_.channels,0)}}else if(c.op==="fusedConvReluMaxPool2d"){const d=u(c.input,c.id);f={width:Math.ceil(d.width/2),height:Math.ceil(d.height/2),channels:i.convChannels.get(c.conv.id).outputChannels}}else{const d=c.inputs.map(g=>{const _=u(g.value,c.id);return g.upsample?{..._,width:_.width*2,height:_.height*2}:_});if(d.some(g=>!rn(g,d[0])))throw new Error(`Fused decoder ${c.id} has mismatched spatial shapes`);f={width:d[0].width,height:d[0].height,channels:i.convChannels.get(c.conv.id).outputChannels}}s.set(c.id,f),a.push(f)}const l=new Map;o.nodes.forEach((c,f)=>{for(const d of hr(c))l.set(d,f)}),l.set(i.spec.output,o.nodes.length);const p=o.nodes.map((c,f)=>({node:c,outputShape:a[f],lastUse:l.get(c.id)??f}));return{...o,inputShape:r,valueShapes:s,plannedNodes:p}}class Xn{constructor(){h(this,"_stats",new Map)}track(e,t){let n=this._stats.get(e);return n||(n={created:0,destroyed:0,live:0,peakLive:0,resources:new Set},this._stats.set(e,n)),n.resources.has(t)||(n.resources.add(t),n.created++,n.live++,n.peakLive=Math.max(n.peakLive,n.live)),t}release(e,t,n){if(!t)return!1;const o=this._stats.get(e);if(!(o!=null&&o.resources.delete(t)))return!1;try{n()}finally{o.destroyed++,o.live--}return!0}snapshot(e=0){let t=0,n=0,o=0,r=0;const s={};for(const[a,u]of this._stats){const l={created:u.created,destroyed:u.destroyed,live:u.live,peakLive:u.peakLive};s[a]=l,t+=l.live,n+=l.created,o+=l.destroyed,r+=l.peakLive}return{live:t,created:n,destroyed:o,peakLive:r,pending:e,byKind:s}}}const on=new WeakMap;function _r(i){let e=on.get(i);return e||(e={ready:new Map,pending:new Map},on.set(i,e)),e}const Y=8,Q=8,G=4,ye=Q*G,J=Q,q=8,S=8,Z=S+2;function Mt(i,e){return Math.ceil(i/e)*e}function U(i){return Math.ceil(i/4)}function yr(i,e){return i.width*i.height*U(i.channels)*4*e}let Ce;function Hn(i){const e=i&32768?-1:1,t=i>>>10&31,n=i&1023;return t===0?e*n*2**-24:t===31?n===0?e*(1/0):NaN:e*(1+n/1024)*2**(t-15)}function mr(){if(!Ce){Ce=new Float32Array(65536);for(let i=0;i<Ce.length;i++)Ce[i]=Hn(i)}return Ce}function sn(i){if(i.desc.dataType==="Float32")return new Float32Array(i.data.buffer,i.data.byteOffset,i.data.byteLength/4);const e=new Uint16Array(i.data.buffer,i.data.byteOffset,i.data.byteLength/2),t=new Float32Array(e.length);if(e.length<4096)for(let n=0;n<e.length;n++)t[n]=Hn(e[n]);else{const n=mr();for(let o=0;o<e.length;o++)t[o]=n[e[o]]}return t}function vt(i,e,t,n){const o=Mt(t.byteLength,4),r=i.createBuffer({label:e,size:o,usage:n,mappedAtCreation:!0});return new Uint8Array(r.getMappedRange()).set(new Uint8Array(t.buffer,t.byteOffset,t.byteLength)),r.unmap(),r}function an(i,e,t){const n=new Uint32Array(Mt(t.length,4));return n.set(t),vt(i,e,n,GPUBufferUsage.UNIFORM)}function wr(i,e,t,n){const o=U(t.inputChannels),r=U(t.outputChannels),s=r*t.kernelHeight*t.kernelWidth*o*4*4,a=n==="fp16"&&t.weight.desc.dataType==="Float16",u=a?new Uint16Array(s):n==="fp16"?new P(s):new Float32Array(s),l=a?new Uint16Array(t.weight.data.buffer,t.weight.data.byteOffset,t.weight.data.byteLength/2):sn(t.weight);for(let f=0;f<r;f++)for(let d=0;d<t.kernelHeight;d++)for(let g=0;g<t.kernelWidth;g++)for(let _=0;_<o;_++)for(let y=0;y<4;y++){const x=f*4+y;for(let m=0;m<4;m++){const b=_*4+m,W=(((f*t.kernelHeight+d)*t.kernelWidth+g)*o+_)*16+m*4+y;if(x<t.outputChannels&&b<t.inputChannels){const H=((x*t.inputChannels+b)*t.kernelHeight+d)*t.kernelWidth+g;u[W]=l[H]}}}const p=new Float32Array(r*4);p.set(sn(t.bias));const c=vt(i,`oidn/${e}/weights/${n}`,u,GPUBufferUsage.STORAGE);try{return{weights:c,bias:vt(i,`oidn/${e}/bias`,p,GPUBufferUsage.STORAGE)}}catch(f){throw c.destroy(),f}}function F(i){return i==="fp16"?"vec4<f16>":"vec4<f32>"}function oe(i){return i==="fp16"?`enable f16;
`:""}function se(i,e){return e==="fp16"?`vec4<f16>(${i})`:i}function ge(i,e){return e==="relu"?`max(${i}, vec4<f32>(0.0))`:i}function Ne(i,e,t){return t==="fp32"?`
let inputValue = vec4<f32>(${i});
let weightBase = ${e};
acc = fma(vec4<f32>(weights[weightBase]), vec4<f32>(inputValue.x), acc);
acc = fma(vec4<f32>(weights[weightBase + 1u]), vec4<f32>(inputValue.y), acc);
acc = fma(vec4<f32>(weights[weightBase + 2u]), vec4<f32>(inputValue.z), acc);
acc = fma(vec4<f32>(weights[weightBase + 3u]), vec4<f32>(inputValue.w), acc);
`:`
let inputValue = vec4<f16>(${i});
let weightBase = ${e};
var partial = vec4<f16>(0.0h);
partial = fma(weights[weightBase], vec4<f16>(inputValue.x), partial);
partial = fma(weights[weightBase + 1u], vec4<f16>(inputValue.y), partial);
partial = fma(weights[weightBase + 2u], vec4<f16>(inputValue.z), partial);
partial = fma(weights[weightBase + 3u], vec4<f16>(inputValue.w), partial);
acc += vec4<f32>(partial);
`}function vr(i,e,t){const n=t==="fp16"?"vec4<f16>":"vec4<f32>",o=t==="fp16"?`var partial = vec4<f16>(0.0h);
partial = fma(subgroupBroadcast(weights[weightBase], 0u), ${n}(inputValue.x), partial);
partial = fma(subgroupBroadcast(weights[weightBase + 1u], 0u), ${n}(inputValue.y), partial);
partial = fma(subgroupBroadcast(weights[weightBase + 2u], 0u), ${n}(inputValue.z), partial);
partial = fma(subgroupBroadcast(weights[weightBase + 3u], 0u), ${n}(inputValue.w), partial);
acc += vec4<f32>(partial);`:`acc = fma(subgroupBroadcast(weights[weightBase], 0u), vec4<f32>(inputValue.x), acc);
acc = fma(subgroupBroadcast(weights[weightBase + 1u], 0u), vec4<f32>(inputValue.y), acc);
acc = fma(subgroupBroadcast(weights[weightBase + 2u], 0u), vec4<f32>(inputValue.z), acc);
acc = fma(subgroupBroadcast(weights[weightBase + 3u], 0u), vec4<f32>(inputValue.w), acc);`;return`
let inputValue = ${n}(${i});
let weightBase = ${e};
${o}
`}function Vn(i){return i==="fp16"?`
    var partial: array<vec4<f16>, ${G}>;
    for (var row = 0u; row < ${G}u; row++) {
      partial[row] = vec4<f16>(0.0h);
    }
    for (var tileK = 0u; tileK < ${q}u; tileK++) {
      let weightBase =
        (tileK * ${J}u + localId.x) * 4u;
      for (var row = 0u; row < ${G}u; row++) {
        let tileSpatial =
          localId.y * ${G}u + row;
        let inputValue =
          inputTile[tileSpatial * ${q}u + tileK];
        partial[row] = fma(
          weightTile[weightBase],
          vec4<f16>(inputValue.x),
          partial[row]
        );
        partial[row] = fma(
          weightTile[weightBase + 1u],
          vec4<f16>(inputValue.y),
          partial[row]
        );
        partial[row] = fma(
          weightTile[weightBase + 2u],
          vec4<f16>(inputValue.z),
          partial[row]
        );
        partial[row] = fma(
          weightTile[weightBase + 3u],
          vec4<f16>(inputValue.w),
          partial[row]
        );
      }
    }
    for (var row = 0u; row < ${G}u; row++) {
      acc[row] += vec4<f32>(partial[row]);
    }
`:`
    for (var tileK = 0u; tileK < ${q}u; tileK++) {
      let weightBase =
        (tileK * ${J}u + localId.x) * 4u;
      for (var row = 0u; row < ${G}u; row++) {
        let tileSpatial =
          localId.y * ${G}u + row;
        let inputValue =
          inputTile[tileSpatial * ${q}u + tileK];
        acc[row] = fma(
          weightTile[weightBase],
          vec4<f32>(inputValue.x),
          acc[row]
        );
        acc[row] = fma(
          weightTile[weightBase + 1u],
          vec4<f32>(inputValue.y),
          acc[row]
        );
        acc[row] = fma(
          weightTile[weightBase + 2u],
          vec4<f32>(inputValue.z),
          acc[row]
        );
        acc[row] = fma(
          weightTile[weightBase + 3u],
          vec4<f32>(inputValue.w),
          acc[row]
        );
      }
    }
`}function xr(i,e,t,n,o){const r=F(i),s=F(i),a=F(e),u=se(ge("acc",t),e);return`${oe(i)}
struct Params {
  inputWidth: u32,
  inputHeight: u32,
  outputWidth: u32,
  outputHeight: u32,
  inputBlocks: u32,
  outputBlocks: u32,
}

@group(0) @binding(0) var<storage, read> inputData: array<${r}>;
@group(0) @binding(1) var<storage, read> weights: array<${s}>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outputData: array<${a}>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(${Y}, ${Y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.outputWidth || gid.y >= params.outputHeight || gid.z >= ${o}u) {
    return;
  }
  var acc = bias[gid.z];
  for (var ky = 0u; ky < 3u; ky++) {
    let inputY = i32(gid.y) + i32(ky) - 1;
    if (inputY < 0 || inputY >= i32(params.inputHeight)) { continue; }
    for (var kx = 0u; kx < 3u; kx++) {
      let inputX = i32(gid.x) + i32(kx) - 1;
      if (inputX < 0 || inputX >= i32(params.inputWidth)) { continue; }
      let pixelBase = (u32(inputY) * params.inputWidth + u32(inputX)) * ${n}u;
      for (var inputBlock = 0u; inputBlock < ${n}u; inputBlock++) {
        ${Ne("inputData[pixelBase + inputBlock]",`((((gid.z * 3u + ky) * 3u + kx) * ${n}u + inputBlock) * 4u)`,i)}
      }
    }
  }
  let outputIndex = (gid.y * params.outputWidth + gid.x) * ${o}u + gid.z;
  outputData[outputIndex] = ${u};
}
`}function br(i,e,t,n,o){const r=F(i),s=F(i),a=F(e),u=se(ge("acc",t),e);return`${oe(i)}
enable subgroups;
struct Params {
  inputWidth: u32,
  inputHeight: u32,
  outputWidth: u32,
  outputHeight: u32,
  inputBlocks: u32,
  outputBlocks: u32,
}
@group(0) @binding(0) var<storage, read> inputData: array<${r}>;
@group(0) @binding(1) var<storage, read> weights: array<${s}>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outputData: array<${a}>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(${Y}, ${Y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outputInBounds =
    gid.x < params.outputWidth && gid.y < params.outputHeight;
  var acc = bias[gid.z];
  for (var ky = 0u; ky < 3u; ky++) {
    let inputY = i32(gid.y) + i32(ky) - 1;
    let clampedY = u32(clamp(inputY, 0, i32(params.inputHeight) - 1));
    for (var kx = 0u; kx < 3u; kx++) {
      let inputX = i32(gid.x) + i32(kx) - 1;
      let clampedX = u32(clamp(inputX, 0, i32(params.inputWidth) - 1));
      let inputInBounds =
        outputInBounds && inputX >= 0 && inputY >= 0 &&
        inputX < i32(params.inputWidth) && inputY < i32(params.inputHeight);
      let pixelBase =
        (clampedY * params.inputWidth + clampedX) * ${n}u;
      for (var inputBlock = 0u; inputBlock < ${n}u; inputBlock++) {
        ${vr(`select(${r}(0.0), inputData[pixelBase + inputBlock], inputInBounds)`,`((((gid.z * 3u + ky) * 3u + kx) * ${n}u + inputBlock) * 4u)`,i)}
      }
    }
  }
  if (outputInBounds) {
    let outputIndex =
      (gid.y * params.outputWidth + gid.x) * ${o}u + gid.z;
    outputData[outputIndex] = ${u};
  }
}
`}function $r(i,e,t,n,o){const r=F(i),s=F(e),a=se(ge("acc",t),e),u=Z*Z*n,l=S*S;return`${oe(i)}
struct Params {
  inputWidth: u32,
  inputHeight: u32,
  outputWidth: u32,
  outputHeight: u32,
  inputBlocks: u32,
  outputBlocks: u32,
}
@group(0) @binding(0) var<storage, read> inputData: array<${r}>;
@group(0) @binding(1) var<storage, read> weights: array<${r}>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outputData: array<${s}>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> inputPatch: array<${r}, ${u}>;

@compute @workgroup_size(${S}, ${S}, 1)
fn main(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let localLinear =
    localId.y * ${S}u + localId.x;
  for (
    var loadIndex = localLinear;
    loadIndex < ${u}u;
    loadIndex += ${l}u
  ) {
    let patchPixel = loadIndex / ${n}u;
    let inputBlock = loadIndex % ${n}u;
    let patchX = patchPixel % ${Z}u;
    let patchY = patchPixel / ${Z}u;
    let inputX =
      i32(workgroupId.x * ${S}u + patchX) - 1;
    let inputY =
      i32(workgroupId.y * ${S}u + patchY) - 1;
    var value = ${r}(0.0);
    if (
      inputX >= 0 && inputX < i32(params.inputWidth) &&
      inputY >= 0 && inputY < i32(params.inputHeight)
    ) {
      let inputIndex =
        (u32(inputY) * params.inputWidth + u32(inputX)) *
        ${n}u + inputBlock;
      value = inputData[inputIndex];
    }
    inputPatch[loadIndex] = value;
  }
  workgroupBarrier();

  let outputX =
    workgroupId.x * ${S}u + localId.x;
  let outputY =
    workgroupId.y * ${S}u + localId.y;
  let outputBlock = workgroupId.z;
  if (
    outputX >= params.outputWidth || outputY >= params.outputHeight ||
    outputBlock >= ${o}u
  ) {
    return;
  }

  var acc = bias[outputBlock];
  for (var ky = 0u; ky < 3u; ky++) {
    for (var kx = 0u; kx < 3u; kx++) {
      let patchBase =
        ((localId.y + ky) * ${Z}u + localId.x + kx) *
        ${n}u;
      for (var inputBlock = 0u; inputBlock < ${n}u; inputBlock++) {
        ${Ne("inputPatch[patchBase + inputBlock]",`((((outputBlock * 3u + ky) * 3u + kx) * ${n}u + inputBlock) * 4u)`,i)}
      }
    }
  }
  let outputIndex =
    (outputY * params.outputWidth + outputX) * ${o}u + outputBlock;
  outputData[outputIndex] = ${a};
}
`}function kr(i,e,t,n,o){const r=F(i),s=F(e),a=se(ge("acc",t),e),u=Q*Q,l=ye*q,p=q*J*4;return`${oe(i)}
struct Params {
  inputWidth: u32,
  inputHeight: u32,
  outputWidth: u32,
  outputHeight: u32,
  inputBlocks: u32,
  outputBlocks: u32,
}
@group(0) @binding(0) var<storage, read> inputData: array<${r}>;
@group(0) @binding(1) var<storage, read> weights: array<${r}>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outputData: array<${s}>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> inputTile: array<${r}, ${l}>;
var<workgroup> weightTile: array<${r}, ${p}>;

@compute @workgroup_size(${Q}, ${Q}, 1)
fn main(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let spatialBase =
    workgroupId.x * ${ye}u +
    localId.y * ${G}u;
  let outputBlock =
    workgroupId.y * ${J}u + localId.x;
  let spatialCount = params.outputWidth * params.outputHeight;
  var acc: array<vec4<f32>, ${G}>;
  if (outputBlock < ${o}u) {
    for (var row = 0u; row < ${G}u; row++) {
      acc[row] = bias[outputBlock];
    }
  }

  let localLinear =
    localId.y * ${Q}u + localId.x;
  let totalK = ${n*9}u;
  for (var kBase = 0u; kBase < totalK; kBase += ${q}u) {
    for (
      var loadIndex = localLinear;
      loadIndex < ${l}u;
      loadIndex += ${u}u
    ) {
      let tileSpatial = loadIndex / ${q}u;
      let tileK = loadIndex % ${q}u;
      let inputSpatialIndex =
        workgroupId.x * ${ye}u + tileSpatial;
      let kIndex = kBase + tileK;
      var value = ${r}(0.0);
      if (inputSpatialIndex < spatialCount && kIndex < totalK) {
        let outputY = inputSpatialIndex / params.outputWidth;
        let outputX = inputSpatialIndex % params.outputWidth;
        let inputBlock = kIndex % ${n}u;
        let kernelIndex = kIndex / ${n}u;
        let kernelY = kernelIndex / 3u;
        let kernelX = kernelIndex % 3u;
        let inputY = i32(outputY) + i32(kernelY) - 1;
        let inputX = i32(outputX) + i32(kernelX) - 1;
        if (
          inputY >= 0 && inputY < i32(params.inputHeight) &&
          inputX >= 0 && inputX < i32(params.inputWidth)
        ) {
          let inputIndex =
            (u32(inputY) * params.inputWidth + u32(inputX)) *
            ${n}u + inputBlock;
          value = inputData[inputIndex];
        }
      }
      inputTile[loadIndex] = value;
    }

    for (
      var loadIndex = localLinear;
      loadIndex < ${p}u;
      loadIndex += ${u}u
    ) {
      let tileK = loadIndex / ${J*4}u;
      let outputRemainder = loadIndex % ${J*4}u;
      let tileOutputBlock = outputRemainder / 4u;
      let outputLane = outputRemainder % 4u;
      let loadedOutputBlock =
        workgroupId.y * ${J}u + tileOutputBlock;
      let kIndex = kBase + tileK;
      var value = ${r}(0.0);
      if (loadedOutputBlock < ${o}u && kIndex < totalK) {
        let inputBlock = kIndex % ${n}u;
        let kernelIndex = kIndex / ${n}u;
        let kernelY = kernelIndex / 3u;
        let kernelX = kernelIndex % 3u;
        let weightIndex =
          ((((loadedOutputBlock * 3u + kernelY) * 3u + kernelX) *
            ${n}u + inputBlock) * 4u + outputLane);
        value = weights[weightIndex];
      }
      weightTile[loadIndex] = value;
    }

    workgroupBarrier();
    ${Vn(i)}
    workgroupBarrier();
  }

  if (outputBlock < ${o}u) {
    for (var row = 0u; row < ${G}u; row++) {
      let spatialIndex = spatialBase + row;
      if (spatialIndex < spatialCount) {
        let outputIndex = spatialIndex * ${o}u + outputBlock;
        outputData[outputIndex] = ${a.replaceAll("acc","acc[row]")};
      }
    }
  }
}
`}function Br(i,e,t,n){const o=F(i),r=ge("acc",e);return`${oe(i)}
struct Params {
  inputWidth: u32,
  inputHeight: u32,
  outputWidth: u32,
  outputHeight: u32,
  inputBlocks: u32,
  outputBlocks: u32,
}
@group(0) @binding(0) var<storage, read> inputData: array<${o}>;
@group(0) @binding(1) var<storage, read> weights: array<${o}>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outputData: array<${o}>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(${Y}, ${Y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.outputWidth || gid.y >= params.outputHeight || gid.z >= ${n}u) {
    return;
  }
  var pooled = vec4<f32>(-3.402823466e+38);
  for (var py = 0u; py < 2u; py++) {
    let centerY = gid.y * 2u + py;
    if (centerY >= params.inputHeight) { continue; }
    for (var px = 0u; px < 2u; px++) {
      let centerX = gid.x * 2u + px;
      if (centerX >= params.inputWidth) { continue; }
      var acc = bias[gid.z];
      for (var ky = 0u; ky < 3u; ky++) {
        let inputY = i32(centerY) + i32(ky) - 1;
        if (inputY < 0 || inputY >= i32(params.inputHeight)) { continue; }
        for (var kx = 0u; kx < 3u; kx++) {
          let inputX = i32(centerX) + i32(kx) - 1;
          if (inputX < 0 || inputX >= i32(params.inputWidth)) { continue; }
          let pixelBase = (u32(inputY) * params.inputWidth + u32(inputX)) * ${t}u;
          for (var inputBlock = 0u; inputBlock < ${t}u; inputBlock++) {
            ${Ne("inputData[pixelBase + inputBlock]",`((((gid.z * 3u + ky) * 3u + kx) * ${t}u + inputBlock) * 4u)`,i)}
          }
        }
      }
      pooled = max(pooled, ${r});
    }
  }
  let outputIndex = (gid.y * params.outputWidth + gid.x) * ${n}u + gid.z;
  outputData[outputIndex] = ${se("pooled",i)};
}
`}function Ir(i,e){const t=F(i);return`${oe(i)}
struct Params {
  inputWidth: u32,
  inputHeight: u32,
  outputWidth: u32,
  outputHeight: u32,
  outputBlocks: u32,
}
@group(0) @binding(0) var<storage, read> inputData: array<${t}>;
@group(0) @binding(1) var<storage, read_write> outputData: array<${t}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${Y}, ${Y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (
    gid.x >= params.outputWidth ||
    gid.y >= params.outputHeight ||
    gid.z >= ${e}u
  ) {
    return;
  }
  var pooled = vec4<f32>(-3.402823466e+38);
  for (var py = 0u; py < 2u; py++) {
    let inputY = gid.y * 2u + py;
    if (inputY >= params.inputHeight) { continue; }
    for (var px = 0u; px < 2u; px++) {
      let inputX = gid.x * 2u + px;
      if (inputX >= params.inputWidth) { continue; }
      let inputIndex =
        (inputY * params.inputWidth + inputX) * ${e}u + gid.z;
      pooled = max(pooled, vec4<f32>(inputData[inputIndex]));
    }
  }
  let outputIndex =
    (gid.y * params.outputWidth + gid.x) * ${e}u + gid.z;
  outputData[outputIndex] = ${se("pooled",i)};
}
`}function Pr(i,e,t,n,o){const r=F(i),s=t[0]+t[1],a=se(ge("acc[row]",e),i),u=Q*Q,l=ye*q,p=q*J*4,c=(f,d)=>`
          {
            let sourceBlock = ${d};
            let sourceX = ${f===n?"u32(inputX) / 2u":"u32(inputX)"};
            let sourceY = ${f===n?"u32(inputY) / 2u":"u32(inputY)"};
            let sourceIndex =
              (sourceY * params.source${f}Width + sourceX) *
              ${t[f]}u + sourceBlock;
            value = input${f}[sourceIndex];
          }`;return`${oe(i)}
struct Params {
  outputWidth: u32,
  outputHeight: u32,
  outputBlocks: u32,
  inputBlocks: u32,
  source0Width: u32,
  source0Height: u32,
  source1Width: u32,
  source1Height: u32,
}
@group(0) @binding(0) var<storage, read> input0: array<${r}>;
@group(0) @binding(1) var<storage, read> input1: array<${r}>;
@group(0) @binding(2) var<storage, read> weights: array<${r}>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> outputData: array<${r}>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> inputTile: array<${r}, ${l}>;
var<workgroup> weightTile: array<${r}, ${p}>;

@compute @workgroup_size(${Q}, ${Q}, 1)
fn main(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let spatialBase =
    workgroupId.x * ${ye}u +
    localId.y * ${G}u;
  let outputBlock =
    workgroupId.y * ${J}u + localId.x;
  let spatialCount = params.outputWidth * params.outputHeight;
  var acc: array<vec4<f32>, ${G}>;
  if (outputBlock < ${o}u) {
    for (var row = 0u; row < ${G}u; row++) {
      acc[row] = bias[outputBlock];
    }
  }

  let localLinear =
    localId.y * ${Q}u + localId.x;
  let totalK = ${s*9}u;
  for (var kBase = 0u; kBase < totalK; kBase += ${q}u) {
    for (
      var loadIndex = localLinear;
      loadIndex < ${l}u;
      loadIndex += ${u}u
    ) {
      let tileSpatial = loadIndex / ${q}u;
      let tileK = loadIndex % ${q}u;
      let outputSpatialIndex =
        workgroupId.x * ${ye}u + tileSpatial;
      let kIndex = kBase + tileK;
      var value = ${r}(0.0);
      if (outputSpatialIndex < spatialCount && kIndex < totalK) {
        let outputY = outputSpatialIndex / params.outputWidth;
        let outputX = outputSpatialIndex % params.outputWidth;
        let inputBlock = kIndex % ${s}u;
        let kernelIndex = kIndex / ${s}u;
        let kernelY = kernelIndex / 3u;
        let kernelX = kernelIndex % 3u;
        let inputY = i32(outputY) + i32(kernelY) - 1;
        let inputX = i32(outputX) + i32(kernelX) - 1;
        if (
          inputY >= 0 && inputY < i32(params.outputHeight) &&
          inputX >= 0 && inputX < i32(params.outputWidth)
        ) {
          if (inputBlock < ${t[0]}u) {
            ${c(0,"inputBlock")}
          } else {
            ${c(1,`inputBlock - ${t[0]}u`)}
          }
        }
      }
      inputTile[loadIndex] = value;
    }

    for (
      var loadIndex = localLinear;
      loadIndex < ${p}u;
      loadIndex += ${u}u
    ) {
      let tileK = loadIndex / ${J*4}u;
      let outputRemainder = loadIndex % ${J*4}u;
      let tileOutputBlock = outputRemainder / 4u;
      let outputLane = outputRemainder % 4u;
      let loadedOutputBlock =
        workgroupId.y * ${J}u + tileOutputBlock;
      let kIndex = kBase + tileK;
      var value = ${r}(0.0);
      if (loadedOutputBlock < ${o}u && kIndex < totalK) {
        let inputBlock = kIndex % ${s}u;
        let kernelIndex = kIndex / ${s}u;
        let kernelY = kernelIndex / 3u;
        let kernelX = kernelIndex % 3u;
        let weightIndex =
          ((((loadedOutputBlock * 3u + kernelY) * 3u + kernelX) *
            ${s}u + inputBlock) * 4u + outputLane);
        value = weights[weightIndex];
      }
      weightTile[loadIndex] = value;
    }

    workgroupBarrier();
    ${Vn(i)}
    workgroupBarrier();
  }

  if (outputBlock < ${o}u) {
    for (var row = 0u; row < ${G}u; row++) {
      let spatialIndex = spatialBase + row;
      if (spatialIndex < spatialCount) {
        let outputIndex = spatialIndex * ${o}u + outputBlock;
        outputData[outputIndex] = ${a};
      }
    }
  }
}
`}function Cr(i,e,t,n,o){const r=F(i),s=t[0]+t[1],a=(l,p)=>{const c=l===n;return`
      {
        let sourceX = ${c?"u32(inputX) / 2u":"u32(inputX)"};
        let sourceY = ${c?"u32(inputY) / 2u":"u32(inputY)"};
        let sourcePixelBase = (sourceY * params.source${l}Width + sourceX) * ${t[l]}u;
        for (var sourceBlock = 0u; sourceBlock < ${t[l]}u; sourceBlock++) {
          let inputBlock = ${p}u + sourceBlock;
          ${Ne(`input${l}[sourcePixelBase + sourceBlock]`,`((((gid.z * 3u + ky) * 3u + kx) * ${s}u + inputBlock) * 4u)`,i)}
        }
      }
`},u=se(ge("acc",e),i);return`${oe(i)}
struct Params {
  outputWidth: u32,
  outputHeight: u32,
  outputBlocks: u32,
  inputBlocks: u32,
  source0Width: u32,
  source0Height: u32,
  source1Width: u32,
  source1Height: u32,
}
@group(0) @binding(0) var<storage, read> input0: array<${r}>;
@group(0) @binding(1) var<storage, read> input1: array<${r}>;
@group(0) @binding(2) var<storage, read> weights: array<${r}>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> outputData: array<${r}>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(${Y}, ${Y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.outputWidth || gid.y >= params.outputHeight || gid.z >= ${o}u) {
    return;
  }
  var acc = bias[gid.z];
  for (var ky = 0u; ky < 3u; ky++) {
    let inputY = i32(gid.y) + i32(ky) - 1;
    if (inputY < 0 || inputY >= i32(params.outputHeight)) { continue; }
    for (var kx = 0u; kx < 3u; kx++) {
      let inputX = i32(gid.x) + i32(kx) - 1;
      if (inputX < 0 || inputX >= i32(params.outputWidth)) { continue; }
      ${a(0,0)}
      ${a(1,t[0])}
    }
  }
  let outputIndex = (gid.y * params.outputWidth + gid.x) * ${o}u + gid.z;
  outputData[outputIndex] = ${u};
}
`}function Tr(i,e,t,n,o){const r=F(i),s=t[0]+t[1],a=Z*Z*s,u=S*S,l=se(ge("acc",e),i),p=(c,f)=>`
      let sourceBlock = ${f};
      let sourceIndex =
        (${c===n?"u32(inputY) / 2u":"u32(inputY)"} * params.source${c}Width + ${c===n?"u32(inputX) / 2u":"u32(inputX)"}) *
        ${t[c]}u + sourceBlock;
      value = input${c}[sourceIndex];`;return`${oe(i)}
struct Params {
  outputWidth: u32,
  outputHeight: u32,
  outputBlocks: u32,
  inputBlocks: u32,
  source0Width: u32,
  source0Height: u32,
  source1Width: u32,
  source1Height: u32,
}
@group(0) @binding(0) var<storage, read> input0: array<${r}>;
@group(0) @binding(1) var<storage, read> input1: array<${r}>;
@group(0) @binding(2) var<storage, read> weights: array<${r}>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> outputData: array<${r}>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> inputPatch: array<${r}, ${a}>;

@compute @workgroup_size(${S}, ${S}, 1)
fn main(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let localLinear =
    localId.y * ${S}u + localId.x;
  for (
    var loadIndex = localLinear;
    loadIndex < ${a}u;
    loadIndex += ${u}u
  ) {
    let patchPixel = loadIndex / ${s}u;
    let inputBlock = loadIndex % ${s}u;
    let patchX = patchPixel % ${Z}u;
    let patchY = patchPixel / ${Z}u;
    let inputX =
      i32(workgroupId.x * ${S}u + patchX) - 1;
    let inputY =
      i32(workgroupId.y * ${S}u + patchY) - 1;
    var value = ${r}(0.0);
    if (
      inputX >= 0 && inputX < i32(params.outputWidth) &&
      inputY >= 0 && inputY < i32(params.outputHeight)
    ) {
      if (inputBlock < ${t[0]}u) {
        ${p(0,"inputBlock")}
      } else {
        ${p(1,`inputBlock - ${t[0]}u`)}
      }
    }
    inputPatch[loadIndex] = value;
  }
  workgroupBarrier();

  let outputX =
    workgroupId.x * ${S}u + localId.x;
  let outputY =
    workgroupId.y * ${S}u + localId.y;
  let outputBlock = workgroupId.z;
  if (
    outputX >= params.outputWidth || outputY >= params.outputHeight ||
    outputBlock >= ${o}u
  ) {
    return;
  }

  var acc = bias[outputBlock];
  for (var ky = 0u; ky < 3u; ky++) {
    for (var kx = 0u; kx < 3u; kx++) {
      let patchBase =
        ((localId.y + ky) * ${Z}u + localId.x + kx) *
        ${s}u;
      for (var inputBlock = 0u; inputBlock < ${s}u; inputBlock++) {
        ${Ne("inputPatch[patchBase + inputBlock]",`((((outputBlock * 3u + ky) * 3u + kx) * ${s}u + inputBlock) * 4u)`,i)}
      }
    }
  }
  let outputIndex =
    (outputY * params.outputWidth + outputX) * ${o}u + outputBlock;
  outputData[outputIndex] = ${l};
}
`}function un(i,e){const t=F(i),n=Array.from({length:e},(a,u)=>`@group(0) @binding(${u}) var<storage, read> input${u}: array<vec4<f32>>;`).join(`
`),o=Array.from({length:e},(a,u)=>{const l=u*3;return`if (channel < ${l+3}u) { return input${u}[pixel][channel - ${l}u]; }`}).join(`
  `),r=e,s=e+1;return`${oe(i)}
struct Params {
  width: u32,
  height: u32,
  outputBlocks: u32,
  inputChannels: u32,
}
${n}
@group(0) @binding(${r}) var<storage, read_write> outputData: array<${t}>;
@group(0) @binding(${s}) var<uniform> params: Params;

fn readChannel(pixel: u32, channel: u32) -> f32 {
  ${o}
  return 0.0;
}

@compute @workgroup_size(${Y}, ${Y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.outputBlocks) {
    return;
  }
  let pixel = gid.y * params.width + gid.x;
  let firstChannel = gid.z * 4u;
  let value = vec4<f32>(
    readChannel(pixel, firstChannel),
    readChannel(pixel, firstChannel + 1u),
    readChannel(pixel, firstChannel + 2u),
    readChannel(pixel, firstChannel + 3u)
  );
  outputData[pixel * params.outputBlocks + gid.z] = ${se("value",i)};
}
`}function Sr(i){return i.op==="concat"?i.inputs:i.op==="fusedUpsampleConcatConv2d"?i.inputs.map(e=>e.value):[i.input]}function Er(i,e="auto"){const t=i.features.has("shader-f16");if(e==="fp16"&&!t)throw new Error("OIDN FP16 was requested but the GPUDevice does not have shader-f16 enabled");return e==="auto"?t?"fp16":"fp32":e}class Ar{constructor(e,t,n={}){h(this,"_device");h(this,"precision");h(this,"kernelSetting");h(this,"maxSpatialInputBlocks");h(this,"subgroupsAvailable");h(this,"_model");h(this,"_packedConvs",new Map);h(this,"_pipelineCache");h(this,"_pipelinePromises");h(this,"_executionCache",new Map);h(this,"_retiredExecutions",new Set);h(this,"_clock",0);h(this,"_shapeCacheSize");h(this,"_profileNextExecution",!1);h(this,"_lastExecutionProfile");h(this,"_profileOperations",0);h(this,"_resources",new Xn);h(this,"_disposed",!1);this._device=e;const o=_r(e);if(this._pipelineCache=o.ready,this._pipelinePromises=o.pending,this.precision=Er(e,n.precision??"auto"),this.kernelSetting=n.kernel??"auto",this.subgroupsAvailable=e.features.has("subgroups"),this.maxSpatialInputBlocks=this.precision==="fp16"&&e.limits.maxComputeInvocationsPerWorkgroup>=S*S&&e.limits.maxComputeWorkgroupSizeX>=S&&e.limits.maxComputeWorkgroupSizeY>=S?Math.floor(e.limits.maxComputeWorkgroupStorageSize/(Z*Z*4*2)):0,this._shapeCacheSize=Math.max(1,n.shapeCacheSize??2),t.inputChannels%3!==0||t.inputChannels<3||t.inputChannels>9)throw new Error(`Native OIDN expects 3, 6, or 9 input channels, got ${t.inputChannels}`);this._model={spec:t.spec,inputChannels:t.inputChannels,outputChannels:t.outputChannels,channelsByValue:new Map(t.channelsByValue),convChannels:new Map(t.convChannels)};for(const r of wt(this._model,{fuseConvPool:!1}).nodes)if(r.op!=="conv2d"&&r.op!=="maxPool2d"&&r.op!=="fusedConvReluMaxPool2d"&&r.op!=="fusedUpsampleConcatConv2d")throw new Error(`Native OIDN descriptor ${t.spec.id} leaves unsupported ${r.op} node ${r.id} after graph optimization`);try{for(const[r,s]of t.convTensors){const a=wr(e,r,s,this.precision);this._resources.track("gpu-buffer",a.weights),this._resources.track("gpu-buffer",a.bias),this._packedConvs.set(r,a)}}catch(r){for(const s of this._packedConvs.values())this._releaseBuffer(s.weights),this._releaseBuffer(s.bias);throw this._packedConvs.clear(),r}}_pipeline(e,t){let n=this._pipelineCache.get(e);return n||(n=this._device.createComputePipeline({label:`oidn/${e}`,layout:"auto",compute:{module:this._device.createShaderModule({label:`oidn/${e}`,code:t}),entryPoint:"main"}}),this._pipelineCache.set(e,n)),n}_pipelineAsync(e,t){const n=this._pipelineCache.get(e);if(n)return Promise.resolve(n);const o=this._pipelinePromises.get(e);if(o)return o;const r=this._device.createComputePipelineAsync({label:`oidn/${e}`,layout:"auto",compute:{module:this._device.createShaderModule({label:`oidn/${e}`,code:t}),entryPoint:"main"}}).then(s=>(this._pipelineCache.set(e,s),this._pipelinePromises.delete(e),s),s=>{throw this._pipelinePromises.delete(e),s});return this._pipelinePromises.set(e,r),r}_nodePipelineSpec(e,t){if(e.op==="conv2d"){const n=t?"fp32":this.precision,o=U(this._model.convChannels.get(e.id).inputChannels),r=U(this._model.convChannels.get(e.id).outputChannels),s=this._selectConvKernel(o,t);return{key:`conv-${s}/${this.precision}/${n}/${e.activation}/in${o}/out${r}`,kernel:s,code:s==="implicit-gemm"?kr(this.precision,n,e.activation,o,r):s==="spatial"?$r(this.precision,n,e.activation,o,r):s==="subgroup"?br(this.precision,n,e.activation,o,r):xr(this.precision,n,e.activation,o,r)}}if(e.op==="maxPool2d"){const n=U(this._model.channelsByValue.get(e.id));return{key:`max-pool/${this.precision}/out${n}`,kernel:"direct",code:Ir(this.precision,n)}}if(e.op==="fusedConvReluMaxPool2d"){const n=U(this._model.convChannels.get(e.conv.id).inputChannels),o=U(this._model.convChannels.get(e.conv.id).outputChannels);return{key:`conv-pool/${this.precision}/${e.conv.activation}/in${n}/out${o}`,kernel:"direct",code:Br(this.precision,e.conv.activation,n,o)}}if(e.op==="fusedUpsampleConcatConv2d"){if(e.inputs.length!==2)throw new Error(`Native fused decoder ${e.id} requires two inputs`);const n=e.inputs.map(p=>U(this._model.channelsByValue.get(p.value))),o=e.inputs.findIndex(p=>p.upsample);if(o!==0&&o!==1)throw new Error(`Native fused decoder ${e.id} has no upsample input`);const r=n[0]+n[1],s=this._selectConvKernel(r,!1),a=s==="subgroup"?"direct":s,u=U(this._model.convChannels.get(e.conv.id).outputChannels);return{key:`decoder-${a}/${this.precision}/${e.conv.activation}/${n.join("+")}/out${u}/up${o}`,kernel:a,code:a==="implicit-gemm"?Pr(this.precision,e.conv.activation,n,o,u):a==="spatial"?Tr(this.precision,e.conv.activation,n,o,u):Cr(this.precision,e.conv.activation,n,o,u)}}throw new Error(`Native OIDN does not implement unfused ${e.op} node ${e.id}`)}_selectConvKernel(e,t){const n=this.precision==="fp16"&&e<=this.maxSpatialInputBlocks;return this.kernelSetting==="direct"?"direct":this.kernelSetting==="spatial"?n?"spatial":"direct":this.kernelSetting==="implicit-gemm"?t?"direct":"implicit-gemm":this.kernelSetting==="subgroup"?this.subgroupsAvailable?"subgroup":"direct":this.precision==="fp32"&&!t?"implicit-gemm":"direct"}_nodePipeline(e,t){const{key:n,code:o}=this._nodePipelineSpec(e,t);return this._pipeline(n,o)}async prepare(){if(this._disposed)throw new Error("Native OIDN executor is disposed");const e=wt(this._model,{fuseConvPool:!1}),t=this._model.inputChannels/3,n=[{key:`pack/${this.precision}/${t}`,code:un(this.precision,t)},...e.nodes.map(o=>this._nodePipelineSpec(o,o.id===e.spec.output))];await Promise.all(n.map(({key:o,code:r})=>this._pipelineAsync(o,r)))}_createExecution(e,t){const n=gr(this._model,e,t,{fuseConvPool:!1}),o=new Map,r=[],s=new Map;n.nodes.forEach((l,p)=>{for(const c of Sr(l))s.set(c,p)}),s.set(n.spec.output,n.nodes.length);const a=[],u=l=>(a.push(l),this._resources.track("gpu-buffer",l));try{const l=(m,b,O,W)=>{for(const N of r)N.activeValue&&(s.get(N.activeValue)??-1)<W&&(N.activeValue=void 0);const H=yr(b,O);let R=r.filter(N=>!N.activeValue&&N.capacity>=H).sort((N,X)=>N.capacity-X.capacity)[0];R||(R={buffer:u(this._device.createBuffer({label:`oidn/activation/${e}x${t}/${r.length}`,size:Mt(H,4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST})),capacity:H},r.push(R)),R.activeValue=m,o.set(m,R.buffer)};l(n.spec.input,n.inputShape,this.precision==="fp16"?2:4,-1),n.plannedNodes.forEach(({node:m,outputShape:b},O)=>{const W=m.id===n.spec.output;l(m.id,b,W||this.precision==="fp32"?4:2,O)});const p=this._model.inputChannels/3,c=`pack/${this.precision}/${p}`,f=this._pipeline(c,un(this.precision,p)),d=[],g=[],_=[],y=[];n.plannedNodes.forEach(({node:m,outputShape:b},O)=>{const W=m.id===n.spec.output,H=this._nodePipelineSpec(m,W),R=this._pipeline(H.key,H.code);d.push(R),g.push(H.kernel??"direct");const N=o.get(m.id);let X,B,E;if(m.op==="conv2d"){const M=n.valueShapes.get(m.input);E=m.id,B=[M.width,M.height,b.width,b.height,U(M.channels),U(b.channels)];const A=this._packedConvs.get(E);X=[{binding:0,resource:{buffer:o.get(m.input)}},{binding:1,resource:{buffer:A.weights}},{binding:2,resource:{buffer:A.bias}},{binding:3,resource:{buffer:N}}]}else if(m.op==="maxPool2d"){const M=n.valueShapes.get(m.input);B=[M.width,M.height,b.width,b.height,U(b.channels)],X=[{binding:0,resource:{buffer:o.get(m.input)}},{binding:1,resource:{buffer:N}}]}else if(m.op==="fusedConvReluMaxPool2d"){const M=n.valueShapes.get(m.input);E=m.conv.id,B=[M.width,M.height,b.width,b.height,U(M.channels),U(b.channels)];const A=this._packedConvs.get(E);X=[{binding:0,resource:{buffer:o.get(m.input)}},{binding:1,resource:{buffer:A.weights}},{binding:2,resource:{buffer:A.bias}},{binding:3,resource:{buffer:N}}]}else if(m.op==="fusedUpsampleConcatConv2d"){E=m.conv.id;const M=n.valueShapes.get(m.inputs[0].value),A=n.valueShapes.get(m.inputs[1].value);B=[b.width,b.height,U(b.channels),U(this._model.convChannels.get(E).inputChannels),M.width,M.height,A.width,A.height];const L=this._packedConvs.get(E);X=[{binding:0,resource:{buffer:o.get(m.inputs[0].value)}},{binding:1,resource:{buffer:o.get(m.inputs[1].value)}},{binding:2,resource:{buffer:L.weights}},{binding:3,resource:{buffer:L.bias}},{binding:4,resource:{buffer:N}}]}else throw new Error(`Unexpected native node ${m.op}`);const ae=u(an(this._device,`oidn/${m.id}/params/${e}x${t}`,B));y.push(ae),X.push({binding:X.length,resource:{buffer:ae}}),_.push(this._device.createBindGroup({label:`oidn/${m.id}/bindings`,layout:R.getBindGroupLayout(0),entries:X}))});const x=u(an(this._device,`oidn/input/params/${e}x${t}`,[e,t,U(this._model.inputChannels),this._model.inputChannels]));return y.push(x),{plan:n,valueBuffers:o,slots:r,nodeBindings:_,nodePipelines:d,nodeKernels:g,inputPipeline:f,inputUniform:x,ownedBuffers:y,lastUsed:++this._clock}}catch(l){for(const p of a)this._releaseBuffer(p);throw l}}_execution(e,t){if(this._disposed)throw new Error("Native OIDN executor is disposed");const n=`${e}x${t}`;let o=this._executionCache.get(n);if(!o&&(o=this._createExecution(e,t),this._executionCache.set(n,o),this._executionCache.size>this._shapeCacheSize)){const r=[...this._executionCache.entries()].filter(([s])=>s!==n).sort((s,a)=>s[1].lastUsed-a[1].lastUsed)[0];r&&(this._executionCache.delete(r[0]),this._retiredExecutions.add(r[1]),this._device.queue.onSubmittedWorkDone().catch(()=>{}).then(()=>{this._retiredExecutions.delete(r[1]),this._destroyExecution(r[1])}))}return o.lastUsed=++this._clock,o}profileNextExecution(){return this._device.features.has("timestamp-query")?(this._profileNextExecution=!0,!0):!1}getLastExecutionProfile(){return this._lastExecutionProfile}execute(e,t,n){const o=this._model.inputChannels/3;if(e.length!==o)throw new Error(`Native OIDN expected ${o} input buffers, got ${e.length}`);const r=this._execution(t,n),s=["input-pack",...r.plan.nodes.map(g=>g.id)],a=this._profileNextExecution&&this._device.features.has("timestamp-query");this._profileNextExecution=!1;const u=s.length*2,l=a?this._resources.track("gpu-query-set",this._device.createQuerySet({type:"timestamp",count:u})):void 0,p=u*8;let c,f;try{c=a?this._resources.track("gpu-buffer",this._device.createBuffer({label:`oidn/profile/resolve/${t}x${n}`,size:p,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC})):void 0,f=a?this._resources.track("gpu-buffer",this._device.createBuffer({label:`oidn/profile/readback/${t}x${n}`,size:p,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ})):void 0}catch(g){throw this._releaseBuffer(f),this._releaseBuffer(c),this._releaseQuerySet(l),g}const d=(g,_)=>({label:g,...l?{timestampWrites:{querySet:l,beginningOfPassWriteIndex:_*2,endOfPassWriteIndex:_*2+1}}:{}});try{const g=this._device.createCommandEncoder({label:`oidn/native/${t}x${n}`}),_=e.map((x,m)=>({binding:m,resource:{buffer:x}}));_.push({binding:o,resource:{buffer:r.valueBuffers.get(r.plan.spec.input)}}),_.push({binding:o+1,resource:{buffer:r.inputUniform}});const y=this._device.createBindGroup({label:"oidn/input/bindings",layout:r.inputPipeline.getBindGroupLayout(0),entries:_});{const x=g.beginComputePass(d("oidn/input-pack",0));x.setPipeline(r.inputPipeline),x.setBindGroup(0,y),x.dispatchWorkgroups(Math.ceil(t/Y),Math.ceil(n/Y),U(this._model.inputChannels)),x.end()}r.plan.plannedNodes.forEach(({node:x,outputShape:m},b)=>{const O=g.beginComputePass(d(`oidn/${r.plan.nodes[b].id}`,b+1));O.setPipeline(r.nodePipelines[b]),O.setBindGroup(0,r.nodeBindings[b]),r.nodeKernels[b]==="implicit-gemm"?O.dispatchWorkgroups(Math.ceil(m.width*m.height/ye),Math.ceil(U(m.channels)/J),1):O.dispatchWorkgroups(Math.ceil(m.width/Y),Math.ceil(m.height/Y),U(m.channels)),O.end()}),l&&(g.resolveQuerySet(l,0,u,c,0),g.copyBufferToBuffer(c,0,f,0,p)),this._device.queue.submit([g.finish()])}catch(g){throw this._releaseBuffer(f),this._releaseBuffer(c),this._releaseQuerySet(l),g}return l&&(this._profileOperations++,this._lastExecutionProfile=(async()=>{try{await f.mapAsync(GPUMapMode.READ);const g=new BigUint64Array(f.getMappedRange()),_=s.map((y,x)=>({id:y,durationMs:Number(g[x*2+1]-g[x*2])/1e6}));return{totalMs:_.reduce((y,x)=>y+x.durationMs,0),layers:_}}finally{f.mapState==="mapped"&&f.unmap(),this._releaseQuerySet(l),this._releaseBuffer(c),this._releaseBuffer(f),this._profileOperations--}})()),r.valueBuffers.get(r.plan.spec.output)}async executeCPU(e,t,n){const o=t*n*this._model.inputChannels;if(e.length!==o)throw new Error(`Native OIDN CPU input has ${e.length} values, expected ${o}`);const r=this._execution(t,n),s=this._model.inputChannels/3,a=t*n;r.cpuInputBuffers||(r.cpuInputBuffers=Array.from({length:s},(f,d)=>{const g=this._device.createBuffer({label:`oidn/cpu-input/${t}x${n}/${d}`,size:a*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});return this._resources.track("gpu-buffer",g),r.ownedBuffers.push(g),g}),r.cpuReadbackBuffer=this._device.createBuffer({label:`oidn/cpu-readback/${t}x${n}`,size:a*16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),this._resources.track("gpu-buffer",r.cpuReadbackBuffer),r.ownedBuffers.push(r.cpuReadbackBuffer));for(let f=0;f<s;f++){const d=new Float32Array(a*4);for(let g=0;g<a;g++){const _=g*this._model.inputChannels+f*3,y=g*4;d[y]=e[_],d[y+1]=e[_+1],d[y+2]=e[_+2]}this._device.queue.writeBuffer(r.cpuInputBuffers[f],0,d)}const u=this.execute(r.cpuInputBuffers,t,n),l=this._device.createCommandEncoder({label:`oidn/cpu-readback/${t}x${n}`});l.copyBufferToBuffer(u,0,r.cpuReadbackBuffer,0,a*16),this._device.queue.submit([l.finish()]),await r.cpuReadbackBuffer.mapAsync(GPUMapMode.READ);const p=new Float32Array(r.cpuReadbackBuffer.getMappedRange()),c=new Float32Array(a*3);for(let f=0;f<a;f++)c[f*3]=p[f*4],c[f*3+1]=p[f*4+1],c[f*3+2]=p[f*4+2];return r.cpuReadbackBuffer.unmap(),c}_releaseBuffer(e){this._resources.release("gpu-buffer",e,()=>e.destroy())}_releaseQuerySet(e){this._resources.release("gpu-query-set",e,()=>e.destroy())}_destroyExecution(e){e.slots.forEach(t=>this._releaseBuffer(t.buffer)),e.ownedBuffers.forEach(t=>this._releaseBuffer(t))}getResourceInfo(){return this._resources.snapshot(this._retiredExecutions.size+this._profileOperations)}dispose(){if(!this._disposed){this._disposed=!0;for(const e of this._packedConvs.values())this._releaseBuffer(e.weights),this._releaseBuffer(e.bias);this._packedConvs.clear();for(const e of this._executionCache.values())this._destroyExecution(e);this._executionCache.clear();for(const e of this._retiredExecutions)this._destroyExecution(e);this._retiredExecutions.clear()}}}const cn=new WeakMap;function Or(i,e){let t=cn.get(i);t||(t=new Map,cn.set(i,t));let n=t.get(e);if(!n){const o=i.createShaderModule({label:`oidn/webnn/input-pack/${e}`,code:Mr(e)}),r=i.createShaderModule({label:"oidn/webnn/output-unpack",code:zr()});n={input:i.createComputePipeline({label:`oidn/webnn/input-pack/${e}`,layout:"auto",compute:{module:o,entryPoint:"main"}}),output:i.createComputePipeline({label:"oidn/webnn/output-unpack",layout:"auto",compute:{module:r,entryPoint:"main"}})},t.set(e,n)}return n}const he=8;function Kn(i,e){return Math.ceil(i/e)*e}function Ur(i,e,t,n){const o=i.createBuffer({label:e,size:Kn(t.byteLength,4),usage:n,mappedAtCreation:!0});return new Uint8Array(o.getMappedRange()).set(new Uint8Array(t.buffer,t.byteOffset,t.byteLength)),o.unmap(),o}function ln(i,e,t){const n=new Uint32Array(Kn(t.length,4));return n.set(t),Ur(i,e,n,GPUBufferUsage.UNIFORM)}function pt(i,e,t,n){var o,r,s,a;return!!((a=(s=(r=(o=i==null?void 0:i[e])==null?void 0:o[t])==null?void 0:r.dataTypes)==null?void 0:s.includes)!=null&&a.call(s,n))}function Nr(i,e){if(e==="fp16"&&i.desc.dataType==="Float16")return new Uint8Array(i.data.buffer,i.data.byteOffset,i.data.byteLength);if(e==="fp32"&&i.desc.dataType==="Float32")return new Uint8Array(i.data.buffer,i.data.byteOffset,i.data.byteLength);const t=i.desc.dataType==="Float32"?new Float32Array(i.data.buffer,i.data.byteOffset,i.data.byteLength/4):new P(i.data.buffer,i.data.byteOffset,i.data.byteLength/2),n=e==="fp16"?new P(t):new Float32Array(t);return new Uint8Array(n.buffer,n.byteOffset,n.byteLength)}function Mr(i){const e=Array.from({length:i},(n,o)=>`@group(0) @binding(${o}) var<storage, read> input${o}: array<vec4<f32>>;`).join(`
`),t=Array.from({length:i},(n,o)=>{const r=o*3;return`if (channel < ${r+3}u) {
      return input${o}[pixel][channel - ${r}u];
    }`}).join(`
  `);return`enable f16;
struct Params { width: u32, height: u32, channels: u32, padding: u32 }
${e}
@group(0) @binding(${i}) var<storage, read_write> outputData: array<f16>;
@group(0) @binding(${i+1}) var<uniform> params: Params;

fn readChannel(pixel: u32, channel: u32) -> f32 {
  ${t}
  return 0.0;
}

@compute @workgroup_size(${he}, ${he}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.channels) {
    return;
  }
  let pixel = gid.y * params.width + gid.x;
  let outputIndex = (gid.z * params.height + gid.y) * params.width + gid.x;
  outputData[outputIndex] = f16(readChannel(pixel, gid.z));
}
`}function zr(){return`enable f16;
struct Params { width: u32, height: u32, padding0: u32, padding1: u32 }
@group(0) @binding(0) var<storage, read> inputData: array<f16>;
@group(0) @binding(1) var<storage, read_write> outputData: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${he}, ${he}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let pixel = gid.y * params.width + gid.x;
  let plane = params.width * params.height;
  outputData[pixel] = vec4<f32>(
    f32(inputData[pixel]),
    f32(inputData[plane + pixel]),
    f32(inputData[plane * 2u + pixel]),
    0.0
  );
}
`}function Dr(i,e){if(e==="fp32")throw new Error("OIDN WebNN GPU interop currently requires FP16 exportable tensors");if(!i.features.has("shader-f16"))throw new Error("OIDN WebNN requires shader-f16 on the shared GPUDevice");return"fp16"}function Wr(i,e,t){return t==="relu"?i.relu(e):e}class Rr{constructor(e,t,n={}){h(this,"_device");h(this,"_model");h(this,"precision");h(this,"support");h(this,"_context");h(this,"_builderConstructor");h(this,"_shapeCache",new Map);h(this,"_shapePromises",new Map);h(this,"_retiredExecutions",new Set);h(this,"_pendingCreationCount",0);h(this,"_shapeCacheSize");h(this,"_clock",0);h(this,"_inputPipeline");h(this,"_outputPipeline");h(this,"_resources",new Xn);h(this,"_disposed",!1);this._device=e,this._model=t,this.precision=Dr(e,n.precision??"auto"),this._shapeCacheSize=Math.max(1,n.shapeCacheSize??2),this.support={available:!1,fp16Conv:!1,gpuInterop:!1};const o=Or(e,t.inputChannels/3);this._inputPipeline=o.input,this._outputPipeline=o.output}async prepare(){var n,o,r;if(this._disposed)throw new Error("OIDN WebNN executor is disposed");const e=(n=globalThis.navigator)==null?void 0:n.ml,t=globalThis.MLGraphBuilder;if(!(e!=null&&e.createContext)||typeof t!="function")throw this.support.reason="WebNN is not exposed by this browser",new Error(this.support.reason);this._builderConstructor=t;try{try{this._context=await e.createContext({deviceType:"gpu",powerPreference:"high-performance"})}catch{this._context=await e.createContext({deviceType:"gpu"})}if(this._resources.track("ml-context",this._context),this._disposed)throw new Error("OIDN WebNN executor is disposed");if(typeof this._context.createExportableTensor!="function"||typeof this._context.exportToGPU!="function")throw this.support.reason="WebNN WebGPU tensor interop is unavailable",new Error(this.support.reason);const s=((r=(o=this._context).opSupportLimits)==null?void 0:r.call(o))??{};if(this.support.fp16Conv=pt(s,"conv2d","input","float16")&&pt(s,"conv2d","filter","float16")&&pt(s,"conv2d","output","float16"),!this.support.fp16Conv)throw this.support.reason="WebNN does not support FP16 conv2d",new Error(this.support.reason);let a,u;try{a=this._resources.track("ml-tensor",await this._context.createExportableTensor({dataType:"float16",shape:[4]},this._device)),u=this._resources.track("gpu-buffer",await this._context.exportToGPU(a)),this.support.gpuInterop=!0}catch(l){throw this.support.reason=`WebNN FP16 WebGPU interop failed: ${String(l)}`,new Error(this.support.reason)}finally{this._releaseBuffer(u),this._releaseTensor(a)}this.support.available=!0}catch(s){throw this._releaseContext(),s}}_constant(e,t){return e.constant({dataType:"float16",shape:[...t.desc.dims]},Nr(t,this.precision))}async _createExecution(e,t){if(this._disposed)throw new Error("OIDN WebNN executor is disposed");const n=new this._builderConstructor(this._context),o=new Map,r=new Map;o.set(this._model.spec.input,n.input("input",{dataType:"float16",shape:[1,this._model.inputChannels,t,e]})),r.set(this._model.spec.input,[this._model.inputChannels,t,e]);for(const f of this._model.spec.nodes){let d,g;if(f.op==="conv2d"){const _=r.get(f.input),y=this._model.convTensors.get(f.id),x=n.conv2d(o.get(f.input),this._constant(n,y.weight),{bias:this._constant(n,y.bias),padding:[1,1,1,1],inputLayout:"nchw",filterLayout:"oihw"});d=Wr(n,x,f.activation),g=[y.outputChannels,_[1],_[2]]}else if(f.op==="maxPool2d"){const _=r.get(f.input);d=n.maxPool2d(o.get(f.input),{windowDimensions:[2,2],strides:[2,2],padding:[0,_[1]%2,0,_[2]%2],layout:"nchw"}),g=[_[0],Math.ceil(_[1]/2),Math.ceil(_[2]/2)]}else if(f.op==="upsample2d"){const _=r.get(f.input);d=n.resample2d(o.get(f.input),{mode:"nearest-neighbor",axes:[2,3],scales:[2,2]}),g=[_[0],_[1]*2,_[2]*2]}else{const _=f.inputs.map(y=>r.get(y));if(_.some(y=>y[1]!==_[0][1]||y[2]!==_[0][2]))throw new Error(`WebNN concat ${f.id} has mismatched spatial shapes`);d=n.concat(f.inputs.map(y=>o.get(y)),1),g=[_.reduce((y,x)=>y+x[0],0),_[0][1],_[0][2]]}o.set(f.id,d),r.set(f.id,g)}let s,a,u,l,p,c;try{return s=this._resources.track("ml-graph",await n.build({output:o.get(this._model.spec.output)})),a=this._resources.track("ml-tensor",await this._context.createExportableTensor({dataType:"float16",shape:[1,this._model.inputChannels,t,e],writable:!0},this._device)),u=this._resources.track("ml-tensor",await this._context.createExportableTensor({dataType:"float16",shape:[1,this._model.outputChannels,t,e],readable:!0},this._device)),l=this._resources.track("gpu-buffer",this._device.createBuffer({label:`oidn/webnn/output/${e}x${t}`,size:e*t*4*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST})),p=this._resources.track("gpu-buffer",ln(this._device,`oidn/webnn/input/${e}x${t}`,[e,t,this._model.inputChannels])),c=this._resources.track("gpu-buffer",ln(this._device,`oidn/webnn/output/${e}x${t}`,[e,t])),{graph:s,inputTensor:a,outputTensor:u,outputBuffer:l,inputUniform:p,outputUniform:c,width:e,height:t,lastUsed:++this._clock}}catch(f){throw this._releaseBuffer(c),this._releaseBuffer(p),this._releaseBuffer(l),this._releaseTensor(u),this._releaseTensor(a),this._releaseGraph(s),f}}async _execution(e,t){const n=`${e}x${t}`;let o=this._shapeCache.get(n);if(!o){let r=this._shapePromises.get(n);r||(r=(async()=>{this._pendingCreationCount++;try{return await this._createExecution(e,t)}finally{this._pendingCreationCount--}})(),this._shapePromises.set(n,r));try{if(o=await r,this._disposed)throw this._destroyExecution(o),new Error("OIDN WebNN executor is disposed");this._shapeCache.set(n,o)}finally{this._shapePromises.get(n)===r&&this._shapePromises.delete(n)}if(this._shapeCache.size>this._shapeCacheSize){const s=[...this._shapeCache.entries()].filter(([a])=>a!==n).sort((a,u)=>a[1].lastUsed-u[1].lastUsed)[0];s&&(this._shapeCache.delete(s[0]),this._retireExecution(s[1]))}}return o.lastUsed=++this._clock,o}async prewarm(e){for(const t of e)await this._execution(t.width,t.height)}async execute(e,t,n){const o=this._model.inputChannels/3;if(e.length!==o)throw new Error(`OIDN WebNN expected ${o} input buffers, got ${e.length}`);const r=await this._execution(t,n),s=this._resources.track("gpu-buffer",await this._context.exportToGPU(r.inputTensor));try{const u=e.map((f,d)=>({binding:d,resource:{buffer:f}}));u.push({binding:o,resource:{buffer:s}}),u.push({binding:o+1,resource:{buffer:r.inputUniform}});const l=this._device.createBindGroup({label:"oidn/webnn/input-bindings",layout:this._inputPipeline.getBindGroupLayout(0),entries:u}),p=this._device.createCommandEncoder({label:"oidn/webnn/input-pack"}),c=p.beginComputePass();c.setPipeline(this._inputPipeline),c.setBindGroup(0,l),c.dispatchWorkgroups(Math.ceil(t/he),Math.ceil(n/he),this._model.inputChannels),c.end(),this._device.queue.submit([p.finish()])}finally{this._releaseBuffer(s)}this._context.dispatch(r.graph,{input:r.inputTensor},{output:r.outputTensor});const a=this._resources.track("gpu-buffer",await this._context.exportToGPU(r.outputTensor));try{const u=this._device.createBindGroup({label:"oidn/webnn/output-bindings",layout:this._outputPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:a}},{binding:1,resource:{buffer:r.outputBuffer}},{binding:2,resource:{buffer:r.outputUniform}}]}),l=this._device.createCommandEncoder({label:"oidn/webnn/output-unpack"}),p=l.beginComputePass();p.setPipeline(this._outputPipeline),p.setBindGroup(0,u),p.dispatchWorkgroups(Math.ceil(t/he),Math.ceil(n/he)),p.end(),this._device.queue.submit([l.finish()])}finally{this._releaseBuffer(a)}return r.outputBuffer}async executeCPU(e,t,n){const o=await this._execution(t,n),r=t*n,s=new P(r*this._model.inputChannels);for(let l=0;l<r;l++)for(let p=0;p<this._model.inputChannels;p++)s[p*r+l]=e[l*this._model.inputChannels+p];this._context.writeTensor(o.inputTensor,s),this._context.dispatch(o.graph,{input:o.inputTensor},{output:o.outputTensor});const a=new P(await this._context.readTensor(o.outputTensor)),u=new Float32Array(r*this._model.outputChannels);for(let l=0;l<r;l++)for(let p=0;p<this._model.outputChannels;p++)u[l*this._model.outputChannels+p]=a[p*r+l];return u}_destroyExecution(e){this._releaseGraph(e.graph),this._releaseTensor(e.inputTensor),this._releaseTensor(e.outputTensor),this._releaseBuffer(e.outputBuffer),this._releaseBuffer(e.inputUniform),this._releaseBuffer(e.outputUniform)}_retireExecution(e){this._retiredExecutions.add(e),this._device.queue.onSubmittedWorkDone().catch(()=>{}).then(()=>{this._retiredExecutions.delete(e),this._destroyExecution(e)})}_releaseBuffer(e){this._resources.release("gpu-buffer",e,()=>e.destroy())}_releaseTensor(e){this._resources.release("ml-tensor",e,()=>e.destroy())}_releaseGraph(e){this._resources.release("ml-graph",e,()=>{var t;return(t=e.destroy)==null?void 0:t.call(e)})}_releaseContext(){this._resources.release("ml-context",this._context,()=>{var e,t;return(t=(e=this._context).destroy)==null?void 0:t.call(e)})}getResourceInfo(){return this._resources.snapshot(this._pendingCreationCount+this._retiredExecutions.size)}dispose(){if(!this._disposed){this._disposed=!0;for(const e of this._shapeCache.values())this._destroyExecution(e);this._shapeCache.clear();for(const e of this._retiredExecutions)this._destroyExecution(e);this._retiredExecutions.clear(),this._shapePromises.clear(),this._releaseContext()}}}function pn(i,e){return Math.ceil(i/e)*e}function Ge(i){return i.data instanceof GPUBuffer||i.data instanceof GPUTexture}class Lr{constructor(e,t,n={}){h(this,"_device");h(this,"_tileWidth",0);h(this,"_tileHeight",0);h(this,"_tileOverlapX",0);h(this,"_tileOverlapY",0);h(this,"_aux");h(this,"_hdr");h(this,"_dataProcessGPU");h(this,"_nativeExecutor");h(this,"_webNNExecutor");h(this,"_modelSpec");h(this,"_inputChannels");h(this,"_engine");h(this,"_dynamicTileController");h(this,"_lastExecution");this._aux=n.aux||!1,this._hdr=n.hdr||!1,this._engine=n.engine??"auto";const o=n.modelSpec??bn(e),r=wi(e,o);this._modelSpec=r.spec,this._inputChannels=r.inputChannels;const s=this._aux?9:3;if(r.inputChannels!==s)throw new Error(`OIDN model expects ${r.inputChannels} input channels, but aux=${this._aux} provides ${s}`);this._dynamicTileController=new hi(n.maxTileSize??512,n.dynamicTile),this._device=t.device,this._engine==="webnn"?this._webNNExecutor=new Rr(this._device,r,{precision:n.precision}):this._nativeExecutor=new Ar(this._device,r,{precision:n.precision,kernel:n.kernel})}getDevice(){return this._device}async prepare(){if(this._webNNExecutor){await this._webNNExecutor.prepare();const e=pn(this._modelSpec.receptiveField/2,ne),t=[this._dynamicTileController.tileSize,this._dynamicTileController.minTileSize];await this._webNNExecutor.prewarm([...new Set(t)].map(n=>({width:n+2*e,height:n+2*e})));return}await this._nativeExecutor.prepare()}getRuntimeInfo(){var e;return{configuredEngine:this._engine,gpuEngine:this._webNNExecutor?"webnn":"wgsl",precision:(this._webNNExecutor??this._nativeExecutor).precision,kernel:this._nativeExecutor?{configured:this._nativeExecutor.kernelSetting,maxSpatialInputBlocks:this._nativeExecutor.maxSpatialInputBlocks,subgroupsAvailable:this._nativeExecutor.subgroupsAvailable}:void 0,webnn:(e=this._webNNExecutor)==null?void 0:e.support,resources:(this._webNNExecutor??this._nativeExecutor).getResourceInfo(),model:this._modelSpec.id,modelFamily:this._modelSpec.family,inputChannels:this._inputChannels,dynamicTile:{enabled:this._dynamicTileController.enabled,currentTileSize:this._dynamicTileController.tileSize,minTileSize:this._dynamicTileController.minTileSize,maxTileSize:this._dynamicTileController.maxTileSize,targetTileTimeMs:this._dynamicTileController.targetTileTimeMs},lastExecution:this._lastExecution}}profileNextExecution(){var e;return((e=this._nativeExecutor)==null?void 0:e.profileNextExecution())??!1}getLastExecutionProfile(){var e;return(e=this._nativeExecutor)==null?void 0:e.getLastExecutionProfile()}_updateModel(e,t){const n=this._dynamicTileController.tileSize;let o=Dt(e,n),r=Dt(t,n);const s=pn(this._modelSpec.receptiveField/2,ne);let a=s,u=s;e<=n&&(a=0),t<=n&&(u=0);const l=Math.max(o,r),p=Math.max(a,u);o=l,r=l,a=p,u=p,(o!==this._tileWidth||r!==this._tileHeight||a!==this._tileOverlapX||u!==this._tileOverlapY)&&(this._tileWidth=o,this._tileHeight=r,this._tileOverlapX=a,this._tileOverlapY=u)}_getTileSizeWithOverlap(){return{width:this._tileWidth+2*this._tileOverlapX,height:this._tileHeight+2*this._tileOverlapY}}_processImageData(e,t,n,o){const r=e.data,s=r.length/4,a=this._aux?9:3,u=new Float32Array(s*a);if(t&&!n||n&&!t)throw new Error("Normal map and albedo map are both required");if(t&&n&&(t.width!==n.width||t.height!==n.height||e.width!==t.width||e.height!==t.height))throw new Error("Image size mismatch");const l=t==null?void 0:t.data,p=n==null?void 0:n.data;for(let c=0;c<r.length;c+=4){const f=c/4*a;for(let d=0;d<3;d++)o?u[f+d]=r[c+d]:u[f+d]=r[c+d]/255,l&&(u[f+d+3]=l[c+d]/255),p&&(u[f+d+6]=p[c+d]/255)}return u}_readTile(e,t,n,o){const r=new Float32Array(n.width*n.height*t);for(let s=0;s<n.height;s++)for(let a=0;a<n.width;a++){const u=((s+n.y)*o+(a+n.x))*t,l=(s*n.width+a)*t;for(let p=0;p<t;p++)r[l+p]=e[u+p]}return r}_writeTile(e,t,n,o,r,s){const{data:a,width:u}=e,l=n.x-t.x,p=n.y-t.y;for(let c=0;c<n.height;c++)for(let f=0;f<n.width;f++){const d=((c+p)*r+f+l)*3,g=((c+n.y)*u+(f+n.x))*4;for(let _=0;_<3;_++)s?a[g+_]=o[d+_]:a[g+_]=Math.min(Math.max(o[d+_]*255,0),255);e.data[g+3]=s?1:255}}async _executeTile(e,t,n,o,r,s,a,u,l){const p=this._aux?9:3,c=this._tileOverlapX,f=this._tileOverlapY;let d=this._getTileSizeWithOverlap(),g={width:this._tileWidth,height:this._tileHeight},_=o>0?o*g.width-c:0,y=Math.min(_+d.width,s);_=Math.max(y-d.width,0);let x=r>0?r*g.height-f:0,m=Math.min(x+d.height,a);x=Math.max(m-d.height,0);const b=d.width,O=d.height,W=new rt(_,x,b,O);let H,R,N=1;const X=this._device;let B=this._dataProcessGPU;if(e instanceof Float32Array){let L=this._readTile(e,p,W,s);u&&(N=ri({data:L,channels:p}),L=oi({data:L,channels:p,inputScale:N})),R=await(this._webNNExecutor??this._nativeExecutor).executeCPU(L,b,O)}else{B||(B=this._dataProcessGPU=new ai(X,u)),B.setImageSize(s,a),B.setInputTile(W),o===0&&r===0&&B.copyInputDataToOutput(e.color);const{color:L,albedo:le,normal:V}=B.forward(e.color,this._aux?e.albedo:void 0,this._aux?e.normal:void 0,l);H=await(this._webNNExecutor??this._nativeExecutor).execute(this._aux?[L,le,V]:[L],b,O)}let E;const ae=Math.min(g.width,s),M=Math.min(g.height,a),A=new rt(o*ae,r*M,ae,M);if(A.width=Math.min(A.width,s-A.x),A.height=Math.min(A.height,a-A.y),e instanceof Float32Array){u&&(R=si({data:R,channels:3,inputScale:N})),this._writeTile(n,W,A,R,d.width,u);for(let L=0;L<M;L++)for(let le=0;le<ae;le++){const V=(L*ae+le)*4,Ie=((L+A.y)*s+(le+A.x))*4;for(let we=0;we<4;we++)t.data[V+we]=n.data[Ie+we]}}else B.setOutputTile(A,W),E=B.inverse(H,e.color);return E}tileExecute({color:e,albedo:t,normal:n,done:o,progress:r,denoiseAlpha:s}){if(this._aux&&(!t||!n))throw new Error("Normal map and albedo map are both required");if(!this._aux&&(t||n))throw new Error("Normal map and albedo map are not required");const a=e.width,u=e.height,l=this._dynamicTileController.tileSize,p=a>l||u>l;this._updateModel(a,u);const c=this._hdr||!1;let f;Ge(e)||(f=this._processImageData(e,t,n,c));const d=this._tileWidth,g=this._tileHeight,_=Math.ceil(u/g),y=Math.ceil(a/d);function x(B,E){return c?{data:new Float32Array(B*E*4),width:B,height:E}:new ImageData(B,E)}const m=Ge(e)?void 0:x(a,u),b=Ge(e)?void 0:x(Math.min(d,a),Math.min(g,u));let O=!1;const W=()=>typeof performance>"u"?Date.now():performance.now(),H=W(),R=[],N=B=>{typeof requestAnimationFrame>"u"?setTimeout(B,0):requestAnimationFrame(B)},X=async(B,E)=>{if(O)return;const ae=W(),M=await this._executeTile(Ge(e)?{color:e.data,albedo:t==null?void 0:t.data,normal:n==null?void 0:n.data}:f,b,m,B,E,a,u,c,s);if(O)return;const A=m||{data:M,width:a,height:u};r==null||r(A,b,new rt(B*d,E*g,d,g),B+E*y,y*_);const L=B+1<y||E+1<_,le=()=>{if(R.push(W()-ae),!O)if(L)N(()=>{O||(B+1<y?X(B+1,E):E+1<_&&X(0,E+1))});else{const V=[...R].sort((nt,it)=>nt-it),Ie=Math.floor(V.length/2),we=V.length%2?V[Ie]:(V[Ie-1]+V[Ie])/2;this._lastExecution={width:a,height:u,tileWidth:d,tileHeight:g,tileCount:y*_,durationMs:W()-H,tileTimeMs:{min:V[0],median:we,mean:V.reduce((nt,it)=>nt+it,0)/V.length,max:V[V.length-1]}},p&&this._dynamicTileController.observe(R),o(A)}};gi(this._device.queue).then(le)};return X(0,0),()=>{O=!0}}dispose(){var e,t,n;(e=this._dataProcessGPU)==null||e.dispose(),(t=this._nativeExecutor)==null||t.dispose(),(n=this._webNNExecutor)==null||n.dispose()}}async function Gr(){var a;if(!navigator.gpu)throw new Error("WebGPU is not available");const i={powerPreference:"high-performance"},e=await navigator.gpu.requestAdapter(i);if(!e)throw new Error("No WebGPU adapter is available");const t={},n=[];e.features.has("timestamp-query")&&n.push("timestamp-query"),e.features.has("bgra8unorm-storage")&&n.push("bgra8unorm-storage"),e.features.has("shader-f16")&&n.push("shader-f16"),t.requiredFeatures=n;const o=e.limits;t.requiredLimits={maxComputeWorkgroupStorageSize:o.maxComputeWorkgroupStorageSize,maxComputeWorkgroupsPerDimension:o.maxComputeWorkgroupsPerDimension,maxStorageBufferBindingSize:o.maxStorageBufferBindingSize,maxBufferSize:o.maxBufferSize,maxComputeWorkgroupSizeX:o.maxComputeWorkgroupSizeX,maxComputeInvocationsPerWorkgroup:o.maxComputeInvocationsPerWorkgroup};const r=await e.requestDevice(t),s=e.info??await((a=e.requestAdapterInfo)==null?void 0:a.call(e));return Yr(r,s)}async function Yr(i,e){return{device:i,adapterInfo:e}}async function Fr(i,e,t){const n=await Gr(),o=ei(i),r=new Lr(o,n,t);return await r.prepare(),r}async function Xr(i,e,t){return fetch(i).then(n=>n.arrayBuffer()).then(n=>Fr(n,e,t))}class je extends Error{}const Hr="/astra-viewer/oidn/rt_ldr.tza";let Ye=null;function Vr(){if(!Ye){if(!("gpu"in navigator))return Promise.reject(new je("This browser has no WebGPU (navigator.gpu is undefined). Chrome or Edge 113+ required."));Ye=Xr(Hr).catch(i=>{throw Ye=null,new je(`Could not initialise the denoiser: ${i}`)})}return Ye}async function qr(i,e){const t=await Vr(),{width:n,height:o}=i,r=document.createElement("canvas");r.width=n,r.height=o;const s=r.getContext("2d",{willReadFrequently:!0});if(!s)throw new je("Could not get a 2D context for the denoise target");s.drawImage(i,0,0,n,o);const a=s.getImageData(0,0,n,o);return new Promise((u,l)=>{try{t.tileExecute({color:a,done(){e==null||e(1),u(r)},progress(p,c,f,d,g){c&&s.putImageData(c,f.x,f.y),g>0&&(e==null||e((d+1)/g))}})}catch(p){l(new je(`Denoising failed: ${p}`))}})}export{je as DenoiseUnavailable,qr as denoiseCanvas};
