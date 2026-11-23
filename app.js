"use strict";
const $ = id => document.getElementById(id);
const video = $("webcam"), canvas = $("preview"), ctx = canvas.getContext("2d", {willReadFrequently:true});
let session = null, stream = null, interval = null, busy = false;
const setStatus = msg => { $("status").textContent = msg; };
function errorDetails(e) {
  if (e instanceof Error && e.message) return e.message;
  if (typeof e === "string") return e;
  try { return JSON.stringify(e) || String(e); }
  catch { return String(e); }
}
async function loadModel() {
  const modelUrl = "./modelo/perros-gatos-cnn-ad3.onnx";
  try {
    if (typeof ort === "undefined") throw new Error("No se cargó la biblioteca ONNX Runtime Web (CDN bloqueado o sin conexión).");
    setStatus("Comprobando archivo ONNX…");
    const res = await fetch(modelUrl, {cache: "no-store"});
    if (!res.ok) throw new Error(`El archivo ONNX no está disponible: HTTP ${res.status} ${res.statusText}. URL: ${new URL(modelUrl, location.href).href}`);
    const type = res.headers.get("content-type") || "";
    if (/text\/html/i.test(type)) throw new Error(`El servidor devolvió HTML en vez de ONNX (${type}). Revisa la ruta y las redirecciones del hosting.`);
    const bytes = new Uint8Array(await res.arrayBuffer());
    if (bytes.byteLength < 1000) throw new Error(`El archivo ONNX es sospechosamente pequeño (${bytes.byteLength} bytes). Comprueba la subida.`);
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
    ort.env.wasm.numThreads = 1;
    setStatus("Inicializando ONNX Runtime Web…");
    session = await ort.InferenceSession.create(bytes, {executionProviders:["wasm"]});
    setStatus("Modelo preparado. Puedes activar la cámara.");
    $("start").disabled = false;
    console.info("Entrada:",session.inputNames, session.inputMetadata);
  } catch(e) {
    const details = errorDetails(e);
    setStatus("No se pudo cargar el modelo: " + details);
    console.error("Error original cargando ONNX:", e, "Detalles:", details);
  }
}

function inputTensor() {
  // Captura a 100 x 100, equivalente al cv2.resize del notebook.
  ctx.drawImage(video, 0, 0, 100, 100);
  let img = ctx.getImageData(0,0,100,100), pixels = img.data;
  const hwc = new Float32Array(100*100);
  for(let i=0;i<hwc.length;i++) {
    const p=i*4, r=pixels[p], g=pixels[p+1], b=pixels[p+2];
    // cv2.COLOR_RGB2GRAY usa aproximadamente estos pesos.
    const gray = Math.round(0.299*r + 0.587*g + 0.114*b);
    hwc[i] = gray/255;
    pixels[p]=pixels[p+1]=pixels[p+2]=gray;
  }
  ctx.putImageData(img,0,0);
  const meta = session.inputMetadata[session.inputNames[0]];
  const dims = meta?.dimensions || meta?.dims || [];
  const nchw = dims.length===4 && (dims[1]===1 || dims[1]==="1") && dims[3]!==1;
  const shape=nchw ? [1,1,100,100] : [1,100,100,1];
  // Un canal: el orden plano de los 10.000 valores es el mismo en ambos formatos.
  return new ort.Tensor("float32",hwc,shape);
}
async function predict() {
  if(!session || !stream || busy || video.readyState<2) return;
  busy=true;
  try {
    const feeds = {[session.inputNames[0]]:inputTensor()};
    const results = await session.run(feeds);
    const pDog = Number(results[session.outputNames[0]].data[0]);
    if(!Number.isFinite(pDog)) throw new Error("La salida no es un número válido");
    const dog=Math.min(1,Math.max(0,pDog)),cat=1-dog;
    $("dogValue").textContent=(dog*100).toFixed(1)+"%";
    $("catValue").textContent=(cat*100).toFixed(1)+"%";
    $("dogBar").style.width=(dog*100)+"%";
    $("catBar").style.width=(cat*100)+"%";
    $("prediction").textContent=dog>=.5 ? "Perro" : "Gato";
  }catch(e){setStatus("Error en inferencia: "+e.message);console.error(e);stopCamera();}
  finally{busy=false;}
}
async function startCamera(){
  if(!session) return;
  try {
    stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"environment"},audio:false});
    video.srcObject=stream; await video.play();
    $("videoOverlay").hidden=true;
    $("start").disabled=true; $("stop").disabled=false; $("capture").disabled=false;
    setStatus("Cámara activa. Predicción cada 500 ms.");
    interval=setInterval(predict,500); await predict();
  }catch(e){setStatus("No se pudo abrir la cámara: "+e.message);stopCamera();}
}
function stopCamera(){
  if(interval){clearInterval(interval);interval=null;}
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
  video.srcObject=null;
  $("videoOverlay").hidden=false;
  $("start").disabled=!session;$("stop").disabled=true;$("capture").disabled=true;
}
$("start").disabled=true;
$("start").addEventListener("click",startCamera);
$("stop").addEventListener("click",()=>{stopCamera();setStatus("Cámara detenida.");});
$("capture").addEventListener("click",predict);
window.addEventListener("pagehide",stopCamera);
loadModel();
