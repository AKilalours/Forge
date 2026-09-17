"""FORGE: one interface, two detectors. Text and Image.

RUNS ON CPU. The text side scores documents with the trained FORGE arms loaded from
local checkpoints. The image side computes every value from the uploaded bytes at
request time. Neither path calls out to a network service.

TEXT TAB. Loads an arm's checkpoint on CPU in float32 (training ran in bf16, which
cannot matmul on CPU), windows the document, mean-pools the window scores, and reports
the probability against the threshold fitted on the validation split. The arm selector
switches between the control arm and the mirror arm so the two can be compared on the
same document. The arms are in-distribution detectors: docs/evaluation.md records that
both of them miss 63 to 96 percent of out-of-distribution AI text, and the page says so
rather than implying a general-purpose detector.

IMAGE TAB. The verdict is decided in a fixed priority order, and the panel names which
rule fired:

  1. A self-declaration in the file (IPTC digitalSourceType trainedAlgorithmicMedia, a
     C2PA manifest, a Stable Diffusion parameter block). This is evidence from the
     generator itself and outranks any statistical guess.
  2. The visual detector, but only when its polarity and operating point have been
     measured on labelled images and recorded under reports/experiments. The label
     index is resolved from measurement, never from id2label documentation.
  3. Otherwise a refusal. No score is synthesised from metadata, because metadata-based
     guessing fires hardest on screenshots, chat-app downloads and re-saved photographs:
     exactly the false accusation against innocent human images that this project exists
     to prevent.

The detector is a measured baseline, not a trained FORGE model, and it is uncalibrated:
it has no validation split of its own. The evidence panel says that in those words, and
a test forbids the old "calibrated on validation data" wording from returning.

THREE VERDICTS, NOT TWO. AI, NO AI, and UNCERTAIN. The uncertain band exists because the
operating point was fitted at zero false positives on a small human set, and an image
landing between the human ceiling and the confident-AI floor is a case the measurement
does not cover. Reporting it as a verdict would be inventing precision.

ATTRIBUTION. Occlusion attribution in the model's own preprocessed tensor space, one
batched forward pass over a 5x5 grid, so the map shows which regions actually move the
detector's score. The low-level forensic maps (error level analysis, residuals) are a
separate, clearly labelled section: they are image statistics, not model attribution.

EVIDENCE STREAMS SHOW A WORD, NOT A PERCENTAGE. `strength` is how much a stream has to
say, not a probability. "Camera metadata 50%" reads as "50% likely to be a photograph",
which is not what it means and is not recoverable from a caption underneath.

Run:  python -m uvicorn api.forge_app:app --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_TEXT_CHARS = 50_000

# MPO is a JPEG container holding more than one frame: dual-lens phones and 3D cameras
# write it, and so does every iPhone that captures depth. Rejecting it turned away ordinary
# photographs, which is the exact population this project exists to protect from false
# accusations, and it was rejected by an allowlist rather than by anything measured.
# The list and the decoder hints live with the analysis in forge.image.analysis now that
# the Streamlit page performs the same analysis; they are re-exported here because the
# page's copy names the formats and the tests import them from this module.
from forge.image.analysis import ACCEPTED, NEEDS_DECODER  # noqa: E402,F401

app = FastAPI(title="FORGE", version="0.2.0")


@app.get("/health")
def health() -> dict:
    """Report the detectors this process can actually serve, by asking them.

    These three fields were hardcoded False with the mode string "detectors pending
    training", and stayed that way after both detectors were built and wired. A health
    endpoint that answers from a literal is not a health endpoint: it cannot report the
    outage it exists to report, and here it reported an outage that was not happening.

    `scorer.available()` and `detector_state()` already existed for exactly this and are
    documented never to raise. Both load lazily and are cached, so the first call pays the
    load and later calls are cheap.
    """
    from forge.image.detector import detector_state
    from forge.inference.scorer import available as text_arms

    arms = text_arms()
    image = detector_state()
    ready = sorted(name for name, state in arms.items() if state == "ready")
    return {
        "status": "ok",
        "image_detector_loaded": bool(image.get("available")),
        "image_detector": image,
        "text_detector_loaded": bool(ready),
        "text_arms": arms,
        "mode": (
            f"text arms ready: {', '.join(ready) or 'none'}; "
            f"image detector: {'ready' if image.get('available') else 'unavailable'}"
        ),
    }


@app.post("/v1/image/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    stability: bool = False,
) -> JSONResponse:
    """Analyse one image. The work lives in forge.image.analysis.

    It moved there when the Streamlit deployment needed the same payload in-process. This
    route keeps what is genuinely HTTP: the size limit, and turning a refusal into a status
    code rather than an exception.
    """
    from forge.image.analysis import UnsupportedImage, analyse

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"file exceeds {MAX_IMAGE_BYTES} bytes")

    try:
        payload = analyse(data, filename=file.filename or "upload", with_stability=stability)
    except UnsupportedImage as refusal:
        raise HTTPException(status_code=415, detail=str(refusal)) from refusal
    return JSONResponse(payload)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


@app.post("/v1/text/analyze")
def analyze_text(req: TextRequest) -> JSONResponse:
    """Score with both arms. The work lives in forge.inference.text_api.

    It moved there when the Streamlit deployment needed the same result in-process rather
    than over HTTP. Two copies of the scoring loop would put the two pages one edit apart
    from disagreeing about what a verdict is.
    """
    from forge.inference.text_api import analyse

    return JSONResponse(analyse(req.text))


@app.get("/v1/results")
def results() -> JSONResponse:
    """Measured results, read from the committed run records. Never recomputed."""
    from api.results import build

    return JSONResponse(build())


@app.get("/v1/image/detector")
def image_detector_state() -> JSONResponse:
    """Whether a visual detector can serve, and if not, the reason. Never raises."""
    from forge.image.detector import detector_state

    return JSONResponse(detector_state())


@app.get("/v1/text/arms")
def text_arms() -> JSONResponse:
    """Which arms can serve right now, and for those that cannot, the reason."""
    from forge.inference.scorer import available

    return JSONResponse(available())


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # The stylesheet is shared with streamlit_app.py and lives in forge.ui.theme, so the
    # two shells over these detectors cannot drift into looking like different products.
    from forge.ui.theme import CSS

    return _PAGE.replace("%%FORGE_CSS%%", CSS)


_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FORGE</title>
<style>
%%FORGE_CSS%%</style></head><body>
<header>
  <div class="brand">FORGE <span>Detect</span></div>
  <div class="tabs">
    <div class="tab on" data-tab="text">Text</div>
    <div class="tab" data-tab="image">Image</div>
  </div>
  <div class="status" id="status">
    <span class="dot na"></span><span id="statusText">CPU &middot; checking detectors&hellip;</span>
  </div>
</header>
<main>
  <section id="pane-image" class="hide">
    <div class="drop" id="drop">
      <h2>Drop an image, or click to choose one</h2>
      <p>JPEG, PNG, WEBP, TIFF, BMP, GIF, MPO. Analysed in memory, never written to disk.</p>
      <input type="file" id="file" accept="image/*" hidden>
    </div>
    <div id="outImage"></div>
  </section>
  <section id="pane-text">
    <textarea id="text" placeholder="Paste text to analyse. A few paragraphs works best; very short passages carry little signal."></textarea>
    <button id="go">Analyse text</button>
    <div id="outText"></div>
  </section>
</main>
<footer>Built by Akila Lourdes Miriyala Francis</footer>
<script>
const $=s=>document.querySelector(s);
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===t));
  $('#pane-image').classList.toggle('hide',t.dataset.tab!=='image');
  $('#pane-text').classList.toggle('hide',t.dataset.tab!=='text');
});
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const cls=s=>({high:'hot',medium:'warn',present:'ok',low:'ok',not_found:'na',not_checked:'na'}[s]||'na');
const lbl=s=>({not_found:'not found',not_checked:'not checked'}[s]||s);
const row=(k,v,st)=>`<div class="row"><span class="k">${esc(k)}</span><span class="v">${
  v!=null&&v!==''?esc(v):'<span class="pill na">not available</span>'} ${
  st?`<span class="pill ${cls(st)}">${esc(lbl(st))}</span>`:''}</span></div>`;
const rows=list=>list.map(r=>row(r.label,r.value,r.status)).join('');

// Manipulation rows carry raw statistics, not probabilities. "Resizing 373.65" reads as a
// quantity of editing; it is the strength of a periodic pattern on an open-ended scale. So
// the chip is the headline and the number is demoted to a technical line beneath it.
const techRows=list=>list.map(r=>`<div class="techrow">
  <div class="row"><span class="k">${esc(r.label)}</span><span class="v">${
    r.status?`<span class="pill ${cls(r.status)}">${esc(lbl(r.status))}</span>`
            :'<span class="pill na">not available</span>'}</span></div>
  ${r.value!=null&&r.value!==''?`<div class="tech">raw statistic ${esc(r.value)}</div>`:''}
  ${r.note?`<div class="tech">${esc(r.note)}</div>`:''}</div>`).join('');

// System state, read from the server rather than asserted in markup. A page that claims
// "detector ready" while nothing is loaded is the same class of lie as a fabricated score.
(async()=>{
  let s={}, img={}; try{ s=await (await fetch('/v1/text/arms')).json(); }catch(e){}
  try{ img=await (await fetch('/v1/image/detector')).json(); }catch(e){}
  const ready=Object.values(s).filter(v=>v==='ready').length;
  const dot=$('#status .dot'), txt=$('#statusText');
  txt.textContent='CPU · text detector '+(ready?'ready':'not loaded')
    +' · image detector '+(img.available?'ready':'not loaded');
  dot.className='dot '+(ready||img.available?'ok':'na');
})();

const drop=$('#drop'),input=$('#file');
drop.onclick=()=>input.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');
  if(e.dataTransfer.files[0])sendImage(e.dataTransfer.files[0])};
input.onchange=()=>input.files[0]&&sendImage(input.files[0]);

async function sendImage(f){
  // Remembered so the opt-in stability pass can re-run on the same file without asking
  // the user to pick it again. A fresh upload resets the opt-in.
  if(f){ window._forgeFile=f; window._forgeStability=false; } else { f=window._forgeFile; }
  if(!f) return;
  $('#outImage').innerHTML='<div class="card">Analysing '+esc(f.name)+'…</div>';
  const body=new FormData(); body.append('file',f);
  let r; try{ r=await fetch('/v1/image/analyze?stability='+(window._forgeStability?'true':'false'),
    {method:'POST',body}); }
  catch(e){ $('#outImage').innerHTML='<div class="card err">'+esc(e)+'</div>'; return; }
  if(!r.ok){ const d=await r.json().catch(()=>({detail:r.statusText}));
    $('#outImage').innerHTML='<div class="card err">'+esc(d.detail)+'</div>'; return; }
  renderImage(await r.json());
}

// The overall assessment banner. ONE shape for both tabs, with the score slot either
// filled by a calibrated model or visibly pending. Built pending rather than omitted so
// that the day a detector exists it fills in, instead of the panel being designed then.
// It is never filled from metadata: a number synthesised from EXIF and JPEG tables would
// fire hardest on screenshots and re-saved photographs, which are exactly the innocent
// files this project exists to protect.
function banner(o){
  // A declared verdict carries no probability: the marker is a label, not a score. The
  // gauge is hidden rather than filled with an invented number.
  const pending = !o.available;
  const hasScore = o.probability !== null && o.probability !== undefined;
  const pct = (pending || !hasScore) ? null : (o.probability*100);
  const cls = pending ? 'pending' : (o.verdict==='ai'?'hot':o.verdict==='human'?'ok':'warn');
  const pos = pct===null ? 50 : Math.min(100, Math.max(0, pct));
  // "NO AI DETECTED", never "HUMAN". A detector cannot establish that a person made
  // something; it can only report whether the input looks like the AI distribution it was
  // trained on. "Human" is a claim about the world, "no AI detected" is a claim about the
  // model, and only the second one is true.
  const WORD = {ai:'AI DETECTED', human:'NO AI DETECTED', uncertain:'UNCERTAIN'};
  return `<div class="assess big ${cls}">
    <div class="verdictbox">
      <div class="mark">${pending?'?':(o.verdict==='ai'?'!':o.verdict==='human'?'✓':'~')}</div>
      <div>
        <h2>${esc(pending?'No verdict':(WORD[o.verdict]||o.verdict.toUpperCase()))}</h2>
        <p>${esc(o.reason||'')}</p>
      </div>
    </div>
    <div class="scorebox">
      <div class="scorelabel">${esc(o.score_label||'AI probability')}</div>
      <div class="score ${pct===null?'na':''}">${pct===null
        ?(pending?'not available':'declared'):pct.toFixed(1)+'%'}</div>
      ${pct===null?'':`<div class="gauge"><div class="needle" style="left:${pos}%"></div></div>
      <div class="gaugeends"><span>human</span><span>uncertain</span><span>AI</span></div>`}
      <div class="scorenote">${esc(o.score_note||'')}</div>
    </div></div>`;
}

function renderImage(d){
  const a=d.assessment, ev=d.evidence;
  const DIR={toward_ai:'points to AI',toward_capture:'points to capture',neutral:''};
  const primary = ev.streams.find(s=>s.key==='visual_model');
  const supporting = ev.streams.filter(s=>s.key!=='visual_model');

  const streamRow = s=>`<div class="stream"><div class="top"><span>${esc(s.label)}</span>
      <span><span class="pill ${s.available?'':'na'}">${esc(s.state)}</span></span></div>
      <div class="track"><div class="fill ${s.direction==='toward_ai'?'ai':
        s.direction==='toward_capture'?'cap':(s.available?'':'off')}"
        style="width:${s.available?s.strength:100}%"></div></div>
      <div class="sub">${esc(s.summary)}${s.available&&DIR[s.direction]
        ?` <span class="dir">${esc(DIR[s.direction])}</span>`:''}</div></div>`;

  let h = banner({
    available: !!a.available,
    verdict: a.verdict,
    probability: a.confidence,
    reason: a.reason,
    score_label: a.confidence==null ? 'Determination' : 'AI probability',
    score_note: a.detail||'',
  }) + `<div class="grid">`;

  h+=`<div class="card"><h3>Image</h3>
    ${d.preview?`<img class="prev" src="${d.preview}" alt="preview">`:''}
    <div style="margin-top:11px">
    ${row('File',d.filename)}${row('Size',(d.size_bytes/1024).toFixed(0)+' KB')}</div></div>`;

  h+=`<div class="card"><h3>Evidence</h3>
    <div class="grouplabel">Detector</div>${primary?streamRow(primary):''}
    <div class="grouplabel">Supporting signals</div>${supporting.map(streamRow).join('')}
    <div class="row" style="margin-top:8px"><span class="k">Agreement</span>
    <span class="v"><span class="pill ${cls(ev.conflict)}">${
      ev.conflict==='low'?'consistent':esc(ev.conflict)}</span></span></div>
    ${ev.conflict!=='low'?`<p class="caveat">${esc(ev.conflict_reason)}</p>`:''}
    </div>`;

  h+=`<div class="card"><h3>File signals</h3>${rows(d.authenticity)}</div>`;

  // Detector robustness: does the VERDICT survive the transforms an image meets in the
  // wild? Runs on every request now that it costs about a second, so there is no button
  // and no second pass over the same file.
  const R=d.robustness||[];
  if(R.length){
    const held=R.filter(r=>!r.error&&!r.changed).length, n=R.filter(r=>!r.error).length;
    h+=`<div class="card wide"><h3>Robustness
      <span class="sum">${held} of ${n} edits keep the same verdict</span></h3>
      <p class="caveat" style="margin:0 0 12px">Every edit is re-scored and compared with the
      original. "flipped" means that edit changes the answer.</p>
      <div class="rgrid">${R.map(r=>r.error
        ?`<div class="chip"><div class="n">${esc(r.attack)}</div>
          <div class="d">failed</div></div>`
        :`<div class="chip"><div class="n">${esc(r.attack)}
          <span class="pill ${r.changed?'hot':'ok'}">${r.changed?'flipped':'held'}</span></div>
          <div class="d">${(r.ai_probability*100).toFixed(1)}%${r.attack==='original'?''
            :` · ${r.delta>=0?'+':''}${(r.delta*100).toFixed(1)} pts`}</div></div>`).join('')}
      </div></div>`;
  }

  // ATTRIBUTION, from the detector itself. Each region is hidden and the image re-scored:
  // a warm cell is one the decision rested on. The old panel here showed compression and
  // noise statistics, which know nothing about the model and were read as if they did.
  const attr=d.attribution_map;
  if(attr&&attr.image){
    h+=`<div class="card wide"><h3>What the verdict rests on
      <span class="sum">${attr.grid}×${attr.grid} occlusion</span></h3>
      <div class="attr"><img src="${attr.image}" alt="attribution map">
        <div><p class="caveat">${esc(attr.reading)}</p>
        <div class="row"><span class="k">Score with everything visible</span>
          <span class="v">${(attr.base_probability*100).toFixed(1)}%</span></div>
        <div class="row"><span class="k">Largest single drop when hidden</span>
          <span class="v">${(attr.peak.drop*100).toFixed(1)} pts</span></div>
        <p class="caveat">Measured by hiding each region and re-scoring, not by a gradient
        approximation, so it holds for any detector this project loads.</p></div></div></div>`;
  }

  const maps=d.maps||[];
  if(maps.length){
    h+=`<details class="card wide fold"><summary>Pixel statistics</summary>
      <div class="foldbody">
      <div class="maps">${maps.map(m=>`<figure>
        <img src="${m.image}" alt="${esc(m.title)}">
        <figcaption><b>${esc(m.title)}</b><br>${esc(m.short||m.what_it_shows)}</figcaption>
        </figure>`).join('')}</div>
      <p class="caveat">Computed from the file without any model. The attribution panel
      above is what the detector used.</p></div></details>`;
  }

  const T=d.timings_ms||{}, order=['preview','detector','forensics','perceptual_hash',
    'evidence','attribution','forensic_maps','detector_robustness'];
  const NAME={preview:'Preview',detector:'Detector inference',forensics:'Forensic analysis',
    perceptual_hash:'Perceptual hash',evidence:'Evidence engine',
    attribution:'Attribution map',forensic_maps:'Pixel diagnostics',
    detector_robustness:'Robustness'};
  h+=`<details class="card wide fold"><summary>Details
      <span class="sum">${(d.elapsed_ms/1000).toFixed(1)} s on CPU</span></summary>
    <div class="foldbody">
    <h3>Manipulation analysis</h3>${techRows(d.manipulation)}
    <h3 style="margin-top:18px">Provenance</h3>${rows(d.provenance)}
    <h3 style="margin-top:18px">Timing</h3>
    ${order.filter(k=>k in T).map(k=>`<div class="row"><span class="k">${esc(NAME[k]||k)}</span>
      <span class="v">${T[k].toLocaleString()} ms</span></div>`).join('')}
    ${row('Total',d.elapsed_ms.toLocaleString()+' ms')}
    ${d.detector&&d.detector.available
      ?`<p class="tech">Detector ${esc(d.detector.model_id)}, uncalibrated baseline.</p>`:''}
    </div></details>`;

  h+='</div>';
  $('#outImage').innerHTML=h;
}


// ---------------------------------------------------------------------------
// The Results tab was removed from the interface: the measured numbers belong in
// docs/evaluation.md and docs/writeup.md, where they sit next to their method, rather
// than in a third tab competing with the two live detectors. /v1/results and
// api/results.py stay, and still read only from committed run records, so the figures
// remain available as JSON without a page that has to be kept in step with the docs.

$('#go').onclick=async()=>{
  const text=$('#text').value.trim();
  if(!text){ $('#outText').innerHTML='<div class="card err">Paste some text first.</div>'; return; }
  $('#outText').innerHTML='<div class="card">Analysing…</div>';
  let r; try{ r=await fetch('/v1/text/analyze',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({text})}); }
  catch(e){ $('#outText').innerHTML='<div class="card err">'+esc(e)+'</div>'; return; }
  const d=await r.json();
  const why=Object.entries(d.unavailable||{})
    .map(([a,m])=>`<div class="card err"><h3>${esc(a)} unavailable</h3><p>${esc(m)}</p></div>`)
    .join('');
  if(!d.available){
    $('#outText').innerHTML=banner({
      available:false,
      reason:'Neither arm could be loaded, so no score is produced. A number here would be '
        +'fabricated.',
      score_label:'AI probability, calibrated',
      score_note:'Load a trained checkpoint into outputs/ to enable scoring.',
    })+`<div class="grid">${why}
      <div class="card"><h3>Input</h3>${row('Words',d.words)}</div></div>`;
    return;
  }

  // The deployed arm is the headline. The other arm is shown beside it because the
  // comparison is the experiment, but a product has one verdict, not two.
  const primary = d.arms.find(a=>a.arm==='mirror') || d.arms[0];
  const agree = d.arms.length>1 && d.arms[0].verdict===d.arms[1].verdict;
  const head = banner({
    available: true,
    verdict: primary.verdict,
    probability: primary.ai_probability,
    reason: (d.arms.length>1
      ? (agree ? 'Both arms agree.' : 'THE ARMS DISAGREE: '
          + d.arms.map(a=>a.label+' says '+a.verdict).join(', ') + '.')
      : 'Only one arm is loaded.')
      + ` Mean over ${primary.n_windows} window${primary.n_windows===1?'':'s'}, `
      + `${d.words} words.`,
    score_label: 'AI probability, calibrated',
    score_note: `${esc(primary.label)}, threshold ${primary.threshold.toFixed(6)} at a `
      + `${primary.fpr_budget} false-positive budget. Validation ECE ${primary.val_ece}. `
      + 'Out of distribution this model is badly miscalibrated: see the note below.',
  });

  const spark = w => w.length<2 ? '' :
    `<div class="spark">${w.map(p=>`<i style="height:${Math.max(2,Math.round(p*100))}%"></i>`).join('')}</div>`;

  const cards = d.arms.map(a=>`<div class="card"><h3>${esc(a.label)}</h3>
    ${row('Verdict',a.verdict,a.verdict==='ai'?'hot':(a.verdict==='human'?'ok':'warn'))}
    ${row('AI probability (mean over windows)',(a.ai_probability*100).toFixed(2)+'%')}
    ${row('Highest single window',(a.max_window_probability*100).toFixed(2)+'%')}
    ${row('Deployed threshold',a.threshold.toFixed(6))}
    ${row('FPR budget',a.fpr_budget)}
    ${row('Distance from threshold',(a.confidence*100).toFixed(1)+'%')}
    ${row('Windows scored',a.n_windows)}
    ${row('Its validation FNR',(a.val_fnr*100).toFixed(3)+'%')}
    ${row('Its validation ECE',a.val_ece)}
    ${spark(a.windows)}</div>`).join('');

  $('#outText').innerHTML=head+`<div class="grid">${cards}${why}
    <div class="card wide"><h3>What this score is not</h3>
      <p>${esc(d.caveat)}</p><p>${esc(d.abstention)}</p>
      <p class="tech" style="margin-top:9px">Provenance, for reproducibility rather than
      display: ${d.arms.map(a=>esc(a.model_version)).join(' · ')}</p></div></div>`;
};
</script></body></html>"""
