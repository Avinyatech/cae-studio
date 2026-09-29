import { Mesh } from "./api";

export interface ViewerFrame {
  label: string;
  freqHz?: number;
  vectors: Record<string, number[]>; // nodeId -> [T1,T2,T3,R1,R2,R3]
}

export function buildViewerHtml(mesh: Mesh, frames: ViewerFrame[]): string {
  const nodesJson = JSON.stringify(mesh.nodes);
  const elementsJson = JSON.stringify(mesh.elements);
  const fixedJson = JSON.stringify(mesh.fixed_ids);
  const framesJson = JSON.stringify(
    frames.map((f) => ({ label: f.label, freq: f.freqHz ?? null, vec: f.vectors }))
  );

  return `<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
  html,body{margin:0;padding:0;background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,sans-serif;color:#1a1a1a;}
  #wrap{padding:8px;box-sizing:border-box;}
  .modebar{display:flex;overflow-x:auto;gap:6px;margin-bottom:8px;padding-bottom:2px;-webkit-overflow-scrolling:touch;}
  .modebtn{flex:0 0 auto;background:#f7f7f8;border:1px solid #e5e5e5;border-radius:8px;padding:6px 10px;font-size:11px;color:#6b6b6b;white-space:nowrap;}
  .modebtn b{display:block;font-size:12px;color:#1a1a1a;}
  .modebtn.active{border-color:#3b82f6;color:#3b82f6;}
  .modebtn.active b{color:#3b82f6;}
  .stage{background:#f7f7f8;border:1px solid #e5e5e5;border-radius:10px;padding:6px;position:relative;}
  .hint{position:absolute;top:10px;left:12px;font-size:12px;font-weight:600;color:#6b6b6b;}
  .controls{display:flex;align-items:center;gap:10px;margin-top:8px;font-size:12px;color:#6b6b6b;}
  .controls input[type=range]{flex:1;}
  .playbtn{background:#3b82f6;color:#fff;border:none;border-radius:6px;width:32px;height:32px;font-size:14px;}
  svg{width:100%;height:auto;display:block;}
</style>
</head>
<body>
<div id="wrap">
  <div class="modebar" id="modebar"></div>
  <div class="stage">
    <div class="hint" id="freqLabel"></div>
    <svg id="svg" viewBox="0 0 360 320"></svg>
  </div>
  <div class="controls">
    <button class="playbtn" id="playBtn">&#9654;</button>
    <label style="display:flex;align-items:center;gap:6px;flex:1;">
      Amplitude
      <input type="range" id="ampSlider" min="1" max="40" value="14">
    </label>
  </div>
</div>
<script>
(function(){
  var nodes = ${nodesJson};
  var elements = ${elementsJson};
  var fixedIds = ${fixedJson};
  var frames = ${framesJson};

  var svgNS = "http://www.w3.org/2000/svg";
  var svg = document.getElementById('svg');
  var modebar = document.getElementById('modebar');
  var freqLabel = document.getElementById('freqLabel');
  var ampSlider = document.getElementById('ampSlider');
  var playBtn = document.getElementById('playBtn');

  var current = 0, playing = true, t0 = performance.now();
  var cos30 = Math.cos(Math.PI/6), sin30 = Math.sin(Math.PI/6);

  // compute bounds for auto-fit scale/offset
  var xs=[], ys=[], zs=[];
  Object.keys(nodes).forEach(function(id){ var n=nodes[id]; xs.push(n[0]); ys.push(n[1]); zs.push(n[2]); });
  var minX=Math.min.apply(null,xs), maxX=Math.max.apply(null,xs);
  var minY=Math.min.apply(null,ys), maxY=Math.max.apply(null,ys);
  var margin = Math.max(maxX-minX, maxY-minY) * 0.35 + 20;

  function project(x,y,z){
    var sx = (x - y) * cos30;
    var sy = (x + y) * sin30 - z;
    return [sx, sy];
  }
  // determine projected bbox using corners + margin allowance for deformation
  var corners = [[minX-margin,minY-margin,-margin],[maxX+margin,minY-margin,-margin],
                 [minX-margin,maxY+margin,margin],[maxX+margin,maxY+margin,margin]];
  var pxs=[], pys=[];
  corners.forEach(function(c){ var p=project(c[0],c[1],c[2]); pxs.push(p[0]); pys.push(p[1]); });
  var bw = Math.max.apply(null,pxs)-Math.min.apply(null,pxs);
  var bh = Math.max.apply(null,pys)-Math.min.apply(null,pys);
  var viewW = 360, viewH = 320;
  var scale = Math.min(viewW/bw, viewH/bh) * 0.92;
  var cx = viewW/2 - ((Math.max.apply(null,pxs)+Math.min.apply(null,pxs))/2)*scale;
  var cy = viewH/2 - ((Math.max.apply(null,pys)+Math.min.apply(null,pys))/2)*scale;

  function toScreen(x,y,z){
    var p = project(x,y,z);
    return [cx + p[0]*scale, cy + p[1]*scale];
  }

  function buildModeBar(){
    frames.forEach(function(f,i){
      var btn = document.createElement('div');
      btn.className = 'modebtn' + (i===current?' active':'');
      var freqStr = f.freq!=null ? (f.freq>=1000?Math.round(f.freq).toLocaleString():f.freq.toFixed(1))+' Hz' : '';
      btn.innerHTML = '<b>' + f.label + '</b>' + freqStr;
      btn.addEventListener('click', function(){ current=i; refreshBar(); });
      modebar.appendChild(btn);
    });
  }
  function refreshBar(){
    Array.prototype.forEach.call(modebar.children, function(el,i){
      el.className = 'modebtn' + (i===current?' active':'');
    });
  }

  function el(tag, attrs){
    var e = document.createElementNS(svgNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function polyPoints(fn, quad){
    return quad.map(function(id){ var p = fn(id); return p[0].toFixed(2)+','+p[1].toFixed(2); }).join(' ');
  }

  function render(){
    svg.innerHTML = '';
    svg.setAttribute('viewBox', '0 0 ' + viewW + ' ' + viewH);
    var frame = frames[current];
    var amp = parseFloat(ampSlider.value);
    var phase = playing ? Math.sin((performance.now()-t0)/600) : 1;

    function undefP(id){ var n = nodes[id]; return toScreen(n[0],n[1],n[2]); }
    function defP(id){
      var n = nodes[id], v = frame.vec[id] || [0,0,0,0,0,0];
      return toScreen(n[0]+v[0]*amp*phase, n[1]+v[1]*amp*phase, n[2]+v[2]*amp*phase);
    }

    elements.forEach(function(q){
      svg.appendChild(el('polygon', {
        points: polyPoints(undefP, q), fill:'none', stroke:'#3b82f6',
        'stroke-opacity':'0.25','stroke-width':'1','stroke-dasharray':'3,3'
      }));
    });
    elements.forEach(function(q){
      svg.appendChild(el('polygon', {
        points: polyPoints(defP, q), fill:'rgba(239,68,68,0.14)',
        stroke:'#ef4444','stroke-width':'1.6','stroke-linejoin':'round'
      }));
    });
    Object.keys(nodes).forEach(function(idStr){
      var id = parseInt(idStr,10);
      var isFixed = fixedIds.indexOf(id) !== -1;
      var p = defP(id);
      svg.appendChild(el('circle', {
        cx:p[0], cy:p[1], r: isFixed?3.2:2.8,
        fill: isFixed?'#1a1a1a':'#ef4444', stroke:'#fff','stroke-width':'0.8'
      }));
    });

    var freqStr = frame.freq!=null ? ((frame.freq>=1000?Math.round(frame.freq).toLocaleString():frame.freq.toFixed(1))+' Hz') : '';
    freqLabel.textContent = frame.label + (freqStr ? (' — ' + freqStr) : '');
  }

  function loop(){ render(); requestAnimationFrame(loop); }
  buildModeBar();
  playBtn.addEventListener('click', function(){
    playing = !playing;
    playBtn.innerHTML = playing ? '&#9654;' : '&#10074;&#10074;';
    if (playing) t0 = performance.now();
  });
  requestAnimationFrame(loop);
})();
</script>
</body>
</html>`;
}
