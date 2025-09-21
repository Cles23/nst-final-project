(() => {
  const id = s => document.getElementById(s);
  const img = id('preview-image');
  const stage = id('stage');
  if (stage){ stage.style.overflow='hidden'; stage.style.position='relative'; }

  APP_STATE.crop = APP_STATE.crop || { zoom:1, x:0, y:0, rot:0 };

  const zoom = id('zoom'), panX = id('pan-x'), panY = id('pan-y'), rot = id('rotation');

  // Remove CSS transform - use server preview instead
  function apply(){
    // No more CSS transforms - let server handle it
    schedulePreview();
  }

  function bind(el, key, map=v=>v){
    el?.addEventListener('input', () => {
      APP_STATE.crop[key] = map(Number(el.value));
      apply();
      const out = document.getElementById(`${el.id}-out`);
      if (out) out.textContent = el.value;
    });
  }
  
  bind(zoom, 'zoom', v => Math.max(1, Math.min(5, v)));
  bind(panX, 'x', v => Math.max(-100, Math.min(100, v)));
  bind(panY, 'y', v => Math.max(-100, Math.min(100, v)));
  bind(rot,  'rot', v => ((v%360)+360)%360);

  // Remove the CSS transform on image load
  img.addEventListener('load', () => {
    // Reset any CSS transforms
    img.style.transform = '';
    img.style.transformOrigin = '';
  });

  // Debounced server preview
  let t=null;
  function schedulePreview(){
    clearTimeout(t);
    t=setTimeout(async () => {
      if (!APP_STATE.hasImage) return;
      
      try {
        const res = await fetch('/api/preview', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ 
            edits: APP_STATE.edits || {}, 
            crop: APP_STATE.crop || {} 
          })
        });
        
        if (!res.ok) {
          console.error('Crop preview failed:', res.status);
          return;
        }
        
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        img.src = url;
      } catch (err) {
        console.error('Crop preview error:', err);
      }
    }, 250);
  }

  // Reset crop
  document.getElementById('crop-reset-btn')?.addEventListener('click', () => {
    APP_STATE.crop = { zoom:1, x:0, y:0, rot:0 };
    ['zoom','pan-x','pan-y','rotation'].forEach(idv=>{
      const el=id(idv), out=id(`${idv}-out`);
      if (el) el.value = idv==='zoom'?1:0;
      if (out) out.textContent = el ? el.value : '';
    });
    
    // Reset CSS transforms and trigger server preview
    img.style.transform = '';
    img.style.transformOrigin = '';
    schedulePreview();
  });

  // Export for other modules
  window.CROP = { schedulePreview };
})();
