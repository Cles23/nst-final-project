(() => {
  const id = s => document.getElementById(s);
  const img = id('preview-image');

  const fields = ['exposure','contrast','whites','blacks','temperature','vibrance','saturation','sharpness','grain','dehaze'];
  const outputs = Object.fromEntries(fields.map(k => [k, id(`${k}-out`)]));

  // Server preview (debounced)
  let previewTimeout = null;
  function schedulePreview() {
    clearTimeout(previewTimeout);
    previewTimeout = setTimeout(async () => {
      if (!APP_STATE.hasImage) return;
      
      try {
        const res = await fetch('/api/preview', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ 
            edits: APP_STATE.edits || {}, 
            crop: APP_STATE.crop || {} 
          })
        });
        
        if (!res.ok) {
          console.error('Preview failed:', res.status);
          return;
        }
        
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        img.src = url;
      } catch (err) {
        console.error('Preview error:', err);
      }
    }, 300);
  }

  function bind(name) {
    const el = id(name), out = outputs[name];
    if (!el) return;
    
    el.addEventListener('input', () => {
      APP_STATE.edits[name] = Number(el.value);
      if (out) out.textContent = el.value;
      schedulePreview();
    });
  }
  
  fields.forEach(bind);

  // Export for other modules to use
  window.EDIT = { schedulePreview };
})();
