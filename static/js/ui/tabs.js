// Remove the upload and export tab logic since they're now in the sidebar
window.APP_STATE = { imgEl: null, hasImage: false, edits: {} };

(() => {
  const q = (s, r=document) => r.querySelector(s);
  const qa = (s, r=document) => [...r.querySelectorAll(s)];

  // Primary tabs (only edit, crop, nst now)
  const tabs = qa('.tabs .tab');
  const panels = qa('.tab-panel');
  tabs.forEach(btn => btn.addEventListener('click', () => {
    tabs.forEach(b => b.classList.toggle('is-active', b===btn));
    panels.forEach(p => p.classList.toggle('is-active', p.id === `tab-${btn.dataset.tab}`));
  }));

  // Subtabs (Edit)
  const subtabs = qa('.subtabs .subtab');
  const subPanels = qa('.subtab-panel');
  subtabs.forEach(btn => btn.addEventListener('click', () => {
    subtabs.forEach(b => b.classList.toggle('is-active', b===btn));
    subPanels.forEach(p => p.classList.toggle('is-active', p.id === `subtab-${btn.dataset.subtab}`));
  }));

  // Cache preview img
  APP_STATE.imgEl = q('#preview-image');
})();

// Start with editing tab active (no need for upload tab)
document.addEventListener('DOMContentLoaded', () => {
  const editTab = document.querySelector('.tab[data-tab="edit"]');
  if (editTab) editTab.click();
});
