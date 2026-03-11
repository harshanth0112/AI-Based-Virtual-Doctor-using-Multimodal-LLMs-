function showToast(msg,type='success'){
  const t=document.getElementById('toast');if(!t)return;
  t.textContent=msg;t.className='show';
  t.style.background=type==='error'?'rgba(220,38,38,.15)':'rgba(16,185,129,.15)';
  t.style.borderColor=type==='error'?'rgba(220,38,38,.35)':'rgba(16,185,129,.35)';
  t.style.color=type==='error'?'#FCA5A5':'#6EE7B7';
  setTimeout(()=>t.className='',3200);
}
function toggleSidebar(){document.querySelector('.sidebar').classList.toggle('open')}
const EKW=["chest pain","can't breathe","cannot breathe","heart attack","stroke","unconscious","severe bleeding","seizure","convulsion","overdose","poisoning","choking","suicidal","paralysis","face drooping"];
function checkEmergency(t){return EKW.filter(k=>t.toLowerCase().includes(k))}
