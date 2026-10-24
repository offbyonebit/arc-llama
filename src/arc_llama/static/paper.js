/* Shared Paper shell. Receives the existing controllers' state; never polls APIs. */
(function () {
  'use strict';
  const $ = s => document.querySelector(s);
  const node = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
  const icons = {
    Chat:'<path d="M4 5h16v11H9l-5 4z"/>',
    Work:'<rect x="4" y="6" width="16" height="14" rx="2"/><path d="M9 6V3h6v3M4 12h16M10 12v3h4v-3"/>',
    Models:'<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5"/>',
    Discover:'<circle cx="12" cy="12" r="9"/><path d="M16 8l-2.5 5.5L8 16l2.5-5.5z"/>',
    System:'<circle cx="12" cy="12" r="3"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1"/>'
  };
  const chat = !!$('#chat-log');
  document.body.classList.add('paper', chat ? 'paper-chat' : 'paper-dashboard');
  const header = $('body > header');
  const rail = node('aside','paper-rail'); rail.setAttribute('aria-label','Arc Llama');
  const logo = node('a','paper-logo'); logo.href='/chat'; logo.title='Arc Llama';
  const img = $('header .brand-logo');
  if (img) { img.dataset.light = img.dataset.dark = '/assets/arc-llama-logo.png?v=paper-1'; img.src=img.dataset.light; logo.append(img); }
  rail.append(logo);
  const nav = $('.app-nav');
  const links = Array.from(nav.querySelectorAll('.nav-link'));
  for (const name of ['Chat','Work','Models','Discover','System']) {
    const link = links.find(l => l.textContent.trim() === name);
    if (!link) continue;
    link.setAttribute('aria-label',name); link.title=name;
    link.replaceChildren();
    const svg = document.createElementNS('http://www.w3.org/2000/svg','svg'); svg.setAttribute('viewBox','0 0 24 24'); svg.setAttribute('aria-hidden','true'); svg.classList.add('paper-icon'); svg.innerHTML=icons[name];
    link.append(svg,node('span','paper-nav-label',name));
    if (name==='Work') { const dot=node('span','paper-nav-dot'); dot.hidden=true; dot.setAttribute('aria-label','Permission waiting'); link.append(dot); }
    if (name==='Models') { const badge=node('span','paper-nav-badge'); badge.hidden=true; link.append(badge); }
    nav.append(link);
  }
  rail.append(nav,node('div','paper-spacer'));
  const theme=$('#theme-toggle'); if (theme) { theme.classList.add('paper-rail-button'); rail.append(theme); }
  document.body.prepend(rail);
  const mobileTheme=node('button','paper-mobile-theme','◐'); mobileTheme.type='button'; mobileTheme.setAttribute('aria-label','Toggle theme'); mobileTheme.addEventListener('click',()=>theme?.click());
  const actions=$('header .meta') || $('header .header-actions');
  actions?.append(mobileTheme);
  const control=node('div','paper-model-control');
  const pill=node('button','paper-model-pill'); pill.type='button'; pill.id='paper-model-pill'; pill.setAttribute('aria-label','Switch model'); pill.setAttribute('aria-haspopup','menu'); pill.setAttribute('aria-expanded','false'); pill.setAttribute('aria-controls','paper-model-menu');
  const ring=node('span','paper-ring'); ring.setAttribute('aria-hidden','true');
  const modelName=node('span','paper-model-name','Choose a model');
  const modelState=node('span','paper-model-state');
  pill.append(ring,modelName,modelState,node('span','','⌄'));
  const menu=node('div','paper-model-menu'); menu.id='paper-model-menu'; menu.hidden=true; menu.setAttribute('aria-label','Choose model');
  const menuTitle=node('p','paper-menu-title','Your models');
  const count=node('span','paper-model-count'); menuTitle.append(count);
  const options=node('div','paper-model-options'); options.setAttribute('role','menu'); options.setAttribute('aria-label','Your models');
  const scrollHint=node('p','paper-model-scroll-hint'); scrollHint.id='paper-model-scroll-hint'; scrollHint.hidden=true;
  const updateScrollHint=()=>{
    const above=options.scrollTop>1;
    const below=options.scrollTop+options.clientHeight<options.scrollHeight-1;
    scrollHint.hidden=options.scrollHeight<=options.clientHeight+1;
    if(scrollHint.hidden) options.removeAttribute('aria-describedby'); else options.setAttribute('aria-describedby',scrollHint.id);
    scrollHint.textContent=above&&below?'↑ More models above · More below ↓':above?'↑ Scroll for earlier models':'Scroll for more models ↓';
  };
  options.addEventListener('scroll',updateScrollHint,{passive:true});
  new ResizeObserver(()=>{if(!menu.hidden)updateScrollHint();}).observe(options);
  const manage=node('a','paper-menu-link','Manage models →'); manage.href='/';
  menu.append(menuTitle,options,scrollHint,manage); control.append(pill,menu);
  if (actions) actions.prepend(control); else header.append(control);
  const positionMenu=()=>{
    if(menu.hidden)return;
    const anchor=pill.getBoundingClientRect();
    const bottomSpace=innerWidth<=820?74:14;
    const top=Math.min(anchor.bottom+8,Math.max(8,innerHeight-bottomSpace-180));
    menu.style.top=`${top}px`;
    menu.style.left=`${Math.max(14,Math.min(anchor.right-menu.offsetWidth,innerWidth-menu.offsetWidth-14))}px`;
    menu.style.right='auto';
    menu.style.maxHeight=`${Math.max(100,innerHeight-top-bottomSpace)}px`;
  };
  window.addEventListener('resize',()=>{positionMenu();if(!menu.hidden)requestAnimationFrame(updateScrollHint);});
  window.addEventListener('scroll',positionMenu,{passive:true});
  const setMenu = open => { menu.hidden=!open; pill.setAttribute('aria-expanded',String(open)); if(open) { positionMenu(); options.querySelector('[aria-checked="true"]')?.focus(); requestAnimationFrame(updateScrollHint); } };
  pill.addEventListener('click',()=>setMenu(menu.hidden));
  document.addEventListener('click',e=>{ if(!control.contains(e.target)) setMenu(false); });
  let modelSignature='';
  let chooseModel=()=>{};
  const idOf=m=>m.id || m.name;
  const nameOf=m=>m.display_name || m.metadata?.display_name || idOf(m);
  function syncModels(models, selected, choose, busy=false) {
    chooseModel=choose;
    const m=models.find(x=>idOf(x)===selected);
    const loading=m?.state==='loading' || m?.state==='draining';
    const running=!!(m?.loaded ?? m?.metadata?.loaded);
    modelName.textContent=m ? nameOf(m) : 'Choose a model';
    modelState.textContent=m ? loading ? m.state==='draining'?'Stopping':'Loading' : running ? 'Running' : 'Not loaded' : '';
    ring.className='paper-ring'+(loading?' loading':running?' running':'');
    pill.title=m ? `${nameOf(m)} · ${modelState.textContent}` : 'Choose a model';
    const sig=JSON.stringify([models.map(x=>[idOf(x),nameOf(x),x.loaded,x.metadata?.loaded,x.state]),selected,busy]);
    if(sig===modelSignature) return;
    modelSignature=sig;
    const focused=document.activeElement?.dataset?.model;
    count.textContent=`${models.length} ${models.length===1?'model':'models'}`;
    options.replaceChildren();
    if(!models.length) options.append(node('p','paper-notice','No models available. Add a model in Models.'));
    for(const model of models) {
      const item=node('button','paper-model-option'); item.type='button'; item.dataset.model=idOf(model); item.setAttribute('role','menuitemradio'); item.setAttribute('aria-checked',String(idOf(model)===selected));
      item.append(node('span','',nameOf(model)),node('small','',model.state==='loading'?'Loading':model.state==='draining'?'Stopping':(model.loaded ?? model.metadata?.loaded) ? 'Running' : 'Not loaded'));
      item.disabled=busy;
      item.addEventListener('click',()=>{ chooseModel(idOf(model)); setMenu(false); pill.focus(); });
      options.append(item);
    }
    if(focused) Array.from(options.children).find(x=>x.dataset.model===focused)?.focus();
    if(!menu.hidden) requestAnimationFrame(updateScrollHint);
  }
  menu.addEventListener('keydown',e=>{
    const items=Array.from(options.querySelectorAll('button:not(:disabled)'));
    if(e.key==='ArrowDown'||e.key==='ArrowUp') { e.preventDefault(); const i=items.indexOf(document.activeElement); items[(i+(e.key==='ArrowDown'?1:items.length-1)+items.length)%items.length]?.focus(); }
    if(e.key==='Home') {e.preventDefault();items[0]?.focus();}
    if(e.key==='End') {e.preventDefault();items.at(-1)?.focus();}
  });
  function external(label) { const badge=node('span','paper-external','External'); badge.title=label; return badge; }
  const api=window.ArcPaper={ syncModels, external };
  api.jobs=jobs=>{const active=(jobs||[]).filter(j=>['queued','downloading','registering'].includes(j.status));const badge=$('.paper-nav-badge');badge.textContent=active.length;badge.hidden=!active.length;badge.setAttribute('aria-label',`${active.length} active downloads`);};
  if(!chat) {
    const title=node('span','paper-header-title','Models'); header.querySelector('.brand').replaceChildren(title);
    const dashboard=$('.dashboard');
    const layout=node('div','paper-model-layout'); layout.dataset.area='models';
    const content=node('div','paper-model-content');
    const local=$('#model-list').closest('section'); local.id='paper-downloaded'; local.setAttribute('aria-labelledby','choose-title');
    const library=$('.library-section'); library.id='paper-discover'; library.setAttribute('aria-labelledby','library-title');
    const next=$('.next-step-section'); content.append(next,$("#library-jobs"),$("#library-recovery-status"),local,library);
    const inspector=$('.readiness-section'); inspector.classList.add('paper-inspector'); inspector.setAttribute('aria-label','Model details');
    const close=node('button','paper-panel-close paper-inspector-close','×'); close.type='button'; close.setAttribute('aria-label','Close model details'); close.addEventListener('click',()=>{document.body.classList.remove('paper-model-sheet'); $('.model-card.selected')?.focus();}); inspector.prepend(close);
    layout.append(content,inspector); dashboard.append(layout);
    api.view=view=>{
      title.textContent=view==='system'?'System':view==='discover'?'Discover':'Models';
      library.hidden=view!=='discover'; local.hidden=view==='discover';
      if(view!=='models') document.body.classList.remove('paper-model-sheet');
    };
    api.openModel=()=>document.body.classList.add('paper-model-sheet');
    $('#library-title').textContent='Discover models'; $('#library-title').append(external('Hugging Face uses an external service'));
    $('#plugins-title')?.append(external('Connected tools may use external services'));
    $('#choose-title').textContent='Your models';
    api.inspector=(model, load, stop)=>{
      $('#paper-state-ladder')?.remove(); $('#paper-model-actions')?.remove();
      if(!model)return;
      const ladder=node('div','paper-state-ladder');ladder.id='paper-state-ladder';
      const file=model.file_readiness?.available;
      ladder.append(node('div','paper-state-line'+(file===true?' yes':''),file===true?'Downloaded · files available':file===false?'File unavailable':'Download status unknown'));
      const fit=model.vram_estimate?.fit;
      ladder.append(node('div','paper-state-line estimate',fit===true?'Estimated to fit':fit===false?'Estimated to exceed memory':'Memory fit not estimated'));
      ladder.append(node('div','paper-state-line'+(model.loaded?' yes':''),model.state==='loading'?'Loading':model.state==='draining'?'Stopping':model.loaded?'Running · loaded':'Not running'));
      const actions=node('div','paper-model-actions');actions.id='paper-model-actions';
      const action=node('button',model.loaded?'':'primary',model.state==='loading'?'Loading…':model.loaded?'Stop model':'Load model');action.type='button';action.disabled=model.state==='loading'||model.state==='draining'||(!model.loaded&&file===false);action.addEventListener('click',()=>model.loaded?stop(model.name):load(model.name));
      const work=node('a','','Use in Work');work.href=`/work?model=${encodeURIComponent(model.name)}`;
      actions.append(action,work);
      $('#readiness-card .readiness-heading').after(ladder);$('#readiness-card').append(actions);
    };
  } else {
    for (const link of nav.querySelectorAll('a[href="/chat"],a[href="/work"]')) {
      link.addEventListener('click', event => {
        if (event.button || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || $('#mode-switch').hidden) return;
        event.preventDefault(); $(link.getAttribute('href')==='/work'?'#mode-work':'#mode-chat').click();
      });
    }
    const nativeSelect=$('#model-select'); nativeSelect.hidden=true;
    menu.insertBefore(nativeSelect,options); // Kept for existing state and integrations.
    const wrap=$('#input-wrap');
    const contextButton=node('button','paper-context-toggle');contextButton.type='button';contextButton.id='paper-context-toggle';contextButton.setAttribute('aria-label','Context usage');contextButton.setAttribute('aria-expanded','false');contextButton.setAttribute('aria-controls','paper-context-pop');
    const pop=node('div','paper-context-pop');pop.id='paper-context-pop';pop.hidden=true;pop.setAttribute('role','region');pop.setAttribute('aria-label','Conversation context');
    pop.append(node('h3','','Conversation context'),$('#ctx-meter'),node('p','','Token counts include the reply allowance. Estimates are marked with ~.'),$('.compact-controls'));wrap.append(contextButton,pop);
    const setContext=open=>{pop.hidden=!open;contextButton.setAttribute('aria-expanded',String(open));};
    contextButton.addEventListener('click',()=>setContext(pop.hidden));
    document.addEventListener('click',e=>{if(!pop.contains(e.target)&&!contextButton.contains(e.target))setContext(false);});
    api.budget=budget=>{
      const total=budget.total_tokens, max=budget.context_tokens;
      if(Number.isFinite(total)&&max) {const pct=Math.min(100,Math.max(0,total/max*100));contextButton.style.setProperty('--context-used',`${pct}%`);contextButton.title=`${budget.exact?'':'Estimated: '}${total.toLocaleString()} of ${max.toLocaleString()} tokens`;contextButton.classList.toggle('over-budget',total>max);}
    };
    const historyButton=$('#history-toggle');
    historyButton.classList.add('paper-labeled-control');
    historyButton.append(node('span','','History'));
    historyButton.title='Resume a saved chat';
    // The context count stays visible beside the composer; the ring opens details.
    const contextLabel=node('span','paper-context-visible','Context');
    $('#input-area-context-anchor')?.remove();
    wrap.after(contextLabel);
    wrap.append($("#plugin-tools"));
    const originalBudget=api.budget;
    api.budget=budget=>{
      originalBudget(budget);
      const total=budget.total_tokens,max=budget.context_tokens;
      contextLabel.textContent=Number.isFinite(total)?`${budget.exact?'':'~'}${total.toLocaleString()}${max?` / ${max.toLocaleString()} tokens (${Math.round(total/max*100)}%)`:' tokens'}`:'Context unavailable';
      contextLabel.title='Conversation context including the reply allowance. Open the ring for compaction.';
    };
    const scrim=node('button','paper-scrim');scrim.type='button';scrim.hidden=true;scrim.setAttribute('aria-label','Close panels');document.body.append(scrim);
    const panels=[$('#history-panel'),$('#settings-panel')];
    const closePanels=()=>{for(const panel of panels)panel.classList.remove('open');scrim.hidden=true;};
    scrim.addEventListener('click',closePanels);
    for(const panel of panels) {
      const close=node('button','paper-panel-close','×');close.type='button';close.setAttribute('aria-label',panel.id==='history-panel'?'Close history':'Close settings');close.addEventListener('click',()=>{panel.classList.remove('open');$(panel.id==='history-panel'?'#history-toggle':'#settings-toggle')?.focus();});panel.prepend(close);
      new MutationObserver(()=>{scrim.hidden=!panels.some(p=>p.classList.contains('open'));}).observe(panel,{attributes:true,attributeFilter:['class']});
    }
    $('#work-model').parentElement.hidden=true;
    $('#work-harness').parentElement.append(external('Connected agent tools may use external services'));
    const view=$('#work-view');
    const review=node('aside','paper-work-review');review.hidden=true;review.setAttribute('aria-label','Work review');
    const closeReview=node('button','paper-panel-close','×');closeReview.type='button';closeReview.setAttribute('aria-label','Close review');
    const reviewTabs=node('div','paper-review-tabs');reviewTabs.setAttribute('role','tablist');reviewTabs.setAttribute('aria-label','Work review');
    const changes=node('button','','Changes');changes.type='button';changes.setAttribute('role','tab');
    const recent=node('button','','Recent tasks');recent.type='button';recent.setAttribute('role','tab');
    const reviewBody=node('div','paper-review-body');reviewTabs.append(changes,recent);review.append(closeReview,reviewTabs,reviewBody);view.append(review);
    let reviewKind='changes'; let reviewSource=null; let recentSource=null;
    const reviewToggle=node('button','icon-button','☷');reviewToggle.type='button';reviewToggle.id='paper-review-toggle';reviewToggle.title='Review work';reviewToggle.setAttribute('aria-label','Review work');reviewToggle.hidden=true;actions?.insertBefore(reviewToggle,mobileTheme);
    const sessionsToggle=node('button','paper-labeled-control','Sessions');sessionsToggle.type='button';sessionsToggle.id='paper-sessions-toggle';sessionsToggle.title='Resume a saved Work session';sessionsToggle.setAttribute('aria-label','Work sessions');sessionsToggle.setAttribute('aria-expanded','false');sessionsToggle.hidden=true;actions?.insertBefore(sessionsToggle,reviewToggle);
    api.workSessions=()=>{if(!review.hidden&&reviewKind==='recent')drawReview();};
    document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!review.hidden){openReview(false);sessionsToggle.focus();}});
    api.closeWorkSessions=()=>{if(reviewKind==='recent')openReview(false);};
    sessionsToggle.addEventListener('click',()=>{reviewKind='recent';recentSource=$('#work-recent');drawReview();openReview(true);window.dispatchEvent(new Event('arc-paper-sessions-open'));});
    let sourceAnchor=null;
    function restoreSource() {
      if(sourceAnchor?.isConnected && reviewBody.firstElementChild) sourceAnchor.replaceWith(reviewBody.firstElementChild);
      sourceAnchor=null;
    }
    const openReview=open=>{if(!open)restoreSource();review.hidden=!open;view.classList.toggle('paper-review-open',open);reviewToggle.setAttribute('aria-expanded',String(open));sessionsToggle.setAttribute('aria-expanded',String(open&&reviewKind==='recent'));};
    function drawReview() {
      restoreSource();
      reviewBody.replaceChildren();
      changes.setAttribute('aria-selected',String(reviewKind==='changes')); recent.setAttribute('aria-selected',String(reviewKind==='recent'));
      const source=reviewKind==='changes'?reviewSource:recentSource;
      if(source && (reviewKind!=='recent'||$('#work-recent-list')?.childElementCount)) {source.hidden=false;if(source.parentElement && source.parentElement!==reviewBody){sourceAnchor=document.createComment('Paper review location');source.before(sourceAnchor);}reviewBody.append(source);} else reviewBody.append(node('p','paper-notice',reviewKind==='changes'?'Changes appear here when a task edits files.':'No saved Work sessions yet. Completed tasks are saved here automatically.'));
    }
    changes.addEventListener('click',()=>{reviewKind='changes';drawReview();});recent.addEventListener('click',()=>{reviewKind='recent';drawReview();});
    closeReview.addEventListener('click',()=>{openReview(false);reviewToggle.focus();});reviewToggle.addEventListener('click',()=>{drawReview();openReview(review.hidden);});
    api.workState=({pending,result,recentTasks})=>{
      $('.paper-nav-dot').hidden=!pending;
      reviewToggle.hidden=!document.body.classList.contains('work-mode');
      sessionsToggle.hidden=reviewToggle.hidden;
      if(result) reviewSource=result;
      if(recentTasks)recentSource=recentTasks.hidden?null:recentTasks;
      if(reviewSource&&!review.hidden&&reviewKind==='changes')drawReview();
      if(!document.body.classList.contains('work-mode'))openReview(false);
    };
    // Keep the working result in the conversation until Review is requested.
    api.openReview=(result)=>{if(result)reviewSource=result;reviewKind='changes';drawReview();openReview(true);};
    api.mode=()=>{reviewToggle.hidden=!document.body.classList.contains('work-mode');sessionsToggle.hidden=reviewToggle.hidden;if(reviewToggle.hidden)openReview(false);};
  }
  document.addEventListener('keydown',e=>{if(e.key==='Escape'){if(!menu.hidden){setMenu(false);pill.focus();}$('#paper-context-pop')?.setAttribute('hidden','');$('#paper-context-toggle')?.setAttribute('aria-expanded','false');document.body.classList.remove('paper-model-sheet');}});
})();
