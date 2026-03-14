// ─── I18N ──────────────────────────────────────────────────────────────
const TRANSLATIONS = {
  en: {
    home_headline:      'PREDICT<br>THE MATCH',
    quick_title:        'QUICK',
    quick_desc:         'Just rankings and H2H. Fast prediction in seconds.',
    quick_inputs:       'RANKING · TOURNAMENT · ROUND · H2H',
    expert_title:       'EXPERT',
    expert_desc:        'Full form data for maximum accuracy.',
    expert_inputs:      'FORM · STREAK · CAREER · OPP QUALITY',
    start_btn:          'START →',
    back_btn:           '← BACK',
    quick_headline:     'QUICK PREDICT',
    expert_headline:    'EXPERT PREDICT',
    match_context:      'Match Context',
    type_label:         'Type',
    tourney_level:      'Tournament Level',
    round_stage:        'Round Stage',
    home_ground:        'Home Ground',
    p1_home:            'P1 Home',
    p2_home:            'P2 Home',
    neutral:            'Neutral',
    player1:            'PLAYER 1',
    player2:            'PLAYER 2',
    name_opt:           'Name (optional)',
    bwf_rank:           'BWF Ranking',
    nationality:        'Nationality',
    wins_last5:         'Wins in last 5',
    wins_last10:        'Wins in last 10',
    wins_last20:        'Wins in last 20',
    streak:             'Current Streak (neg = losing)',
    career_matches:     'Career Matches',
    bwf_elo:            'ELO Rating *',
    elo_hint:           '(Player page → Match Details tab)',
    host_country:       'Host Country',
    match_month:        'Match Month (1–12)',
    h2h_optional_long:  'Head-to-Head (optional — leave blank if unknown)',
    h2h_optional:       'Head-to-Head (optional)',
    p1_wins:            'P1 Wins',
    total_matches:      'Total Matches',
    predict_btn:        'PREDICT →',
    result_title:       'PREDICTION RESULT',
    wins_word:          'WINS',
    ci_label:           'Model spread (P10–P90):',
    threeset_rate_label: '3-Set Rate %',
    set_count_title:    'SET COUNT PREDICTION',
    set_count_note:     'Reference only · Set count is hard to predict (AUC 0.66)',
    driving_factors:    'Driving Factors',
    again_btn:          '← PREDICT AGAIN',
    wins_20:            'wins 2-0',
    wins_21:            'wins 2-1',
    h2h_required:       'Head-to-Head',
    err_rank:           'Rankings, nationalities, ELO, tournament level, and round are required.',
    err_expert:         'All fields are required: rankings, nationalities, ELO, level, round, host country, form (5/10/20), H2H, streak, and career matches.',
    err_predict:        'Prediction failed.',
    badmintonranks_link: '<strong>Data sources:</strong> Rankings, form, H2H → <a href="https://badmintonranks.com" target="_blank" rel="noopener">BadmintonRanks.com</a>',
    footer_thanks: 'Special thanks to <a href="https://badmintonranks.com" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">BadmintonRanks.com</a> for generously providing the dataset that makes this project possible.',
    guide_lookup_title:  '🌐 LOOK UP DATA ON BadmintonRanks.com',
    guide_lookup_desc_quick: '<strong>All fields available on website:</strong><br>• Rankings → Player page → Profile tab<br>• Nationality → Player page → Profile tab<br>• ELO Rating → Player page → Match Details tab<br>• H2H → Player page → Head-to-Head tab<br><br><a href="https://badmintonranks.com/player?id=5012304" target="_blank" rel="noopener" style="color:#2563EB;text-decoration:underline;">Example: Player Page (An Se-young)</a>',
    guide_lookup_desc_expert: '<strong>All Quick fields PLUS:</strong><br>• Form (5/10/20) → Player page → Match Details tab<br>• Streak → Player page → Winning Streak tab<br>• Career Matches → Player page → Profile tab<br>• H2H required (same as Quick)<br><br><em style="color:#666">NOTE: 3-Set Rate is optional (not available on website)</em><br><br><a href="https://badmintonranks.com/player?id=5012304" target="_blank" rel="noopener" style="color:#2563EB;text-decoration:underline;">Example: Player Page (All Tabs)</a>',
    opt_og:             'Olympics',
    opt_wc:             'World Championships',
    opt_is:             'International Series',
    opt_ic:             'International Challenge',
    opt_qr1:            'Q-Round 1',
    opt_qr2:            'Q-Round 2',
    opt_qr3:            'Q-Round 3',
    opt_r1:             'Round 1',
    opt_r2:             'Round 2',
    opt_r3:             'Round 3',
    opt_qf:             'Quarter-final',
    opt_sf:             'Semi-final',
    opt_f:              'Final',
    ph_name1:           'e.g. Viktor Axelsen',
    ph_name2:           'e.g. Shi Yuqi',
  },
  zh: {
    home_headline:      '预测<br>比赛胜负',
    quick_title:        '快速',
    quick_desc:         '只需排名和交手记录，几秒出结果。',
    quick_inputs:       '排名 · 赛事 · 轮次 · 交手',
    expert_title:       '专家',
    expert_desc:        '从 BadmintonRanks.com 获取完整近况数据，预测更精准。',
    expert_inputs:      '近况 · 连胜 · 生涯 · 对手质量',
    start_btn:          '开始 →',
    back_btn:           '← 返回',
    quick_headline:     '快速预测',
    expert_headline:    '专家预测',
    match_context:      '比赛背景',
    type_label:         '类型',
    tourney_level:      '赛事级别',
    round_stage:        '轮次',
    home_ground:        '主场',
    p1_home:            'P1 主场',
    p2_home:            'P2 主场',
    neutral:            '中立',
    player1:            '球员 1',
    player2:            '球员 2',
    name_opt:           '姓名（可选）',
    bwf_rank:           'BWF 排名',
    nationality:        '国籍',
    wins_last5:         '近 5 场胜场',
    wins_last10:        '近 10 场胜场',
    wins_last20:        '近 20 场胜场',
    streak:             '当前连胜（负数=连败）',
    career_matches:     '生涯场次',
    bwf_elo:            'ELO 评分 *',
    elo_hint:           '（球员页面 → 比赛详情标签页）',
    host_country:       '举办国',
    match_month:        '比赛月份（1-12）',
    h2h_optional_long:  '交手记录（可选 — 不知道可留空）',
    h2h_optional:       '交手记录（可选）',
    p1_wins:            'P1 胜场',
    total_matches:      '总场次',
    predict_btn:        '预测 →',
    result_title:       '预测结果',
    wins_word:          '胜',
    ci_label:           '模型区间（P10–P90）：',
    threeset_rate_label: '三局比例 %',
    set_count_title:    '比分预测',
    set_count_note:     '仅供参考 · 大比分本质上难以预测（AUC 0.66）',
    driving_factors:    '主要驱动因素',
    again_btn:          '← 再预测一次',
    wins_20:            '以 2-0 胜',
    wins_21:            '以 2-1 胜',
    h2h_required:       '交手记录',
    err_rank:           '排名、国籍、ELO、赛事级别和轮次为必填项。',
    err_expert:         '所有字段均为必填：排名、国籍、ELO、赛事级别、轮次、举办国、近况（5/10/20场）、交手记录、连胜/败、生涯场次。',
    err_predict:        '预测失败，请重试。',
    badmintonranks_link: '<strong>数据来源：</strong>排名、近况、交手 → <a href="https://badmintonranks.com" target="_blank" rel="noopener">BadmintonRanks.com</a>',
    footer_thanks: '特别感谢 <a href="https://badmintonranks.com" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">BadmintonRanks.com</a> 的 owner 慷慨授权使用数据库，没有他们就没有这个项目。',
    guide_lookup_title:  '🌐 数据查询指南',
    guide_lookup_desc_quick: '<strong>所有数据均可从网站获取：</strong><br>• 排名 → 球员页面 → 档案资料标签页<br>• 国籍 → 球员页面 → 档案资料标签页<br>• ELO 评分 → 球员页面 → 比赛详情标签页<br>• 交手记录 → 球员页面 → 交手记录标签页<br><br><a href="https://badmintonranks.com/player?id=5012304" target="_blank" rel="noopener" style="color:#2563EB;text-decoration:underline;">示例：球员页面（安洗莹）</a>',
    guide_lookup_desc_expert: '<strong>快速模式的所有字段 PLUS：</strong><br>• 近况（5/10/20场） → 球员页面 → 比赛详情标签页<br>• 连胜/败 → 球员页面 → 连胜场次标签页<br>• 生涯场次 → 球员页面 → 档案资料标签页<br>• 交手记录必填（同快速模式）<br><br><em style="color:#666">注意：三局比例（3-Set Rate）为可选项（网站无此数据）</em><br><br><a href="https://badmintonranks.com/player?id=5012304" target="_blank" rel="noopener" style="color:#2563EB;text-decoration:underline;">示例：球员页面（全部标签页）</a>',
    opt_og:             '奥运会',
    opt_wc:             '世界锦标赛',
    opt_is:             '国际系列赛',
    opt_ic:             '国际挑战赛',
    opt_qr1:            '资格赛第1轮',
    opt_qr2:            '资格赛第2轮',
    opt_qr3:            '资格赛第3轮',
    opt_r1:             '第1轮',
    opt_r2:             '第2轮',
    opt_r3:             '第3轮',
    opt_qf:             '四分之一决赛',
    opt_sf:             '半决赛',
    opt_f:              '决赛',
    ph_name1:           '如 Viktor Axelsen',
    ph_name2:           '如 石宇奇',
  },
};

let _lang = localStorage.getItem('hb_lang') || 'en';

function t(key) {
  return TRANSLATIONS[_lang][key] || TRANSLATIONS['en'][key] || key;
}

function setLang(lang) {
  _lang = lang;
  localStorage.setItem('hb_lang', lang);

  // Update lang toggle buttons
  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.classList.toggle('active', btn.textContent.trim() === (lang === 'en' ? 'EN' : '中文'));
  });

  // Update all data-i18n elements
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.dataset.i18n;
    el.innerHTML = t(key);
  });

  // Update placeholders
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });

  // Update select option text
  document.querySelectorAll('[data-i18n-opt]').forEach(el => {
    el.textContent = t(el.dataset.i18nOpt);
  });
}

// ─── VIEW MANAGEMENT ──────────────────────────────────────────────────
let _prevView = 'home';

function showView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById('view-' + name).classList.add('active');
  window.scrollTo(0, 0);
}

// ─── TOGGLE BUTTONS ───────────────────────────────────────────────────
function initToggles() {
  document.querySelectorAll('.toggle-group').forEach(group => {
    group.querySelectorAll('.toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        group.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });
  });
}

function getToggleVal(groupId) {
  const active = document.querySelector('#' + groupId + ' .toggle-btn.active');
  return active ? active.dataset.val : '';
}

// ─── HELPERS ──────────────────────────────────────────────────────────
function intOrNull(id) {
  const v = document.getElementById(id).value.trim();
  return v === '' ? null : parseInt(v, 10);
}
function floatOrNull(id) {
  const v = document.getElementById(id).value.trim();
  return v === '' ? null : parseFloat(v);
}
function strOrEmpty(id) {
  return document.getElementById(id).value.trim();
}
function showError(id, msg) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.style.display = 'block';
}
function clearError(id) {
  document.getElementById(id).style.display = 'none';
}
function setLoading(spinId, btnId, loading) {
  document.getElementById(spinId).style.display = loading ? 'block' : 'none';
  document.getElementById(btnId).disabled = loading;
}

// ─── BUILD PAYLOAD (QUICK) ────────────────────────────────────────────
function buildQuickPayload() {
  const p1Rank = intOrNull('q-p1-rank');
  const p2Rank = intOrNull('q-p2-rank');
  const p1Nat  = document.getElementById('q-p1-nat').value.trim().toUpperCase();
  const p2Nat  = document.getElementById('q-p2-nat').value.trim().toUpperCase();
  const p1Elo  = floatOrNull('q-p1-elo');
  const p2Elo  = floatOrNull('q-p2-elo');
  const level  = document.getElementById('quick-level').value;
  const round  = document.getElementById('quick-round').value;
  if (!p1Rank || !p2Rank || !p1Nat || !p2Nat || !p1Elo || !p2Elo || !level || round === '') return null;

  const matchType = getToggleVal('quick-type-toggle') || 'MS';
  const country   = document.getElementById('quick-country').value.trim().toUpperCase();
  const month     = intOrNull('quick-month') ?? (new Date().getMonth() + 1);
  const p1Name    = strOrEmpty('q-p1-name') || 'Player 1';
  const p2Name    = strOrEmpty('q-p2-name') || 'Player 2';

  const h2hWins  = intOrNull('q-h2h-wins')  ?? 0;
  const h2hTotal = intOrNull('q-h2h-total') ?? 0;

  return {
    mode: 'quick',  // Use Quick mode model (21 features, AUC ~0.87)
    match_type: matchType,
    tournament_level: level,
    round_stage: parseInt(round, 10),
    match_month: month,
    host_country: country,
    player1: {
      name: p1Name,
      ranking: p1Rank,
      nationality: p1Nat,
      elo: p1Elo,
    },
    player2: {
      name: p2Name,
      ranking: p2Rank,
      nationality: p2Nat,
      elo: p2Elo,
    },
    h2h: { p1_wins: h2hWins, total: h2hTotal },
  };
}

// ─── BUILD PAYLOAD (EXPERT) ───────────────────────────────────────────
function buildExpertPayload() {
  const p1Rank = intOrNull('e-p1-rank');
  const p2Rank = intOrNull('e-p2-rank');
  const p1Nat  = strOrEmpty('e-p1-nat').toUpperCase();
  const p2Nat  = strOrEmpty('e-p2-nat').toUpperCase();
  const p1f5   = intOrNull('e-p1-f5');
  const p1f10  = intOrNull('e-p1-f10');
  const p1f20  = intOrNull('e-p1-f20');
  const p2f5   = intOrNull('e-p2-f5');
  const p2f10  = intOrNull('e-p2-f10');
  const p2f20  = intOrNull('e-p2-f20');
  const p1Elo  = floatOrNull('e-p1-elo');
  const p2Elo  = floatOrNull('e-p2-elo');
  const eLevel = document.getElementById('expert-level').value;
  const eRound = document.getElementById('expert-round').value;
  const country = strOrEmpty('expert-country');  // '' = placeholder (disabled), 'NEUTRAL' = no home

  // All Expert fields are required except 3-set rate (not available on BadmintonRanks.com)
  const h2hWins  = intOrNull('e-h2h-wins');
  const h2hTotal = intOrNull('e-h2h-total');
  const p1Streak = intOrNull('e-p1-streak');
  const p2Streak = intOrNull('e-p2-streak');
  const p1Career = intOrNull('e-p1-career');
  const p2Career = intOrNull('e-p2-career');
  const p1_3srate = intOrNull('e-p1-3srate') ?? 0;  // Optional - defaults to 0
  const p2_3srate = intOrNull('e-p2-3srate') ?? 0;  // Optional - defaults to 0

  if (!p1Rank || !p2Rank || !p1Nat || !p2Nat || !p1Elo || !p2Elo || !eLevel || eRound === '' ||
      !country ||
      p1f5 === null || p1f10 === null || p1f20 === null ||
      p2f5 === null || p2f10 === null || p2f20 === null ||
      h2hWins === null || h2hTotal === null ||
      p1Streak === null || p2Streak === null ||
      p1Career === null || p2Career === null) return null;  // 3-set rate NOT required

  const matchType = getToggleVal('expert-type-toggle') || 'MS';
  const month     = intOrNull('expert-month') ?? (new Date().getMonth() + 1);

  return {
    mode: 'expert',  // Use Expert mode model (35 features, AUC ~0.96)
    match_type: matchType,
    tournament_level: eLevel,
    round_stage: parseInt(eRound, 10),
    match_month: month,
    host_country: country === 'NEUTRAL' ? '' : country,
    player1: {
      name: strOrEmpty('e-p1-name') || 'Player 1',
      ranking: p1Rank,
      nationality: p1Nat,
      form5_wins:  p1f5,
      form10_wins: p1f10,
      form20_wins: p1f20,
      streak:      p1Streak,
      career_matches: p1Career,
      '3set_rate': p1_3srate / 100,
      elo: p1Elo,
    },
    player2: {
      name: strOrEmpty('e-p2-name') || 'Player 2',
      ranking: p2Rank,
      nationality: p2Nat,
      form5_wins:  p2f5,
      form10_wins: p2f10,
      form20_wins: p2f20,
      streak:      p2Streak,
      career_matches: p2Career,
      '3set_rate': p2_3srate / 100,
      elo: p2Elo,
    },
    h2h: { p1_wins: h2hWins, total: h2hTotal },
  };
}

// ─── API CALL ─────────────────────────────────────────────────────────
async function callPredict(payload) {
  const resp = await fetch('/api/predict-general', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${resp.status}`);
  }
  return resp.json();
}

// ─── SUBMIT HANDLERS ──────────────────────────────────────────────────
async function submitQuick() {
  clearError('quick-error');
  const payload = buildQuickPayload();
  if (!payload) { showError('quick-error', t('err_rank')); return; }

  setLoading('quick-spinner', 'quick-submit', true);
  try {
    const data = await callPredict(payload);
    _prevView = 'quick';
    renderResult(data);
    showView('result');
  } catch (e) {
    showError('quick-error', e.message || t('err_predict'));
  } finally {
    setLoading('quick-spinner', 'quick-submit', false);
  }
}

async function submitExpert() {
  clearError('expert-error');
  const payload = buildExpertPayload();
  if (!payload) { showError('expert-error', t('err_expert')); return; }

  setLoading('expert-spinner', 'expert-submit', true);
  try {
    const data = await callPredict(payload);
    _prevView = 'expert';
    renderResult(data);
    showView('result');
  } catch (e) {
    showError('expert-error', e.message || t('err_predict'));
  } finally {
    setLoading('expert-spinner', 'expert-submit', false);
  }
}

// ─── RENDER RESULT ────────────────────────────────────────────────────
function renderResult(data) {
  const prob    = data.player1_win_prob;
  const p2prob  = data.player2_win_prob;
  const ciLow   = data.ci_low  ?? prob;
  const ciHigh  = data.ci_high ?? prob;
  const p1Name  = data.player1_name || 'Player 1';
  const p2Name  = data.player2_name || 'Player 2';
  const winner  = data.predicted_winner || p1Name;

  // Winner text
  document.getElementById('result-winner-text').innerHTML =
    `${escHtml(winner)} <span>${t('wins_word')}</span>`;

  // P1 bar
  document.getElementById('res-p1-name').textContent = p1Name;
  document.getElementById('res-p1-pct').textContent  = pct(prob);
  document.getElementById('res-p1-bar').style.width  = pct(prob);

  // CI bracket
  document.getElementById('res-ci-bracket').style.left  = pct(ciLow);
  document.getElementById('res-ci-bracket').style.width = pct(ciHigh - ciLow);
  document.getElementById('res-ci-label').textContent   =
    `${t('ci_label')} ${pct(ciLow)} – ${pct(ciHigh)}`;

  // P2 bar
  document.getElementById('res-p2-name').textContent = p2Name;
  document.getElementById('res-p2-pct').textContent  = pct(p2prob);
  document.getElementById('res-p2-bar').style.width  = pct(p2prob);

  // Set count prediction
  const scenarios  = data.set_count_scenarios;
  const scPanel    = document.getElementById('set-count-panel');
  const scList     = document.getElementById('set-count-list');
  if (scenarios) {
    const rows = [
      { label: `${p1Name} ${t('wins_20')}`, prob: scenarios.p1_2_0, winner: true },
      { label: `${p1Name} ${t('wins_21')}`, prob: scenarios.p1_2_1, winner: true },
      { label: `${p2Name} ${t('wins_20')}`, prob: scenarios.p2_0_2, winner: false },
      { label: `${p2Name} ${t('wins_21')}`, prob: scenarios.p2_1_2, winner: false },
    ].sort((a, b) => b.prob - a.prob);
    scList.innerHTML = rows.map(r => `
      <div class="factor-row">
        <div class="factor-label" style="${r.winner ? '' : 'color:var(--text-muted)'}">${escHtml(r.label)}</div>
        <div class="factor-bar-wrap">
          <div class="factor-bar ${r.winner ? 'p1' : ''}" style="width:${Math.round(r.prob*100)}%;${r.winner ? '' : 'background:#ccc'}"></div>
        </div>
        <div class="factor-pct" style="color:${r.winner ? 'var(--text)' : 'var(--text-muted)'}">
          ${Math.round(r.prob * 100)}%
        </div>
      </div>`).join('') +
      `<div class="sc-note">${t('set_count_note')}</div>`;
    scPanel.style.display = 'block';
  } else {
    scPanel.style.display = 'none';
  }

  // Driving factors
  const factors = data.driving_factors || [];
  const panel   = document.getElementById('factors-panel');
  const list    = document.getElementById('factors-list');
  list.innerHTML = '';
  if (factors.length > 0) {
    panel.style.display = 'block';
    factors.forEach(f => {
      const pctVal = parseInt(f.delta_str.replace(/[^0-9]/g, ''), 10) || 0;
      const cls    = f.direction === 'p1' ? 'p1' : 'p2';
      const sign   = f.direction === 'p1' ? '+' : '−';
      list.innerHTML += `
        <div class="factor-row">
          <div class="factor-label">${escHtml(f.label)}</div>
          <div class="factor-bar-wrap">
            <div class="factor-bar ${cls}" style="width:${pctVal}%"></div>
          </div>
          <div class="factor-pct" style="color:${cls==='p1'?'var(--text)':'var(--red)'}">${sign}${pctVal}%</div>
        </div>`;
    });
  } else {
    panel.style.display = 'none';
  }
}

function pct(v) { return `${Math.round((v || 0) * 100)}%`; }
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ─── COUNTRY DATA ─────────────────────────────────────────────────────
// Sorted alphabetically by country name. Value = BWF code sent to API.
const COUNTRIES = [
  ['AUS','Australia'],['AUT','Austria'],['AZE','Azerbaijan'],
  ['BAN','Bangladesh'],['BEL','Belgium'],['BRA','Brazil'],['BUL','Bulgaria'],
  ['CMR','Cameroon'],['CAN','Canada'],['CHN','China'],['TPE','Chinese Taipei'],
  ['CRO','Croatia'],['CZE','Czechia'],
  ['DEN','Denmark'],
  ['EGY','Egypt'],['ENG','England'],['EST','Estonia'],
  ['FIN','Finland'],['FRA','France'],
  ['GER','Germany'],['GHA','Ghana'],['GRE','Greece'],
  ['HKG','Hong Kong'],['HUN','Hungary'],
  ['IND','India'],['INA','Indonesia'],['IRI','Iran'],['IRL','Ireland'],
  ['ISR','Israel'],['ITA','Italy'],
  ['JPN','Japan'],
  ['KAZ','Kazakhstan'],['KEN','Kenya'],['KOR','Korea'],['KGZ','Kyrgyzstan'],
  ['LAO','Lao'],['LAT','Latvia'],['LTU','Lithuania'],
  ['MAC','Macau'],['MAD','Madagascar'],['MAS','Malaysia'],
  ['MRI','Mauritius'],['MEX','Mexico'],['MDA','Moldova'],['MGL','Mongolia'],['MYA','Myanmar'],
  ['NEP','Nepal'],['NED','Netherlands'],['NZL','New Zealand'],['NGA','Nigeria'],['NOR','Norway'],
  ['PAK','Pakistan'],['PER','Peru'],['PHI','Philippines'],['POL','Poland'],['POR','Portugal'],
  ['ROU','Romania'],['RUS','Russia'],['RSA','South Africa'],['RWA','Rwanda'],
  ['SCO','Scotland'],['SRB','Serbia'],['SGP','Singapore'],
  ['SVK','Slovakia'],['SLO','Slovenia'],['ESP','Spain'],['SRI','Sri Lanka'],['SWE','Sweden'],['SUI','Switzerland'],
  ['THA','Thailand'],['TTO','Trinidad and Tobago'],['TUN','Tunisia'],['TUR','Turkey'],
  ['UGA','Uganda'],['UKR','Ukraine'],['USA','United States'],['UZB','Uzbekistan'],
  ['VIE','Vietnam'],
  ['WAL','Wales'],
  ['ZAM','Zambia'],['ZIM','Zimbabwe'],
];

function populateCountrySelects() {
  const natOpts = '<option value="" disabled selected>— Select —</option>' +
    COUNTRIES.map(([c, n]) => `<option value="${c}">${n} (${c})</option>`).join('');
  const hostOpts = '<option value="">— Neutral —</option>' +
    COUNTRIES.map(([c, n]) => `<option value="${c}">${n} (${c})</option>`).join('');
  // Expert host: user must explicitly choose (disabled placeholder; NEUTRAL = no home advantage)
  const hostOptsExpert = '<option value="" disabled selected>— Select —</option>' +
    '<option value="NEUTRAL">— Neutral (no home advantage) —</option>' +
    COUNTRIES.map(([c, n]) => `<option value="${c}">${n} (${c})</option>`).join('');
  ['q-p1-nat','q-p2-nat','e-p1-nat','e-p2-nat'].forEach(id => {
    document.getElementById(id).innerHTML = natOpts;
  });
  document.getElementById('quick-country').innerHTML = hostOpts;
  document.getElementById('expert-country').innerHTML = hostOptsExpert;
}

// ─── INIT ─────────────────────────────────────────────────────────────
document.getElementById('result-back-btn').addEventListener('click', () => showView(_prevView));
document.getElementById('again-btn').addEventListener('click', () => showView(_prevView));
populateCountrySelects();
initToggles();
document.getElementById('expert-month').value = new Date().getMonth() + 1;
document.getElementById('quick-month').value = new Date().getMonth() + 1;
setLang(_lang);  // apply saved language on load
