import React, { useEffect, useState } from 'react';
import {
  Activity,
  Clapperboard,
  Download,
  Film,
  FolderSearch,
  Home,
  Layers,
  MessageSquare,
  Search,
  Sparkles,
  TrendingUp,
  Users,
} from 'lucide-react';
import { apiGet, apiPost, apiUpload, workspaceFileUrl } from './api';

type WorkspaceTab =
  | 'home'
  | 'research'
  | 'symposium'
  | 'media'
  | 'editor'
  | 'animation'
  | 'content'
  | 'social'
  | 'growth'
  | 'export';

interface ProjectRecord {
  project_id: string;
  name: string;
  slug: string;
  creator_niche: string;
  editing_style: string;
  status: string;
  source_video_path?: string;
  updated_at: string;
}

interface AssetRecord {
  asset_id: string;
  role: string;
  relative_path: string;
  sha256: string;
  size_bytes: number;
  license_info: string;
}

interface ProjectManifest {
  project: ProjectRecord;
  assets: AssetRecord[];
  runs: any[];
  decisions: any[];
}

const NAV_ITEMS: { id: WorkspaceTab; label: string; icon: React.ReactNode }[] = [
  { id: 'home', label: 'Home Dashboard', icon: <Home size={16} /> },
  { id: 'research', label: 'Viral Research Lab', icon: <TrendingUp size={16} /> },
  { id: 'symposium', label: 'AI Symposium', icon: <Users size={16} /> },
  { id: 'media', label: 'Media Library', icon: <FolderSearch size={16} /> },
  { id: 'editor', label: 'Video Editor', icon: <Clapperboard size={16} /> },
  { id: 'animation', label: 'Animation Studio', icon: <Sparkles size={16} /> },
  { id: 'content', label: 'Content Studio', icon: <Film size={16} /> },
  { id: 'social', label: 'Social Studio', icon: <MessageSquare size={16} /> },
  { id: 'growth', label: 'Growth Analytics', icon: <Activity size={16} /> },
  { id: 'export', label: 'Export Center', icon: <Download size={16} /> },
];

export default function App() {
  const [tab, setTab] = useState<WorkspaceTab>('home');
  const [health, setHealth] = useState<any>(null);
  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [activePid, setActivePid] = useState<string>('');
  const [manifest, setManifest] = useState<ProjectManifest | null>(null);
  const [analysis, setAnalysis] = useState<any>(null);
  const [symposium, setSymposium] = useState<any>(null);
  const [timeline, setTimeline] = useState<any>(null);
  const [socialBundle, setSocialBundle] = useState<any>(null);
  const [viralItems, setViralItems] = useState<any[]>([]);
  const [researchQuery, setResearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('');
  const [effects, setEffects] = useState<any[]>([]);
  const [growthData, setGrowthData] = useState<any>(null);
  const [statusMsg, setStatusMsg] = useState('');
  const [busy, setBusy] = useState(false);

  // Forms
  const [newProjName, setNewProjName] = useState('');
  const [newProjNiche, setNewProjNiche] = useState('science_engineering');
  const [newProjStyle, setNewProjStyle] = useState('educational_explainer');
  const [instructText, setInstructText] = useState('');
  const [animTemplate, setAnimTemplate] = useState('em_induction_faraday');
  const [animParamsJson, setAnimParamsJson] = useState('{}');
  const [inspectUrl, setInspectUrl] = useState('');
  const [refStylePath, setRefStylePath] = useState('');
  const [tgChatId, setTgChatId] = useState('@symposium_lab');
  const [showSafeZone, setShowSafeZone] = useState(true);
  const [threadsStyle, setThreadsStyle] = useState('educational');
  const [searchTerm, setSearchTerm] = useState('');
  const [searchHits, setSearchHits] = useState<any[]>([]);
  const [expQuery, setExpQuery] = useState('');
  const [expPlatformFilter, setExpPlatformFilter] = useState('');
  const [expSearchResults, setExpSearchResults] = useState<any[] | null>(null);

  async function refreshBase() {
    try {
      const [h, projs, vitems, fx] = await Promise.all([
        apiGet<any>('/api/health'),
        apiGet<ProjectRecord[]>('/api/projects'),
        apiGet<any[]>('/api/research/items'),
        apiGet<any[]>('/api/effects'),
      ]);
      setHealth(h);
      setProjects(projs);
      setViralItems(vitems);
      setEffects(fx);
      if (!activePid && projs.length > 0) {
        setActivePid(projs[0].project_id);
      }
    } catch (e: any) {
      setStatusMsg(`Connection error: ${e.message}`);
    }
  }

  async function loadProjectDetails(pid: string) {
    if (!pid) return;
    try {
      const man = await apiGet<ProjectManifest>(`/api/projects/${pid}`);
      setManifest(man);
      try {
        setAnalysis(await apiGet(`/api/projects/${pid}/analysis`));
      } catch {
        setAnalysis(null);
      }
      try {
        const symResp = await apiGet<any>(`/api/projects/${pid}/symposium`);
        setSymposium(symResp.latest_session);
      } catch {
        setSymposium(null);
      }
      const tlAsset = man.assets.find((a) => a.role === 'timeline_json' && a.relative_path.includes('youtube'));
      if (tlAsset) {
        setTimeline(await apiGet(workspaceFileUrl(pid, tlAsset.relative_path)));
      } else {
        setTimeline(null);
      }
      const socAsset = man.assets.find((a) => a.role === 'social_package');
      if (socAsset) {
        setSocialBundle(await apiGet(workspaceFileUrl(pid, socAsset.relative_path)));
      } else {
        setSocialBundle(null);
      }
      setGrowthData(await apiGet(`/api/growth?project_id=${encodeURIComponent(pid)}`));
    } catch (e: any) {
      setStatusMsg(e.message);
    }
  }

  useEffect(() => {
    refreshBase();
  }, []);

  useEffect(() => {
    setExpSearchResults(null);
    if (activePid) loadProjectDetails(activePid);
  }, [activePid]);

  async function handleCreateProject(e: React.FormEvent) {
    e.preventDefault();
    if (!newProjName.trim()) return;
    setBusy(true);
    try {
      const created = await apiPost<ProjectManifest>('/api/projects', {
        name: newProjName,
        creator_niche: newProjNiche,
        editing_style: newProjStyle,
      });
      setNewProjName('');
      await refreshBase();
      setActivePid(created.project.project_id);
      setStatusMsg(`Created workspace ${created.project.slug}`);
    } finally {
      setBusy(false);
    }
  }

  async function handleSearchResearch() {
    const params = new URLSearchParams();
    if (researchQuery) params.set('q', researchQuery);
    if (categoryFilter) params.set('category', categoryFilter);
    setViralItems(await apiGet(`/api/research/items?${params.toString()}`));
  }

  async function handleSearchExperiments() {
    const params = new URLSearchParams();
    if (expQuery.trim()) params.set('q', expQuery.trim());
    if (expPlatformFilter) params.set('platform', expPlatformFilter);
    setExpSearchResults(await apiGet(`/api/growth/experiments?${params.toString()}`));
  }

  async function handleDiscoverLive() {
    setBusy(true);
    setStatusMsg('Discovering live multi-niche creator feeds (~100 videos)...');
    try {
      const res = await apiPost<any>('/api/research/discover', { target_count: 100 });
      await handleSearchResearch();
      setStatusMsg(`Discovered ${res.discovered_count} real public videos across ${res.categories_covered?.length || 0} categories`);
    } catch (e: any) {
      setStatusMsg(`Discovery error: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function handleInspectUrl() {
    if (!inspectUrl.trim()) return;
    setBusy(true);
    try {
      await apiPost('/api/research/inspect-url', { url: inspectUrl.trim() });
      setInspectUrl('');
      await handleSearchResearch();
      setStatusMsg('Inspected public YouTube URL via oEmbed');
    } catch (e: any) {
      setStatusMsg(`Inspect error: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function runAction(label: string, fn: () => Promise<void>) {
    setBusy(true);
    setStatusMsg(`Running: ${label}...`);
    try {
      await fn();
      if (activePid) await loadProjectDetails(activePid);
      setStatusMsg(`Completed: ${label}`);
    } catch (e: any) {
      setStatusMsg(`Error (${label}): ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  const ytAsset = manifest?.assets.find((a) => a.role === 'render_youtube');
  const thumbAsset = manifest?.assets.find((a) => a.role === 'thumbnail');
  const animAsset = [...(manifest?.assets || [])].reverse().find((a) => a.role === 'animation_render');
  const shortAssets = (manifest?.assets || []).filter((a) =>
    ['render_short', 'render_tiktok', 'render_reel'].includes(a.role)
  );

  return (
    <div className="studio-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-dot" />
          <div>
            <h1>Symposium Studio</h1>
            <small>Open-Source AI Production</small>
          </div>
        </div>
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            className={`nav-btn ${tab === item.id ? 'active' : ''}`}
            onClick={() => setTab(item.id)}
          >
            {item.icon}
            <span>{item.label}</span>
          </button>
        ))}
        <div style={{ marginTop: 'auto', paddingTop: 12, borderTop: '1px solid var(--border)' }}>
          <div className="muted" style={{ fontSize: 11, marginBottom: 6 }}>Active Project</div>
          <select value={activePid} onChange={(e) => setActivePid(e.target.value)}>
            <option value="">-- Select Project --</option>
            {projects.map((p) => (
              <option key={p.project_id} value={p.project_id}>
                {p.name} ({p.status})
              </option>
            ))}
          </select>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div>
            <strong>{NAV_ITEMS.find((n) => n.id === tab)?.label}</strong>
            {manifest && (
              <span className="muted" style={{ marginLeft: 12 }}>
                {manifest.project.name} · <span className="pill">{manifest.project.status}</span>
              </span>
            )}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {statusMsg && <span className="pill pill-amber">{statusMsg}</span>}
            {activePid && (
              <>
                <button className="btn" disabled={busy} onClick={() => runAction('Analyze Footage', () => apiPost(`/api/projects/${activePid}/analyze`, {}))}>
                  Analyze
                </button>
                <button className="btn" disabled={busy} onClick={() => runAction('Run AI Symposium', () => apiPost(`/api/projects/${activePid}/symposium`, { mode: 'standard' }))}>
                  Symposium
                </button>
                <button className="btn btn-primary" disabled={busy} onClick={() => runAction('Render Deliverables', () => apiPost(`/api/projects/${activePid}/render`))}>
                  Render All
                </button>
              </>
            )}
          </div>
        </header>

        <div className="content-scroll">
          {tab === 'home' && (
            <div className="grid-2">
              <div className="card">
                <h2>Create New Studio Project</h2>
                <form onSubmit={handleCreateProject} style={{ display: 'grid', gap: 10 }}>
                  <input placeholder="Project title (e.g., Electromagnetic Induction Lab)" value={newProjName} onChange={(e) => setNewProjName(e.target.value)} />
                  <div className="grid-2">
                    <input placeholder="Creator niche" value={newProjNiche} onChange={(e) => setNewProjNiche(e.target.value)} />
                    <select value={newProjStyle} onChange={(e) => setNewProjStyle(e.target.value)}>
                      <option value="educational_explainer">Educational Explainer</option>
                      <option value="high_energy_youtube">High-Energy YouTube</option>
                      <option value="cinematic_storytelling">Cinematic Storytelling</option>
                      <option value="documentary">Documentary</option>
                      <option value="technical_tutorial">Technical Tutorial</option>
                      <option value="short_form_viral">Short-Form Viral</option>
                      <option value="comedy">Comedy</option>
                      <option value="personal_vlog">Personal Vlog</option>
                      <option value="zoomer_vlog">Zoomer / Gen-Z Vlog (Live Memes + Jump Cuts)</option>
                    </select>
                  </div>
                  <button className="btn btn-primary" type="submit" disabled={busy}>Create Workspace</button>
                </form>
              </div>

              <div className="card">
                <h2>Engine Health & Capabilities</h2>
                {health ? (
                  <div style={{ display: 'grid', gap: 8 }}>
                    <div className="muted">
                      Symposium Agents: <strong>{health.agents_count}</strong> · LLM: <code>{health.llm_backend || 'deterministic_analytical'}</code> · Workspaces: <code>{health.workspaces_root}</code>
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {health.adapters.map((a: any) => (
                        <span key={a.adapter_id} className={`pill ${a.available ? 'pill-green' : ''}`}>
                          {a.display_name} ({a.available ? 'ready' : 'optional'})
                        </span>
                      ))}
                    </div>
                  </div>
                ) : <p className="muted">Loading backend health...</p>}
              </div>

              <div className="card" style={{ gridColumn: '1 / -1' }}>
                <h2>Projects ({projects.length})</h2>
                <table>
                  <thead>
                    <tr><th>Project</th><th>Niche</th><th>Style</th><th>Status</th><th>Updated</th></tr>
                  </thead>
                  <tbody>
                    {projects.map((p) => (
                      <tr key={p.project_id} onClick={() => setActivePid(p.project_id)} style={{ cursor: 'pointer' }}>
                        <td><strong>{p.name}</strong><br /><small className="muted">{p.project_id}</small></td>
                        <td>{p.creator_niche}</td>
                        <td>{p.editing_style}</td>
                        <td><span className="pill">{p.status}</span></td>
                        <td className="muted">{p.updated_at.slice(0, 19)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {tab === 'research' && (
            <div style={{ display: 'grid', gap: 16 }}>
              <div className="card" style={{ display: 'grid', gap: 10 }}>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                  <input placeholder="Search viral mechanics across all categories (gaming, comedy, tech, documentary, threads)..." value={researchQuery} onChange={(e) => setResearchQuery(e.target.value)} />
                  <select style={{ maxWidth: 220 }} value={categoryFilter} onChange={(e) => setCategoryFilter(e.target.value)}>
                    <option value="">All Categories</option>
                    {['entertainment','gaming','comedy','science','technology','engineering','programming','education','business','startups','sports','lifestyle','personal_storytelling','documentaries','popular_culture','internet_trends'].map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                  <button className="btn" onClick={handleSearchResearch}><Search size={14} /> Search</button>
                  <button className="btn btn-primary" disabled={busy} onClick={handleDiscoverLive}>Discover Live (~100)</button>
                </div>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                  <input placeholder="Inspect explicit YouTube URL via oEmbed (https://www.youtube.com/watch?v=...)" value={inspectUrl} onChange={(e) => setInspectUrl(e.target.value)} />
                  <button className="btn" disabled={busy || !inspectUrl.trim()} onClick={handleInspectUrl}>Inspect URL</button>
                </div>
              </div>
              <div className="grid-2">
                {viralItems.map((it) => (
                  <div className="card" key={it.item_id}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                      <span className="pill">{it.platform} · {it.creator}</span>
                      <span className={`pill ${it.metrics?.views ? 'pill-green' : 'pill-amber'}`}>
                        {it.category} · {it.metrics?.views ? `score ${it.platform_relative_score} (${it.metrics.audience_normalized_ratio ?? '-'}x base)` : 'unscored archetype'}
                      </span>
                    </div>
                    <h3>{it.title}</h3>
                    {it.metrics?.views && (
                      <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
                        Views: {Number(it.metrics.views).toLocaleString()} · Channel median: {it.metrics.channel_baseline_views ? Number(it.metrics.channel_baseline_views).toLocaleString() : 'n/a'} · {it.metrics.metric_source}
                      </div>
                    )}
                    <p className="muted"><strong>Hook:</strong> {it.opening_hook_structure}</p>
                    <p className="muted"><strong>Editing:</strong> {it.editing_style}</p>
                    {it.reusable_techniques?.map((t: any) => (
                      <div key={t.technique_id} style={{ background: '#0a101d', padding: 8, borderRadius: 8, marginTop: 6, fontSize: 12 }}>
                        <strong>Transferable Mechanic: {t.name}</strong>
                        <div className="muted">{t.editing_translation}</div>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}

          {tab === 'symposium' && (
            <div style={{ display: 'grid', gap: 16 }}>
              {symposium ? (
                <>
                  <div className="card">
                    <h2>Selected Strategy ({symposium.mode} · {symposium.participating_agents?.length} agents · backend: {symposium.backend_used})</h2>
                    <p><strong>{symposium.storyboard?.selected_strategy}</strong> — Target Duration: {symposium.storyboard?.target_duration_sec}s · LLM calls: {symposium.llm_calls_count || 0}</p>
                    {symposium.decisions?.map((d: any) => (
                      <div key={d.decision_id} style={{ padding: '8px 0', borderTop: '1px solid var(--border)' }}>
                        <strong>{d.topic}:</strong> {d.chosen_option}
                        <div className="muted">{d.rationale} (Evidence: {d.evidence_refs?.join(', ')})</div>
                      </div>
                    ))}
                    {symposium.challenges?.length > 0 && (
                      <div style={{ marginTop: 10 }}>
                        <h3>Moderator Audit & Conflict Resolutions ({symposium.challenges.length})</h3>
                        {symposium.challenges.map((c: any) => (
                          <div key={c.challenge_id} style={{ padding: '6px 0', borderTop: '1px solid var(--border)', fontSize: 12.5 }}>
                            <span className="pill pill-rose">{c.issue_type}</span> {c.description}
                            <div className="muted"><strong>Resolution:</strong> {c.resolution}</div>
                          </div>
                        ))}
                      </div>
                    )}
                    {symposium.revision_history?.length > 0 && (
                      <div style={{ marginTop: 10 }}>
                        <h4>Revision Log</h4>
                        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12.5 }}>
                          {symposium.revision_history.map((rev: string, i: number) => <li key={i}>{rev}</li>)}
                        </ul>
                      </div>
                    )}
                  </div>

                  {(symposium.growth_objective || symposium.creative_director_review) && (
                    <div className="card">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                        <h2 style={{ margin: 0 }}>Growth Objective, Relevance & Creative Director Gate</h2>
                        {symposium.creative_director_review && (
                          <span className={`pill ${symposium.creative_director_review.approved ? 'pill-green' : 'pill-rose'}`}>
                            {symposium.creative_director_review.approved ? 'CD Gate Approved' : 'Revision Required'} · Rank #{symposium.creative_director_review.scorecard_rank}
                          </span>
                        )}
                      </div>
                      {symposium.relevance_assessment && (
                        <div style={{ marginBottom: 12, paddingBottom: 10, borderBottom: '1px solid var(--border)' }}>
                          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center', marginBottom: 6 }}>
                            <span className="pill">Timeliness: {symposium.relevance_assessment.topic_timeliness}</span>
                            <span className="pill pill-amber">Momentum: {Number(symposium.relevance_assessment.momentum_score).toFixed(2)}</span>
                          </div>
                          <div><strong>Why Viewers Care Today:</strong> <span className="muted">{symposium.relevance_assessment.why_viewer_cares_today}</span></div>
                          <div><strong>Long-Term Relevance:</strong> <span className="muted">{symposium.relevance_assessment.long_term_relevance_note}</span></div>
                          {(symposium.relevance_assessment.limitations || []).length > 0 && (
                            <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
                              Note: {symposium.relevance_assessment.limitations.join(' ')}
                            </div>
                          )}
                        </div>
                      )}
                      {symposium.growth_objective && (
                        <div className="grid-2" style={{ gap: 12 }}>
                          <div>
                            <div style={{ marginBottom: 6 }}>
                              <strong>Target Core + Accessible Audience:</strong>
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 4 }}>
                                {(symposium.growth_objective.target_audience || []).map((aud: string) => (
                                  <span className="pill" key={aud}>{aud}</span>
                                ))}
                              </div>
                            </div>
                            <p style={{ margin: '6px 0' }}><strong>Accessible Entry Hook:</strong> <span className="muted">{symposium.growth_objective.accessible_entry_hook}</span></p>
                            <p style={{ margin: '6px 0' }}><strong>Content Hypothesis:</strong> <span className="muted">{symposium.growth_objective.content_hypothesis}</span></p>
                            <p style={{ margin: '6px 0' }}><strong>Expected Viewer Value:</strong> <span className="muted">{symposium.growth_objective.expected_viewer_value}</span></p>
                          </div>
                          <div>
                            <p style={{ margin: '0 0 6px' }}><strong>Selected Hook Strategy:</strong> <span className="muted">{symposium.growth_objective.selected_hook_strategy}</span></p>
                            {(symposium.growth_objective.retention_risk_analysis || []).length > 0 && (
                              <div style={{ marginBottom: 8 }}>
                                <strong>Retention Risk Analysis:</strong>
                                <ul style={{ margin: '4px 0 0', paddingLeft: 18, fontSize: 12.5 }}>
                                  {symposium.growth_objective.retention_risk_analysis.map((risk: string, i: number) => (
                                    <li className="muted" key={i}>{risk}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                            {(symposium.prior_learnings_applied || []).length > 0 && (
                              <div>
                                <strong>Prior Experiment Learnings Applied:</strong>
                                <ul style={{ margin: '4px 0 0', paddingLeft: 18, fontSize: 12.5 }}>
                                  {symposium.prior_learnings_applied.map((item: string, i: number) => (
                                    <li className="muted" key={i}>{item}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                      {symposium.creative_director_review && (
                        <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid var(--border)' }}>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 4 }}>
                            {(symposium.creative_director_review.checked_criteria || []).map((c: string) => (
                              <span className="pill pill-green" key={c}>{c}</span>
                            ))}
                          </div>
                          {symposium.creative_director_review.revision_notes && (
                            <div className="muted" style={{ fontSize: 12 }}>{symposium.creative_director_review.revision_notes}</div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {(symposium.candidate_evaluations || []).length > 0 && (
                    <div className="card">
                      <h2>Candidate Edit Performance Predictions (11-Factor Ranking)</h2>
                      <p className="muted" style={{ marginTop: 0 }}>
                        Uncertainty-bounded editorial ranking (0.00-1.00). Never fabricates view counts or guarantees virality.
                      </p>
                      <table>
                        <thead>
                          <tr>
                            <th>Rank / Candidate</th>
                            <th>Composite & CI</th>
                            <th>11-Factor Scorecard</th>
                            <th>Evidence Rationale</th>
                          </tr>
                        </thead>
                        <tbody>
                          {symposium.candidate_evaluations.map((cand: any, idx: number) => {
                            const sc = cand.scorecard || {};
                            const ci = cand.confidence_interval || {};
                            return (
                              <tr key={cand.candidate_id}>
                                <td>
                                  <div>
                                    <strong>#{idx + 1} {cand.strategy_name}</strong>{' '}
                                    {cand.selected && <span className="pill pill-green">Selected</span>}
                                  </div>
                                  <div className="muted">{cand.editing_style} · {cand.target_duration_sec}s</div>
                                </td>
                                <td>
                                  <div><strong>{Number(cand.composite_score).toFixed(3)}</strong></div>
                                  <div className="muted">CI [{Number(ci.low ?? 0).toFixed(2)}, {Number(ci.high ?? 0).toFixed(2)}]</div>
                                  <div className="muted">±{Number(ci.uncertainty ?? 0).toFixed(2)}</div>
                                  {Number(cand.historical_lift_applied || 0) !== 0 && (
                                    <div className="pill pill-amber" style={{ marginTop: 4 }}>
                                      Hist lift {cand.historical_lift_applied > 0 ? '+' : ''}{Number(cand.historical_lift_applied).toFixed(3)}
                                    </div>
                                  )}
                                </td>
                                <td>
                                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                                    <span className="pill">Hook {Number(sc.hook_strength ?? 0).toFixed(2)}</span>
                                    <span className="pill">Rel {Number(sc.topic_relevance ?? 0).toFixed(2)}</span>
                                    <span className="pill">Clarity {Number(sc.clarity ?? 0).toFixed(2)}</span>
                                    <span className="pill">Info {Number(sc.information_density ?? 0).toFixed(2)}</span>
                                    <span className="pill">Curiosity {Number(sc.curiosity ?? 0).toFixed(2)}</span>
                                    <span className="pill">Emotion {Number(sc.emotional_engagement ?? 0).toFixed(2)}</span>
                                    <span className="pill">Story {Number(sc.narrative_progression ?? 0).toFixed(2)}</span>
                                    <span className="pill">Visual {Number(sc.visual_quality ?? 0).toFixed(2)}</span>
                                    <span className="pill">Audience {Number(sc.audience_compatibility ?? 0).toFixed(2)}</span>
                                    <span className="pill">Share {Number(sc.shareability ?? 0).toFixed(2)}</span>
                                    <span className="pill">Evergreen {Number(sc.estimated_long_term_relevance ?? 0).toFixed(2)}</span>
                                  </div>
                                </td>
                                <td>
                                  <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12 }}>
                                    {(cand.evidence_rationale || []).map((ev: string, i: number) => <li key={i}>{ev}</li>)}
                                  </ul>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                  <div className="grid-2">
                    {symposium.proposals?.map((p: any) => (
                      <div className="card" key={p.proposal_id}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <strong>{p.agent_title}</strong>
                          <span className="pill">{p.team} · conf {p.confidence}</span>
                        </div>
                        <p className="muted">{p.summary}</p>
                        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12.5 }}>
                          {p.recommendations?.map((r: string, i: number) => <li key={i}>{r}</li>)}
                        </ul>
                      </div>
                    ))}
                  </div>
                </>
              ) : <div className="card"><p className="muted">Run AI Symposium on the active project to view multi-agent debate and storyboard beats.</p></div>}
            </div>
          )}

          {tab === 'media' && (
            <div style={{ display: 'grid', gap: 16 }}>
              <div className="card" style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                <div style={{ flex: 1, minWidth: 240 }}>
                  <strong>Upload Video to Active Project</strong>
                  <div className="muted" style={{ fontSize: 12 }}>
                    Uploads raw footage directly into <code>{activePid || 'selected project'}</code> &amp; probes media streams.
                  </div>
                </div>
                <input
                  type="file"
                  accept="video/*,audio/*,.mov,.mp4,.mkv"
                  disabled={busy || !activePid}
                  onChange={async (e) => {
                    const f = e.target.files?.[0];
                    if (!f || !activePid) return;
                    setBusy(true);
                    setStatusMsg(`Uploading ${f.name}...`);
                    try {
                      await apiUpload(`/api/projects/${activePid}/upload`, f);
                      await refreshBase();
                      await loadProjectDetails(activePid);
                      setStatusMsg(`Uploaded ${f.name}. Ready to Analyze -> Render!`);
                    } catch (err: any) {
                      setStatusMsg(`Upload error: ${err.message}`);
                    } finally {
                      setBusy(false);
                      e.target.value = '';
                    }
                  }}
                />
              </div>
              <div className="card" style={{ display: 'flex', gap: 10 }}>
                <input placeholder="Full-text search transcripts, assets, and research..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} />
                <button className="btn btn-primary" onClick={async () => setSearchHits(await apiGet(`/api/search?q=${encodeURIComponent(searchTerm)}&project_id=${encodeURIComponent(activePid)}`))}>
                  FTS Search
                </button>
              </div>
              {searchHits.length > 0 && (
                <div className="card">
                  <h3>Search Matches ({searchHits.length})</h3>
                  {searchHits.map((h, i) => (
                    <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid var(--border)' }}>
                      <span className="pill">{h.entity_type}</span> <strong style={{ marginLeft: 6 }}>{h.title}</strong>
                      <div className="muted">{h.snippet}</div>
                    </div>
                  ))}
                </div>
              )}
              {analysis ? (
                <div className="grid-2">
                  <div className="card">
                    <h3>Transcript ({analysis.transcript.detected_language} via {analysis.transcript.engine})</h3>
                    {analysis.transcript.segments.map((s: any) => (
                      <div key={s.segment_id} style={{ padding: '6px 0', borderBottom: '1px solid var(--border)', fontSize: 13 }}>
                        <span className="pill">{s.start_sec.toFixed(1)}s–{s.end_sec.toFixed(1)}s</span>{' '}
                        <span className="pill pill-amber">{s.speaker_id}</span>
                        {s.low_confidence && <span className="pill pill-rose">uncertain</span>}
                        <div style={{ marginTop: 4 }}>{s.text}</div>
                      </div>
                    ))}
                  </div>
                  <div className="card">
                    <h3>Scored Highlights ({analysis.highlights.length})</h3>
                    {analysis.highlights.map((h: any) => (
                      <div key={h.highlight_id} style={{ padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <strong>{h.role} ({h.start_sec.toFixed(1)}s–{h.end_sec.toFixed(1)}s)</strong>
                          <span className="pill pill-green">score {h.scores.composite_score}</span>
                        </div>
                        <p className="muted" style={{ margin: '4px 0' }}>{h.transcript_excerpt}</p>
                        {h.visual_opportunity_type && <span className="pill pill-amber">Animation: {h.visual_opportunity_type}</span>}
                      </div>
                    ))}
                  </div>
                </div>
              ) : <div className="card"><p className="muted">Analyze a project's source footage to inspect transcript and scored highlights.</p></div>}
            </div>
          )}

          {tab === 'editor' && (
            <div style={{ display: 'grid', gap: 16 }}>
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                  <h2>Video Preview & Timeline</h2>
                  {timeline && <span className="pill">Rev {timeline.revision} · {timeline.width}x{timeline.height} · {timeline.style_profile}</span>}
                </div>
                <div className="video-stage">
                  {ytAsset ? (
                    <video key={ytAsset.sha256} controls src={workspaceFileUrl(activePid, ytAsset.relative_path)} />
                  ) : (
                    <div style={{ padding: 60 }} className="muted">No rendered YouTube video yet. Click "Render All" above.</div>
                  )}
                </div>
                <div style={{ display: 'flex', gap: 10, marginTop: 12 }}>
                  <input
                    placeholder='Iterative AI instruction (e.g., "Make captions smaller", "Speed up 1.12x", "Cross dissolve", "Turn into 3 shorts")'
                    value={instructText}
                    onChange={(e) => setInstructText(e.target.value)}
                  />
                  <button
                    className="btn btn-primary"
                    disabled={busy || !instructText.trim()}
                    onClick={() => runAction('Apply Agent Instruction', () => apiPost(`/api/projects/${activePid}/instruct`, { instruction: instructText }))}
                  >
                    Apply Instruction
                  </button>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 10 }}>
                  {['Make captions smaller', 'Speed up 1.12x', 'Cross dissolve', 'Mute music', 'Download click sounds from internet and add them', 'Turn this into three shorts'].map((preset) => (
                    <button key={preset} className="btn" disabled={busy || !activePid} onClick={() => setInstructText(preset)}>{preset}</button>
                  ))}
                  <button
                    className="btn"
                    disabled={busy || !activePid}
                    onClick={() => runAction('Download & Add Click SFX', () => apiPost(`/api/projects/${activePid}/sfx/clicks`, {}))}
                  >
                    + Click SFX (Internet)
                  </button>
                </div>
                <div style={{ display: 'flex', gap: 10, marginTop: 10 }}>
                  <input
                    placeholder="Clone editing style from reference video path..."
                    value={refStylePath}
                    onChange={(e) => setRefStylePath(e.target.value)}
                  />
                  <button
                    className="btn"
                    disabled={busy || !activePid || !refStylePath.trim()}
                    onClick={() => runAction('Clone Reference Style', () => apiPost(`/api/projects/${activePid}/style-from-reference`, { reference_video_path: refStylePath.trim() }))}
                  >
                    Clone Reference Style
                  </button>
                </div>
              </div>

              {timeline && (
                <div className="card">
                  <h3>Editable Render Graph Tracks</h3>
                  <div className="timeline-track">
                    <div className="muted"><Layers size={13} /> V1 Main Footage ({timeline.clips.length} clips)</div>
                    <div className="clip-row">
                      {timeline.clips.map((c: any) => (
                        <div className="clip-block" key={c.clip_id}>
                          <strong>{c.clip_id}</strong> ({(c.timeline_end_sec - c.timeline_start_sec).toFixed(1)}s)
                          <div className="muted" style={{ fontSize: 11 }}>{c.label}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="timeline-track">
                    <div className="muted">V2 Generated Animations ({timeline.overlays.length})</div>
                    <div className="clip-row">
                      {timeline.overlays.map((o: any) => (
                        <div className="overlay-block" key={o.overlay_id}>
                          <strong>{o.label}</strong>
                          <div className="muted" style={{ fontSize: 11 }}>{o.timeline_start_sec.toFixed(1)}s–{o.timeline_end_sec.toFixed(1)}s ({o.mode})</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {tab === 'animation' && (
            <div className="grid-2">
              <div className="card">
                <h2>Original Animation Generator</h2>
                <div style={{ display: 'grid', gap: 10 }}>
                  <select value={animTemplate} onChange={(e) => setAnimTemplate(e.target.value)}>
                    <option value="em_induction_faraday">Physics: Electromagnetic Induction (Faraday's Law)</option>
                    <option value="wave_interference_sim">Physics: Wave Interference & Beat Superposition</option>
                    <option value="rlc_signal_flow">Electrical Eng: RLC Circuit & Waveform</option>
                    <option value="binary_tree_trace">Programming: Binary Search Tree Trace</option>
                    <option value="system_architecture_flow">Systems: Distributed Ingestion Pipeline Flow</option>
                    <option value="narrative_timeline">Storytelling: Engineering Milestone Timeline</option>
                    <option value="kinetic_callout">Editorial: Kinetic Key Takeaway Callout</option>
                    <option value="meme_reaction_card">Zoomer: Trending Meme Reaction Card</option>
                  </select>
                  <textarea
                    rows={3}
                    placeholder='Editable SceneSpec parameters JSON (e.g. {"freq_hz": 2.2})'
                    value={animParamsJson}
                    onChange={(e) => setAnimParamsJson(e.target.value)}
                    style={{ fontFamily: 'monospace', fontSize: 12 }}
                  />
                  <button
                    className="btn btn-primary"
                    disabled={busy || !activePid}
                    onClick={() => runAction('Generate Original Animation', async () => {
                      let parsedParams: Record<string, any> = {};
                      try {
                        parsedParams = JSON.parse(animParamsJson || '{}');
                      } catch {
                        parsedParams = {};
                      }
                      await apiPost(`/api/projects/${activePid}/animate`, {
                        template_id: animTemplate,
                        duration_sec: 6.5,
                        parameters: parsedParams,
                      });
                    })}
                  >
                    Generate Editable SceneSpec & Render
                  </button>
                </div>
                <h3 style={{ marginTop: 18 }}>Effects Library ({effects.length} presets)</h3>
                <div style={{ maxHeight: 280, overflowY: 'auto' }}>
                  {effects.map((fx) => (
                    <div key={fx.preset_id} style={{ padding: '6px 0', borderBottom: '1px solid var(--border)', fontSize: 12.5 }}>
                      <span className="pill">{fx.category}</span> <strong>{fx.name}</strong>
                      <div className="muted">{fx.editorial_purpose}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="card">
                <h2>Latest Generated Animation</h2>
                {animAsset ? (
                  <>
                    <div className="muted" style={{ marginBottom: 8 }}>{animAsset.relative_path}</div>
                    <video key={animAsset.sha256} controls style={{ width: '100%', borderRadius: 8 }} src={workspaceFileUrl(activePid, animAsset.relative_path)} />
                  </>
                ) : <p className="muted">No animation generated yet for this project.</p>}
              </div>
            </div>
          )}

          {tab === 'content' && (
            <div style={{ display: 'grid', gap: 16 }}>
              <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h2 style={{ margin: 0 }}>Multi-Format Outputs (16:9 Long-Form + 9:16 Safe-Zone Vertical)</h2>
                  <span className="muted">Each platform receives its own reframed, safe-area-aware cut & captions.</span>
                </div>
                <button className="btn" onClick={() => setShowSafeZone(!showSafeZone)}>
                  {showSafeZone ? 'Hide' : 'Show'} Safe-Zone Overlay
                </button>
              </div>
              <div className="grid-3">
                {shortAssets.map((sa) => (
                  <div className="card" key={sa.asset_id}>
                    <span className="pill pill-amber" style={{ marginBottom: 8 }}>{sa.role}</span>
                    <div className="video-stage">
                      <video controls src={workspaceFileUrl(activePid, sa.relative_path)} />
                      {showSafeZone && <div className="safe-zone-overlay" />}
                    </div>
                  </div>
                ))}
                {shortAssets.length === 0 && <div className="card"><p className="muted">Render project deliverables to inspect vertical Shorts, TikTok, and Instagram Reels variants.</p></div>}
              </div>
            </div>
          )}

          {tab === 'social' && (
            <div style={{ display: 'grid', gap: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2 style={{ margin: 0 }}>YouTube, Threads & Telegram Funnel</h2>
                <button className="btn btn-primary" disabled={busy || !activePid} onClick={() => runAction('Generate Social & Telegram', () => apiPost(`/api/projects/${activePid}/social`, {}))}>
                  Regenerate Social Package
                </button>
              </div>
              {socialBundle ? (
                <div className="grid-2">
                  <div className="card">
                    <h3>YouTube Titles & Description</h3>
                    <ul>{socialBundle.youtube.title_candidates.map((t: string, i: number) => <li key={i}>{t}</li>)}</ul>
                    {thumbAsset && <img src={workspaceFileUrl(activePid, thumbAsset.relative_path)} alt="Thumbnail" style={{ width: '100%', borderRadius: 8, marginBottom: 8 }} />}
                    <pre className="code-box">{socialBundle.youtube.description}</pre>
                  </div>
                  <div className="card">
                    <h3>Threads Post Candidates</h3>
                    <select value={threadsStyle} onChange={(e) => setThreadsStyle(e.target.value)} style={{ marginBottom: 10 }}>
                      {socialBundle.threads_posts.map((tp: any) => (
                        <option key={tp.post_id} value={tp.style}>{tp.style} ({tp.char_count} chars)</option>
                      ))}
                    </select>
                    {socialBundle.threads_posts
                      .filter((tp: any) => tp.style === threadsStyle)
                      .map((tp: any) => (
                        <div key={tp.post_id}>
                          <pre className="code-box">{tp.text}</pre>
                          <div className="muted">Chars: {tp.char_count}/{tp.max_chars} · CTA: {tp.cta_to_telegram}</div>
                        </div>
                      ))}
                    <h3 style={{ marginTop: 14 }}>Telegram Community Lead Post & Poll</h3>
                    <pre className="code-box">{socialBundle.telegram_campaign.lead_post}</pre>
                    {socialBundle.telegram_campaign.poll && (
                      <div style={{ fontSize: 12.5 }}>
                        <strong>Poll: {socialBundle.telegram_campaign.poll.question}</strong>
                        <ul>{socialBundle.telegram_campaign.poll.options.map((o: string, i: number) => <li key={i}>{o}</li>)}</ul>
                      </div>
                    )}
                  </div>
                </div>
              ) : <div className="card"><p className="muted">Generate social package to inspect YouTube metadata, Threads copy, and Telegram funnel.</p></div>}
            </div>
          )}

          {tab === 'growth' && (
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10, gap: 10, flexWrap: 'wrap' }}>
                <h2 style={{ margin: 0 }}>Growth Intelligence & Telegram Funnel Conversion</h2>
                {activePid && (
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    <button
                      className="btn btn-amber"
                      disabled={busy}
                      onClick={() => runAction('Evaluate Growth Experiments', () => apiPost(`/api/growth/${activePid}/evaluate`, {}))}
                    >
                      Evaluate Experiments
                    </button>
                    <input style={{ width: 170 }} value={tgChatId} onChange={(e) => setTgChatId(e.target.value)} placeholder="@channel_handle" />
                    <button
                      className="btn"
                      disabled={busy}
                      onClick={() => runAction('Sync Telegram Bot API', () => apiPost(`/api/growth/${activePid}/telegram-sync`, { chat_id: tgChatId }))}
                    >
                      Sync Telegram
                    </button>
                  </div>
                )}
              </div>
              {growthData ? (
                <>
                  <p className="muted">{growthData.note}</p>
                  {growthData.warnings?.length > 0 && (
                    <div style={{ marginBottom: 10 }}>
                      {growthData.warnings.map((w: string, i: number) => <div className="pill pill-rose" key={i}>{w}</div>)}
                    </div>
                  )}
                  <div style={{ display: 'grid', gap: 10, marginBottom: 14 }}>
                    {growthData.recommendations.map((r: string, i: number) => <div className="pill pill-green" key={i}>{r}</div>)}
                  </div>
                  {(growthData.platform_summaries || []).length > 0 && (
                    <table style={{ marginBottom: 14 }}>
                      <thead><tr><th>Platform</th><th>Views (Latest)</th><th>Engagements</th><th>Eng. Rate</th><th>Snapshots</th><th>TG Members</th></tr></thead>
                      <tbody>
                        {growthData.platform_summaries.map((s: any) => (
                          <tr key={s.platform}>
                            <td>{s.platform}</td>
                            <td>{Number(s.total_views_or_impressions).toLocaleString()}</td>
                            <td>{s.total_engagements}</td>
                            <td>{s.engagement_rate_pct}%</td>
                            <td>{s.snapshots}</td>
                            <td>{s.latest_member_count ?? '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                  {(growthData.target_progress || []).length > 0 && (
                    <div style={{ marginBottom: 16 }}>
                      <h3>Platform Growth Targets (1.0x Baseline · 1.5x Intermediate · 2.0x Stretch)</h3>
                      <table>
                        <thead>
                          <tr>
                            <th>Platform</th>
                            <th>Samples</th>
                            <th>Median 7d Views</th>
                            <th>Median Ratio vs Baseline</th>
                            <th>Reached 1.5x / 2.0x</th>
                            <th>Sample Confidence</th>
                          </tr>
                        </thead>
                        <tbody>
                          {growthData.target_progress.map((tp: any) => (
                            <tr key={tp.platform}>
                              <td><strong>{tp.platform}</strong></td>
                              <td>{tp.published_samples}</td>
                              <td>{tp.median_first_7d_views != null ? Number(tp.median_first_7d_views).toLocaleString() : '-'}</td>
                              <td>{tp.median_ratio_vs_baseline != null ? `${tp.median_ratio_vs_baseline}x` : '-'}</td>
                              <td>{tp.reached_1_5x_count} / {tp.reached_2_0x_count}</td>
                              <td>
                                <span className={`pill ${tp.sample_confidence === 'meaningful_multi_video_sample' ? 'pill-green' : 'pill-amber'}`}>
                                  {tp.sample_confidence}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                  <div style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, gap: 8, flexWrap: 'wrap' }}>
                      <h3 style={{ margin: 0 }}>
                        {expSearchResults !== null ? `Experiment Ledger Search (${expSearchResults.length})` : `Experiment History & Learned Hypotheses (${(growthData.experiments || []).length})`}
                      </h3>
                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        <input
                          style={{ width: 190 }}
                          value={expQuery}
                          onChange={(e) => setExpQuery(e.target.value)}
                          placeholder="Search experiments/hypotheses"
                        />
                        <select style={{ width: 145 }} value={expPlatformFilter} onChange={(e) => setExpPlatformFilter(e.target.value)}>
                          <option value="">All platforms</option>
                          <option value="youtube">youtube</option>
                          <option value="youtube_shorts">youtube_shorts</option>
                          <option value="tiktok">tiktok</option>
                          <option value="instagram_reels">instagram_reels</option>
                          <option value="threads">threads</option>
                        </select>
                        <button className="btn" disabled={busy} onClick={handleSearchExperiments}>Search Ledger</button>
                        {expSearchResults !== null && (
                          <button className="btn" onClick={() => setExpSearchResults(null)}>Project View</button>
                        )}
                      </div>
                    </div>
                    {(expSearchResults ?? growthData.experiments ?? []).length > 0 ? (
                      <table>
                        <thead>
                          <tr>
                            <th>Platform / Strategy</th>
                            <th>Prediction</th>
                            <th>7d Views & Baseline</th>
                            <th>Tier & Verdict</th>
                            <th>Learned Hypotheses</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(expSearchResults ?? growthData.experiments ?? []).map((ex: any) => (
                            <tr key={ex.experiment_id}>
                              <td>
                                <div><strong>{ex.platform}</strong> · {ex.strategy_name}</div>
                                <div className="muted">{ex.title} ({ex.editing_style})</div>
                              </td>
                              <td>
                                <div>{Number(ex.predicted_composite_score).toFixed(3)}</div>
                                <div className="muted">±{Number(ex.predicted_uncertainty).toFixed(2)}</div>
                              </td>
                              <td>
                                <div>{ex.first_7d_views != null ? `${Number(ex.first_7d_views).toLocaleString()} views` : 'Awaiting views'}</div>
                                <div className="muted">
                                  {ex.creator_baseline_median_views != null ? `Base ${Number(ex.creator_baseline_median_views).toLocaleString()}` : 'No baseline yet'}
                                  {ex.ratio_vs_creator_median != null ? ` (${ex.ratio_vs_creator_median}x)` : ''}
                                </div>
                              </td>
                              <td>
                                <div className={`pill ${ex.outcome_verdict === 'outperformed' ? 'pill-green' : ex.outcome_verdict === 'underperformed' ? 'pill-rose' : 'pill-amber'}`}>
                                  {ex.outcome_verdict}
                                </div>
                                <div className="muted" style={{ fontSize: 11.5, marginTop: 4 }}>{ex.growth_tier_reached}</div>
                              </td>
                              <td>
                                {(ex.learned_hypotheses || []).length > 0 ? (
                                  <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12 }}>
                                    {ex.learned_hypotheses.map((h: string, i: number) => <li key={i}>{h}</li>)}
                                  </ul>
                                ) : <span className="muted">Pending evaluation after post-publication data</span>}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : <p className="muted">Run AI Symposium on this project to log pre-publication hypotheses and track post-publication results.</p>}
                  </div>

                  <table>
                    <thead><tr><th>Platform</th><th>Campaign Slug</th><th>Tagged Funnel URL</th><th>Clicks</th><th>Joins</th></tr></thead>
                    <tbody>
                      {(growthData.funnel_links || []).map((l: any) => (
                        <tr key={l.link_id}>
                          <td>{l.source_platform}</td>
                          <td>{l.campaign_slug}</td>
                          <td><code>{l.tagged_url}</code></td>
                          <td>{l.clicks}</td>
                          <td>{l.conversions}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              ) : <p className="muted">Select a project to inspect funnel links and authorized analytics.</p>}
            </div>
          )}

          {tab === 'export' && (
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
                <h2 style={{ margin: 0 }}>Export Center & Provenance Registry</h2>
                {activePid && (
                  <button className="btn btn-primary" disabled={busy} onClick={() => runAction('Write Production Report', () => apiPost(`/api/projects/${activePid}/report`))}>
                    Refresh Production Report
                  </button>
                )}
              </div>
              {manifest ? (
                <table>
                  <thead><tr><th>Role</th><th>Workspace Path</th><th>Size</th><th>SHA-256</th><th>Action</th></tr></thead>
                  <tbody>
                    {manifest.assets.map((a) => (
                      <tr key={a.asset_id}>
                        <td><span className="pill">{a.role}</span></td>
                        <td><code>{a.relative_path}</code></td>
                        <td>{(a.size_bytes / 1024).toFixed(1)} KB</td>
                        <td><code>{a.sha256.slice(0, 12)}...</code></td>
                        <td><a className="btn" href={workspaceFileUrl(activePid, a.relative_path)} target="_blank" rel="noreferrer">Open / Download</a></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : <p className="muted">Select a project to view manifest artifacts.</p>}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
