"""One small style sheet for the whole app, so every tab shares the same design language.

Colors: slate text, one blue accent, green/red only for up/down changes, amber only for warnings.
"""

CSS = """
<style>
  .block-container {padding-top: 4.2rem; max-width: 1240px;}   /* clears the hosted toolbar (Share, GitHub) */
  h1, h2, h3 {letter-spacing: -0.01em;}

  /* Built-in metric widgets (Overview tab) */
  div[data-testid="stMetric"] {
      background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px 18px;
  }

  /* Section labels: small caps above a group of cards */
  .bi-section {font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
               color: #64748B; margin: 1.4rem 0 0.5rem;}

  /* Headline block */
  .bi-headline {font-size: 1.55rem; font-weight: 650; line-height: 1.3; color: #0F172A; margin: 0.2rem 0 0.35rem;}
  .bi-headline.small {font-size: 1.2rem;}
  .bi-sub {color: #475569; font-size: 1rem; line-height: 1.55; margin-bottom: 0.4rem;}

  /* KPI cards */
  .bi-kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 0.6rem 0 1.1rem;}
  .bi-kpi {background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px 16px;}
  .bi-kpi .label {font-size: 0.78rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
                  line-height: 1.3; min-height: 2.05em;}
  .bi-kpi .value {font-size: 1.5rem; font-weight: 650; color: #0F172A; margin-top: 4px; line-height: 1.2;
                  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
  .bi-kpi .note {font-size: 0.85rem; color: #64748B; margin-top: 4px;}
  /* Changes get an arrow but no green/red: whether "up" is good depends on the metric (sales vs response time). */
  .bi-kpi .note.up, .bi-kpi .note.down {color: #334155; font-weight: 500;}

  /* Insight cards (findings, takeaways, recommendations) */
  .bi-card {border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px 16px; margin-bottom: 10px; background: #FFFFFF;}
  .bi-card .title {font-weight: 600; color: #0F172A; font-size: 1rem;}
  .bi-card .detail {color: #475569; font-size: 0.93rem; margin-top: 3px; line-height: 1.5;}
  .bi-card .num {display: inline-block; min-width: 1.6rem; color: #2563EB; font-weight: 700;}
  .bi-tag {display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 1px 8px; border-radius: 999px;
           background: #FEF3C7; color: #92400E; margin-left: 6px; vertical-align: middle;}

  /* Callouts: tone = info | warn | neutral */
  .bi-callout {border-radius: 12px; padding: 12px 16px; margin: 0.4rem 0 0.8rem; border: 1px solid;}
  .bi-callout .title {font-weight: 600; margin-bottom: 4px;}
  .bi-callout ul {margin: 0; padding-left: 1.1rem;}
  .bi-callout li {margin: 2px 0; line-height: 1.5;}
  .bi-callout.info {background: #EFF6FF; border-color: #BFDBFE; color: #1E3A8A;}
  .bi-callout.warn {background: #FFFBEB; border-color: #FDE68A; color: #78350F;}
  .bi-callout.neutral {background: #F8FAFC; border-color: #E2E8F0; color: #334155;}

  /* Empty / unanswerable states */
  .bi-empty {border: 1px dashed #CBD5E1; border-radius: 12px; padding: 18px 20px; background: #FCFCFD;}
  .bi-empty .kicker {font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: #64748B;}
  .bi-empty .title {font-size: 1.1rem; font-weight: 600; color: #0F172A; margin: 4px 0;}
  .bi-empty .body {color: #475569;}

  /* Welcome screen */
  .bi-hero {font-size: 2.4rem; font-weight: 700; color: #0F172A; margin-bottom: 0.2rem;}
  .bi-hero-sub {font-size: 1.15rem; color: #475569; margin-bottom: 1.6rem;}
  .bi-feature .title {font-weight: 600; color: #0F172A; margin-bottom: 4px;}
  .bi-feature .body {color: #475569; font-size: 0.95rem; line-height: 1.5;}

  /* Homepage */
  .bi-kicker {font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #2563EB;}
  .bi-dataset {display: flex; justify-content: space-between; align-items: flex-start; gap: 16px;
               padding: 18px 22px; border: 1px solid #E2E8F0; border-radius: 14px;
               background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 100%); margin-bottom: 0.4rem;}
  .bi-dataset-name {font-size: 1.75rem; font-weight: 700; color: #0F172A; margin: 2px 0 8px; word-break: break-all;}
  .bi-chips {display: flex; flex-wrap: wrap; gap: 6px;}
  .bi-chip {font-size: 0.8rem; color: #334155; background: #FFFFFF; border: 1px solid #E2E8F0;
            border-radius: 999px; padding: 2px 10px;}
  .bi-status {font-size: 0.85rem; font-weight: 600; color: #0F172A; white-space: nowrap; text-align: right;}
  .bi-status-note {font-weight: 400; color: #64748B; font-size: 0.8rem; margin-top: 2px;}
  .bi-dot {display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #22C55E; margin-right: 6px;}
  .bi-panel-title {font-weight: 650; font-size: 1.02rem; color: #0F172A;}
  .bi-panel-sub {color: #64748B; font-size: 0.85rem; margin-bottom: 6px;}
  .bi-nav-title {font-weight: 600; color: #0F172A;}
  .bi-nav-body {color: #64748B; font-size: 0.85rem; min-height: 4.6em; margin: 2px 0 6px; line-height: 1.5;}
  .bi-kpis.compact {gap: 8px; margin: 0.3rem 0 0.6rem;}
  .bi-kpis.compact .bi-kpi {padding: 10px 12px;}
  .bi-kpis.compact .label {min-height: 0;}
  .bi-kpis.compact .value {font-size: 1.25rem;}
  @media (max-width: 1100px) {
    .bi-kpis {grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)) !important;}
  }
  .bi-step {color: #2563EB; font-weight: 700; font-size: 0.85rem;}
</style>
"""
