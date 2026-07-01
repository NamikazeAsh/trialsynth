import argparse
import json
import html
import logging
from pathlib import Path

from trialsynth.base.extract.paths import RESULTS_GROUNDED_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

# todo: consider making a JINJA template for this
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical Trial Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/css/select2.min.css" rel="stylesheet" />
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/js/select2.min.js"></script>
    
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; padding: 0; background: #f8f9fa; color: #212529; line-height: 1.5; }
        .container { max-width: 1600px; margin: 0 auto; padding: 1rem; }
        header { margin-bottom: 1rem; }
        h1 { font-size: 1.2rem; font-weight: 600; margin: 0 0 0.75rem 0; color: #333; }
        .study-selector { background: #fff; padding: 0.75rem 1rem; border: 1px solid #dee2e6; border-radius: 4px; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 15px; }
        .select2-container { width: 400px !important; }
        .study-content { display: none; }
        .study-content.active { display: block; }
        
        .fields-grid { 
            display: grid; 
            grid-template-columns: repeat(3, 1fr); 
            gap: 1.5rem; 
        }
        
        .field { background: #fff; border: 1px solid #dee2e6; display: flex; flex-direction: column; height: 100%; }
        .field-header { padding: 0.6rem 1rem; border-bottom: 1px solid #dee2e6; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; color: #555; }
        .field-content { padding: 1rem; max-height: 700px; overflow-y: auto; flex-grow: 1; }
        .extraction { padding: 0.625rem 0.75rem; margin-bottom: 0.75rem; font-size: 0.875rem; border-left: 3px solid currentColor; background: #fafafa; }
        
        .field-arms .field-header { background: #e7f1f8; color: #1a5a8a; }
        .field-results .field-header { background: #e8f5e9; color: #2e6b33; } 
        .field-safety .field-header { background: #fce8e8; color: #8a3a3a; } 
        .field-genetic .field-header { background: #fff8e6; color: #8a6d1a; }
        .field-inclusion .field-header { background: #e8f4f0; color: #2a6b5a; }
        .field-exclusion .field-header { background: #f3e8f4; color: #6b2a6b; }

        .arm-result-card { border: 1px solid #eee; background: #fff; margin-bottom: 1rem; border-left: 3px solid #2e6b33; }
        .arm-safety-card { border: 1px solid #eee; background: #fff; margin-bottom: 1rem; border-left: 3px solid #8a3a3a; }
        .arm-card-header { background: #fafafa; padding: 5px 10px; font-weight: bold; font-size: 0.75rem; border-bottom: 1px solid #eee; color: #333; }
        .arm-card-content { padding: 8px 10px; }

        .grounding-badge { display: inline-block; padding: 2px 8px; background: #ff4d4f; color: white !important; border-radius: 4px; font-family: monospace; font-size: 0.75rem; font-weight: bold; text-decoration: none; margin-left: 10px; }
        .variation-badge { display: inline-block; padding: 2px 8px; background: #1890ff; color: white !important; border-radius: 4px; font-family: monospace; font-size: 0.75rem; font-weight: bold; margin-left: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        
        .data-table { width: 100%; border-collapse: collapse; margin-top: 5px; }
        .data-table td { padding: 3px 0; border: none; font-size: 0.8rem; }
        .empty-state { color: #999; font-style: italic; font-size: 0.8rem; padding: 10px; text-align: center; }
        
        /* HOVER EVIDENCE */
        .evidence-wrap { display: inline-block; margin-left: 6px; }
        .evidence-icon { display: inline-block; width: 14px; height: 14px; border-radius: 50%; background: #1a5a8a; color: white; font-size: 10px; line-height: 14px; text-align: center; cursor: help; font-weight: bold; }
        .floating-evidence-tooltip {
            display: none;
            position: fixed;
            z-index: 9999;
            max-width: 420px;
            min-width: 280px;
            background: #1f2937;
            color: #fff;
            padding: 10px 12px;
            border-radius: 6px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
            font-size: 0.76rem;
            line-height: 1.45;
        }

        .study-header { background: #fff; border: 1px solid #dee2e6; padding: 1rem; margin-bottom: 1rem; }
        .study-header h2 { margin: 0 0 0.5rem 0; font-size: 1.1rem; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Clinical Trial Dashboard</h1>
            <div class="study-selector">
                <label style="font-size:0.85rem; font-weight:bold;">Study Identifier:</label>
                <select id="study-select">{{OPTIONS}}</select>
            </div>
        </header>
        <main>{{STUDIES}}</main>
    </div>
    <div id="floating-evidence-tooltip" class="floating-evidence-tooltip"></div>
    <script>
        $(document).ready(function() {
            $('#study-select').select2({ placeholder: "Search...", allowClear: true });
            $('#study-select').on('select2:select', function (e) { showStudy(e.params.data.id); });
            const initialId = $('#study-select').val();
            if (initialId) showStudy(initialId);

            const tooltip = document.getElementById('floating-evidence-tooltip');
            const margin = 12;

            function positionTooltip(evt) {
                const rect = tooltip.getBoundingClientRect();
                let x = evt.clientX + 14;
                let y = evt.clientY + 14;
                if (x + rect.width + margin > window.innerWidth) { x = window.innerWidth - rect.width - margin; }
                if (y + rect.height + margin > window.innerHeight) { y = window.innerHeight - rect.height - margin; }
                tooltip.style.left = Math.max(margin, x) + 'px';
                tooltip.style.top = Math.max(margin, y) + 'px';
            }

            document.addEventListener('mouseover', function (evt) {
                const icon = evt.target.closest('.evidence-icon');
                if (!icon) return;
                const text = icon.getAttribute('data-evidence') || '';
                tooltip.innerHTML = text;
                tooltip.style.display = 'block';
                positionTooltip(evt);
            });

            document.addEventListener('mousemove', function (evt) {
                if (tooltip.style.display === 'block') { positionTooltip(evt); }
            });

            document.addEventListener('mouseout', function (evt) {
                if (evt.target.closest('.evidence-icon')) { tooltip.style.display = 'none'; }
            });
        });
        function showStudy(id) {
            document.querySelectorAll('.study-content').forEach(el => el.classList.remove('active'));
            const target = document.getElementById(id);
            if (target) target.classList.add('active');
        }
    </script>
</body>
</html>
'''


def format_grounding(g):
    if not g: return ""
    url = f"https://bioregistry.io/{g['db']}:{g['id']}"
    return f'''<a href="{url}" target="_blank" class="grounding-badge">{g['db']}:{g['id']}</a>'''


def build_evidence_hover(item):
    if not item or not isinstance(item, dict): return ""
    evidence = item.get("source_sentence") or item.get("evidence_text")
    if not evidence: return ""
    evidence_text = html.escape(str(evidence), quote=True)
    return (
        "<span class='evidence-wrap'>"
        f"<span class='evidence-icon' data-evidence=\"{evidence_text}\">i</span>"
        "</span>"
    )


def render_metric_row(name, value_text, item):
    evidence_hover = build_evidence_hover(item)
    grounding = format_grounding(item.get("grounding")) if isinstance(item, dict) else ""
    return (
        f"<tr><td>{html.escape(str(name))}{evidence_hover} {grounding}</td>"
        f"<td style='text-align:right; font-weight:600;'>{html.escape(str(value_text))}</td></tr>"
    )


def render_summary_item(result):
    if isinstance(result, str):
        text = result
        evidence_hover = ""
    elif isinstance(result, dict):
        text = result.get("text", "")
        evidence_hover = build_evidence_hover(result)
    else:
        text = str(result)
        evidence_hover = ""
    return f"<div class='extraction'>{html.escape(str(text))}{evidence_hover}</div>"


def render_criteria_item(item):
    if isinstance(item, str):
        text = item
        evidence_hover = ""
        grounding = ""
    elif isinstance(item, dict):
        text = item.get("text", "")
        evidence_hover = build_evidence_hover(item)
        grounding = format_grounding(item.get("grounding"))
    else:
        text = str(item)
        evidence_hover = ""
        grounding = ""
    return f"<div class='extraction'>{html.escape(str(text))}{evidence_hover} {grounding}</div>"


def generate_study_html(study_id, data, is_first):
    active_class = ' active' if is_first else ''

    # 1. Study Arms
    arm_html = ""
    for a in data.get('arms', []):
        evidence_hover = build_evidence_hover(a)
        arm_html += f'''<div class="extraction">
            <strong>{html.escape(str(a.get("arm_name", "Unknown Arm")))}</strong> <small>(n={a.get("n", "N/A")})</small>{evidence_hover}<br>
            <div style="font-size:0.8rem; margin-top:5px; color:#555;">{html.escape(str(a.get("dosage") or "No dosage specified"))}</div>
        </div>'''

    # 2. Findings
    results_html = ""
    results_html += "<div style='margin:0 0 10px 0; font-weight:bold; font-size:0.75rem; color:#2e6b33; text-transform:uppercase;'>Arm Metrics</div>"
    for a in data.get('arms', []):
        m_rows = "".join([render_metric_row(m.get('name'), m.get('value_text'), m) for m in a.get('metrics', [])])
        if m_rows:
            results_html += f'''<div class="arm-result-card"><div class="arm-card-header">{html.escape(str(a.get("arm_name", "Arm")))}</div><div class="arm-card-content"><table class="data-table">{m_rows}</table></div></div>'''
        else:
            results_html += f'''<div class="arm-result-card"><div class="arm-card-header">{html.escape(str(a.get("arm_name", "Arm")))}</div><div class="empty-state">No metrics extracted</div></div>'''

    if data.get('statistical_comparisons'):
        results_html += "<div style='margin:15px 0 10px 0; font-weight:bold; font-size:0.75rem; color:#2e6b33; text-transform:uppercase;'>Statistical Comparisons</div>"
        for c in data['statistical_comparisons']:
            m_rows = "".join([render_metric_row(m.get('name'), m.get('value_text'), m) for m in c.get('metrics', [])])
            results_html += f'<div class="extraction"><strong>{html.escape(str(c.get("comparison_name", "Comparison")))}</strong><table class="data-table">{m_rows}</table></div>'

    results_html += "<div style='margin:15px 0 10px 0; font-weight:bold; font-size:0.75rem; color:#2e6b33; text-transform:uppercase;'>Summary</div>"
    results_html += "".join([render_summary_item(r) for r in data.get('results', [])])

    # 3. Safety
    safety_html = ""
    for a in data.get('arms', []):
        ae_rows = "".join([render_metric_row(ae.get('event_name'), ae.get('value_text'), ae) for ae in a.get('adverse_events', [])])
        if ae_rows:
            safety_html += f'''<div class="arm-safety-card">
                <div class="arm-card-header">{html.escape(str(a.get("arm_name", "Arm")))}</div>
                <div class="arm-card-content"><table class="data-table">{ae_rows}</table></div>
            </div>'''
        else:
            safety_html += f'''<div class="arm-safety-card"><div class="arm-card-header">{html.escape(str(a.get("arm_name", "Arm")))}</div><div class="empty-state">No safety data</div></div>'''

    # 4. Genetic
    genetic_html = ""
    unique_ids = set()
    # Handle grounded_inclusion list with robustness
    for item in data.get('genetic', {}).get('grounded_inclusion', []):
        if not isinstance(item, dict): continue
        badges = []
        for g_info in item.get('groundings', []):
            if not isinstance(g_info, dict) or 'info' not in g_info: continue
            # Only HGNC genetic groundings, matching the CoGEx ingestion filter
            if g_info['info'].get('db') != 'HGNC': continue
            gid = f"{g_info['info'].get('db')}:{g_info['info'].get('id')}"
            if gid not in unique_ids:
                badges.append(format_grounding(g_info['info']))
                unique_ids.add(gid)
        gene_badges = " ".join(badges)
        variant_badge = f'<span class="variation-badge">{html.escape(str(item.get("variant", "")))}</span>' if item.get('variant') else ""
        evidence_hover = build_evidence_hover(item)
        if gene_badges or variant_badge:
            genetic_html += f'<div class="extraction"><strong>{html.escape(str(item.get("text", "")))}</strong> {evidence_hover} {gene_badges} {variant_badge}</div>'
    if not genetic_html: genetic_html = "<div class='empty-state'>No biomarkers identified.</div>"

    # 5. Inclusion
    inclusion_html = "".join([render_criteria_item(item) for item in data.get('grounded_inclusion_criteria', [])])
    if not inclusion_html: inclusion_html = "<div class='empty-state'>No criteria identified.</div>"

    # 6. Exclusion
    exclusion_html = "".join([render_criteria_item(item) for item in data.get('grounded_exclusion_criteria', [])])
    if not exclusion_html: exclusion_html = "<div class='empty-state'>No exclusion criteria identified.</div>"

    header = f'''<div class="study-header">
        <a href="https://pubmed.ncbi.nlm.nih.gov/{data.get('pmid', '')}/" target="_blank" style="float:right; font-size:0.8rem; color:#1a5a8a;">View PubMed</a>
        <h2>{html.escape(str(data.get('study_info', 'Missing Study Info')))}</h2>
        <div style="font-size:0.8rem; color:#666;">PMID: {data.get('pmid', 'N/A')} | Registry: {", ".join(data.get('trial_ids', []))}</div>
    </div>'''

    grid = f'''<div class="fields-grid">
        <div class="field field-arms"><div class="field-header">Study Arms</div><div class="field-content">{arm_html}</div></div>
        <div class="field field-safety"><div class="field-header">Safety & Adverse Events</div><div class="field-content">{safety_html}</div></div>
        <div class="field field-results"><div class="field-header">Findings</div><div class="field-content">{results_html}</div></div>
        <div class="field field-genetic"><div class="field-header">Genetic Markers</div><div class="field-content">{genetic_html}</div></div>
        <div class="field field-inclusion"><div class="field-header">Inclusion Criteria</div><div class="field-content">{inclusion_html}</div></div>
        <div class="field field-exclusion"><div class="field-header">Exclusion Criteria</div><div class="field-content">{exclusion_html}</div></div>
    </div>'''

    return f'<div id="{study_id}" class="study-content{active_class}">{header}{grid}</div>'


def main():
    parser = argparse.ArgumentParser(description="Generate HTML dashboard.")
    parser.add_argument("--input-dir", default=str(RESULTS_GROUNDED_DIR.base))
    parser.add_argument(
        "--output-html", default=str(RESULTS_DIR.join(name="dashboard.html"))
    )
    args = parser.parse_args()

    results_dir = Path(args.input_dir)
    output_html = Path(args.output_html)
    json_files = sorted(results_dir.glob('*.json'))
    studies = []

    logger.info(f"Scanning {len(json_files)} JSON files...")

    for fname in json_files:
        with open(fname, encoding='utf-8') as f:
            try:
                d = json.load(f)
                if 'pmid' not in d: continue
                trial_id_str = d.get('trial_ids', [''])[0] if d.get('trial_ids') else 'No NCT'
                studies.append((f"study_{d['pmid']}", f"PMID: {d['pmid']} ({trial_id_str})", d))
            except Exception as e:
                logger.error(f"SKIPPING {fname.name}: {e}")
                continue

    if not studies:
        logger.error("No valid studies collected.")
        return

    options = '\n'.join([f'<option value="{s[0]}">{s[1]}</option>' for s in studies])
    studies_content = '\n'.join([generate_study_html(s[0], s[2], i == 0) for i, s in enumerate(studies)])

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE.replace('{{OPTIONS}}', options).replace('{{STUDIES}}', studies_content))

    logger.info(f"Dashboard generated with {len(studies)} studies -> {output_html}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    main()
