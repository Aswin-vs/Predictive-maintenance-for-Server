import os
import sys
import subprocess
import shutil
from pathlib import Path
import markdown

PROJECT_DIR = Path(r"C:\Users\LOQ\Documents\Antigravity\Predictive maintenance for Server")
DOWNLOADS_DIR = Path(r"C:\Users\LOQ\Downloads")
BRAIN_DIR = Path(r"C:\Users\LOQ\.gemini\antigravity-ide\brain\a5dcf0ab-05ac-433e-a194-5a8c6e689f7e")
SCRATCH_DIR = BRAIN_DIR / "scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(CHROME_EXE):
    CHROME_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page {
    size: A4;
    margin: 16mm 14mm 16mm 14mm;
    @bottom-right {
        content: counter(page);
    }
}
@media print {
    body {
        font-size: 9.5pt;
        line-height: 1.5;
        color: #1e293b;
        background: #fff;
    }
    h1, h2, h3, h4 {
        color: #0f172a;
        page-break-after: avoid;
    }
    table, pre, blockquote, .card {
        page-break-inside: avoid;
    }
    a {
        color: #2563eb;
        text-decoration: none;
    }
    .page-break {
        page-break-after: always;
    }
}
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    color: #1e293b;
    line-height: 1.55;
    background: #fff;
    max-width: 960px;
    margin: 0 auto;
    padding: 10px 15px;
}
h1 {
    font-size: 20pt;
    color: #0f172a;
    border-bottom: 2.5px solid #2563eb;
    padding-bottom: 6px;
    margin-top: 24px;
    margin-bottom: 12px;
}
h2 {
    font-size: 15pt;
    color: #1e293b;
    border-bottom: 1.5px solid #cbd5e1;
    padding-bottom: 5px;
    margin-top: 20px;
    margin-bottom: 10px;
}
h3 {
    font-size: 12pt;
    color: #334155;
    margin-top: 16px;
    margin-bottom: 8px;
}
h4 {
    font-size: 10.5pt;
    color: #475569;
    margin-top: 12px;
    margin-bottom: 6px;
}
p {
    margin-top: 0;
    margin-bottom: 8px;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 8.5pt;
}
th, td {
    border: 1px solid #cbd5e1;
    padding: 5px 8px;
    text-align: left;
    vertical-align: top;
}
th {
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 600;
}
tr:nth-child(even) {
    background-color: #f8fafc;
}
code {
    font-family: 'Consolas', 'Cascadia Code', 'Courier New', monospace;
    background: #f1f5f9;
    color: #0f172a;
    padding: 2px 4px;
    border-radius: 4px;
    font-size: 8pt;
}
pre {
    background: #0f172a;
    color: #f8fafc;
    border-radius: 6px;
    padding: 10px 12px;
    overflow-x: auto;
    font-size: 8pt;
    line-height: 1.4;
    margin: 10px 0;
}
pre code {
    background: none;
    color: inherit;
    padding: 0;
    border-radius: 0;
}
blockquote {
    border-left: 4px solid #3b82f6;
    margin: 12px 0;
    padding: 6px 14px;
    background: #eff6ff;
    color: #1e3a8a;
    border-radius: 0 6px 6px 0;
}
hr {
    border: 0;
    border-top: 1px solid #e2e8f0;
    margin: 18px 0;
}
ul, ol {
    margin-top: 0;
    margin-bottom: 8px;
    padding-left: 22px;
}
li {
    margin-bottom: 3px;
}

/* Front Cover Styling */
.cover-page {
    text-align: center;
    padding: 70px 20px 50px;
    page-break-after: always;
}
.cover-badge {
    display: inline-block;
    background: #2563eb;
    color: white;
    font-size: 11pt;
    font-weight: 600;
    padding: 6px 20px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 24px;
}
.cover-title {
    font-size: 26pt;
    color: #0f172a;
    margin-bottom: 14px;
    border: none;
    line-height: 1.25;
}
.cover-subtitle {
    font-size: 13pt;
    color: #64748b;
    margin-bottom: 30px;
    max-width: 750px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.5;
}
.cover-meta {
    margin-top: 60px;
    font-size: 10.5pt;
    color: #475569;
    line-height: 1.85;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 20px;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
    text-align: left;
}
.page-break {
    page-break-after: always;
}

/* Special Addendum Distinct Styling */
.addendum-cover {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    color: #f8fafc;
    padding: 40px 30px;
    border-radius: 12px;
    margin: 20px 0 30px 0;
    border: 2px solid #6366f1;
    box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.2);
}
.addendum-badge {
    display: inline-block;
    background: #fbbf24;
    color: #1e1b4b;
    font-size: 10.5pt;
    font-weight: 700;
    padding: 6px 18px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 16px;
}
.addendum-cover h1 {
    color: #ffffff;
    font-size: 22pt;
    border-bottom: 2px solid #818cf8;
    padding-bottom: 8px;
    margin-top: 0;
}
.addendum-cover p {
    color: #e0e7ff;
    font-size: 11pt;
    line-height: 1.6;
}
.qa-card {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 5px solid #4f46e5;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 16px 0;
}
.qa-card h3 {
    color: #1e1b4b;
    margin-top: 0;
    font-size: 12.5pt;
}
.qa-tag {
    display: inline-block;
    background: #e0e7ff;
    color: #3730a3;
    font-size: 8pt;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 12px;
    margin-bottom: 8px;
}
"""

def compile_html_to_pdf(html_path, pdf_path):
    cmd = [
        CHROME_EXE,
        '--headless=new',
        '--disable-gpu',
        '--run-all-compositor-stages-before-draw',
        f'--print-to-pdf={pdf_path}',
        str(html_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to compile PDF: {res.stderr}")
    return pdf_path

def main():
    print("="*70)
    print("Edge-AI Predictive Maintenance — Master Project Dossier PDF Builder")
    print("="*70)

    # 1. Load Document Sources
    arch_file = PROJECT_DIR / "docs" / "architecture" / "ARCHITECTURE.md"
    viva_file = PROJECT_DIR / "docs" / "VIVA_MASTER_GUIDE.md"
    audit_file = PROJECT_DIR / "AUDIT_REPORT.md"
    walkthrough_file = PROJECT_DIR / "docs" / "project_viva_walkthrough.md"
    runbook_file = PROJECT_DIR / "docs" / "master_execution_guide.md"
    addendum_file = PROJECT_DIR / "docs" / "SESSION_32_VIVA_DEFENSE_ADDENDUM.md"

    # Verify existing files
    sources = [
        ("Architecture Reference", arch_file),
        ("Viva Master Guide", viva_file),
        ("Pre-Release Audit Report", audit_file),
        ("Live Demonstration Walkthrough", walkthrough_file),
        ("Master Execution Runbook", runbook_file),
        ("Session Defense Addendum", addendum_file)
    ]

    for name, path in sources:
        if not path.exists():
            print(f"ERROR: Missing source document: {path}")
            return 1
        print(f"[FOUND] {name}: {path.name} ({path.stat().st_size / 1024:.1f} KB)")

    # 2. Build Cover Page HTML
    cover_html = """
    <div class="cover-page">
        <div class="cover-badge">B.Tech Final Year Master Project Deliverable</div>
        <h1 class="cover-title">Edge-AI Enabled Predictive Maintenance for Server Hardware</h1>
        <div class="cover-subtitle">
            Complete Technical & Academic Project Dossier: Unified Architecture Reference, Mathematical Proofs, Pre-Release Production Audit, Multi-Device Runbook, Comprehensive Viva Master Guide, and Advanced Technical Defense Addendum
        </div>
        <div class="cover-meta">
            <strong>Degree Program:</strong> B.Tech in Computer Science & Engineering (Artificial Intelligence Specialization)<br>
            <strong>Academic Year:</strong> 2026<br>
            <strong>System Architecture:</strong> 3-Device Heterogeneous Distributed Mesh (Patient Server, RPi4 Doctor Edge, LOQ RTX 4060 Hospital)<br>
            <strong>Edge AI Runtime:</strong> Google LiteRT INT8 FlatBuffers (&lt; 0.6 ms Execution on ARM Cortex-A72)<br>
            <strong>Cloud Deep Engine:</strong> PyTorch FP32 Multi-Model Fusion, TimescaleDB, and Ollama Qwen 3.5 4B Context-Aware RAG<br>
            <strong>Inter-Node Fabric:</strong> Tailscale WireGuard Mesh (256-bit ChaCha20-Poly1305) &amp; Paho MQTT<br>
            <strong>Verification Status:</strong> 100/100 Production Certified (36/36 Automated Regression Tests Passing)<br>
            <strong>Compilation Date:</strong> October 2026<br>
            <strong>Repository:</strong> https://github.com/Aswin-vs/Predictive-maintenance-for-Server
        </div>
    </div>
    """

    # 3. Table of Contents & Dossier Executive Overview
    toc_html = """
    <div class="page-break"></div>
    <h1>Master Project Dossier: Architectural Table of Contents</h1>
    <p>This master dossier constitutes the single authoritative, complete technical, empirical, and academic documentation for the <strong>Edge-AI Predictive Maintenance for Server Hardware</strong> project. It is structured into 6 comprehensive parts:</p>
    
    <table>
        <thead>
            <tr>
                <th style="width: 12%;">Dossier Part</th>
                <th style="width: 28%;">Title / Scope</th>
                <th style="width: 40%;">Key Technical Content</th>
                <th style="width: 20%;">Source Document</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Part I</strong></td>
                <td><strong>Technical Architecture Reference</strong></td>
                <td>15 detailed sections + 3 appendices covering system overview, 3-device topology, 28/36-feature schemas, 10 AI models, edge pipeline, cloud fusion, TimescaleDB, and latency waterfall.</td>
                <td><code>docs/architecture/ARCHITECTURE.md</code></td>
            </tr>
            <tr>
                <td><strong>Part II</strong></td>
                <td><strong>Exhaustive Viva Defense Master Guide</strong></td>
                <td>22 technical chapters with over 100 examiner-ready defense answers: survival loss (alpha=5.0), INT8 affine quantization, Welford's algorithm, hardware plugins, and physical sensors.</td>
                <td><code>docs/VIVA_MASTER_GUIDE.md</code></td>
            </tr>
            <tr>
                <td><strong>Part III</strong></td>
                <td><strong>Master Pre-Release Audit &amp; Technical Remediation Report</strong></td>
                <td>Official 100/100 production certification report, ground truth reconciliation matrices across LOQ/RPi4/IdeaPad, defect logs (BUG-01 to BUG-11), and automated test suites.</td>
                <td><code>AUDIT_REPORT.md</code></td>
            </tr>
            <tr>
                <td><strong>Part IV</strong></td>
                <td><strong>Live Viva Demonstration &amp; Inspection Script</strong></td>
                <td>Step-by-step presentation script for examiners: 2-mode operator dashboard, live sensor streaming, synthetic fault injection, SHAP waterfall charts, and AI chat assistant.</td>
                <td><code>docs/project_viva_walkthrough.md</code></td>
            </tr>
            <tr>
                <td><strong>Part V</strong></td>
                <td><strong>Master Execution &amp; Operations Runbook</strong></td>
                <td>Complete multi-node startup procedures: Option A 1-click master launcher, Option B step-by-step execution, service verification, port bindings, and curl health check probes.</td>
                <td><code>docs/master_execution_guide.md</code></td>
            </tr>
            <tr>
                <td><strong>Part VI</strong><br><span style="color:#4f46e5;font-weight:700;">[SPECIAL]</span></td>
                <td><strong>Special Comprehensive Addendum: Advanced Technical &amp; Viva Inquiries</strong></td>
                <td><strong>Marked separately at the last pages of this dossier.</strong> Exhaustive, code-grounded investigations into: unhealthy-only training impossibility, definitive 10-model empirical score ledger, 93% health dial vs 99.30% accuracy, multi-database normalization without monolithic CSV, WireGuard endpoint security, confidence score math &amp; 14% drop root-cause, 4 data leakage fixes, INT8 affine quantization, and per-server adaptive baseline self-learning.</td>
                <td><code>docs/SESSION_32_VIVA_DEFENSE_ADDENDUM.md</code></td>
            </tr>
        </tbody>
    </table>
    <br>
    <hr>
    """

    # 4. Render Markdown sections to HTML
    print("Converting Markdown sources to publication HTML...")
    rendered_parts = []

    # Part I: Architecture
    print("-> Rendering Part I: Architecture Reference...")
    arch_md = arch_file.read_text(encoding='utf-8')
    rendered_parts.append('<div class="page-break"></div>\n' + markdown.markdown(arch_md, extensions=['tables', 'fenced_code', 'toc']))

    # Part II: Viva Master Guide
    print("-> Rendering Part II: Viva Master Guide...")
    viva_md = viva_file.read_text(encoding='utf-8')
    rendered_parts.append('<div class="page-break"></div>\n' + markdown.markdown(viva_md, extensions=['tables', 'fenced_code', 'toc']))

    # Part III: Audit Report
    print("-> Rendering Part III: Pre-Release Audit Report...")
    audit_md = audit_file.read_text(encoding='utf-8')
    rendered_parts.append('<div class="page-break"></div>\n' + markdown.markdown(audit_md, extensions=['tables', 'fenced_code', 'toc']))

    # Part IV: Viva Demo Walkthrough
    print("-> Rendering Part IV: Live Viva Demonstration Script...")
    walkthrough_md = walkthrough_file.read_text(encoding='utf-8')
    rendered_parts.append('<div class="page-break"></div>\n' + markdown.markdown(walkthrough_md, extensions=['tables', 'fenced_code', 'toc']))

    # Part V: Master Execution Guide
    print("-> Rendering Part V: Master Execution Runbook...")
    runbook_md = runbook_file.read_text(encoding='utf-8')
    rendered_parts.append('<div class="page-break"></div>\n' + markdown.markdown(runbook_md, extensions=['tables', 'fenced_code', 'toc']))

    # Part VI: Special Addendum (Marked separately with distinctive styling at the end)
    print("-> Rendering Part VI: SPECIAL COMPREHENSIVE ADDENDUM (Final Pages)...")
    addendum_md = addendum_file.read_text(encoding='utf-8')
    addendum_rendered = markdown.markdown(addendum_md, extensions=['tables', 'fenced_code', 'toc'])
    
    addendum_html = f"""
    <div class="page-break"></div>
    <div class="addendum-cover">
        <div class="addendum-badge">&#9733; SPECIAL COMPREHENSIVE ADDENDUM</div>
        <h1>In-Depth Technical Audit &amp; Advanced Viva Defense Inquiries</h1>
        <p><strong>Audience Note:</strong> This final dedicated addendum synthesizes and formally records the exhaustive technical inquiries, theoretical proofs, and empirical audits investigated during the October 2026 defense preparation. Every explanation is cross-verified directly against the production codebase, configuration files, and live physical hardware telemetry.</p>
    </div>
    {addendum_rendered}
    """
    rendered_parts.append(addendum_html)

    # 5. Assemble Full Dossier HTML
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Edge-AI Predictive Maintenance — Complete Master Project Dossier</title>
<style>{CSS}</style>
</head>
<body>
{cover_html}
{toc_html}
{''.join(rendered_parts)}
</body>
</html>"""

    dossier_html_path = SCRATCH_DIR / "Edge_AI_Complete_Master_Project_Dossier.html"
    dossier_html_path.write_text(full_html, encoding='utf-8')
    print(f"Assembled HTML Dossier: {dossier_html_path.stat().st_size / 1024:.1f} KB")

    # 6. Target Destinations
    # User requested: "also save it in root project directory"
    root_pdf_path = PROJECT_DIR / "Edge_AI_Complete_Master_Project_Dossier.pdf"
    downloads_pdf_path = DOWNLOADS_DIR / "Edge_AI_Complete_Master_Project_Dossier.pdf"
    artifact_pdf_path = BRAIN_DIR / "Edge_AI_Complete_Master_Project_Dossier.pdf"

    # 7. Compile via Headless Browser
    print("Compiling Master PDF via Headless Chrome Engine...")
    compile_html_to_pdf(dossier_html_path, root_pdf_path)
    print(f"[SUCCESS] Saved to Project Root: {root_pdf_path} ({root_pdf_path.stat().st_size / 1024:.1f} KB)")

    # Copy to Downloads and Artifact directory for convenience
    try:
        shutil.copy2(root_pdf_path, downloads_pdf_path)
        print(f"[SUCCESS] Saved to Downloads: {downloads_pdf_path}")
    except Exception as e:
        print(f"Notice: Could not copy to Downloads: {e}")

    try:
        shutil.copy2(root_pdf_path, artifact_pdf_path)
        print(f"[SUCCESS] Saved to IDE Artifacts: {artifact_pdf_path}")
    except Exception as e:
        print(f"Notice: Could not copy to Artifacts: {e}")

    # Optional: cleanup scratch html
    # dossier_html_path.unlink()
    print("="*70)
    print("Master Dossier PDF generation complete!")
    print("="*70)

if __name__ == "__main__":
    main()
