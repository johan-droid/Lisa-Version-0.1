from __future__ import annotations

import json


def render_dashboard_html(
    ws_path: str = "/ws", legacy_ws_path: str = "/ws/dashboard"
) -> str:
    ws_path_literal = json.dumps(str(ws_path)).replace("</", "<\\/")
    legacy_ws_path_literal = json.dumps(str(legacy_ws_path)).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LISA Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #08111f;
      --panel: rgba(8, 17, 31, 0.82);
      --panel-border: rgba(92, 160, 255, 0.18);
      --accent: #6fd5ff;
      --accent-2: #90f0b6;
      --text: #e6f1ff;
      --muted: #93a8c7;
    }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      background:
        radial-gradient(circle at top, rgba(111, 213, 255, 0.18), transparent 30%),
        linear-gradient(180deg, #07101d 0%, #0b1524 55%, #050b14 100%);
      color: var(--text);
      min-height: 100vh;
    }}
    .shell {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      margin-bottom: 24px;
    }}
    h1 {{ margin: 0; font-size: 32px; letter-spacing: -0.03em; }}
    .subtitle {{ color: var(--muted); margin-top: 6px; }}
    .status {{
      padding: 10px 14px;
      border: 1px solid var(--panel-border);
      border-radius: 999px;
      background: rgba(255,255,255,0.04);
      color: var(--accent);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .card {{
      padding: 16px;
      border-radius: 18px;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      box-shadow: 0 20px 60px rgba(0,0,0,0.25);
      backdrop-filter: blur(16px);
    }}
    .metric-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; }}
    .metric-value {{ font-size: 30px; margin-top: 10px; font-weight: 700; }}
    .metric-detail {{ color: var(--muted); margin-top: 8px; font-size: 13px; line-height: 1.45; }}
    .panels {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 14px;
      margin-top: 14px;
    }}
    .wide {{
      margin-top: 14px;
    }}
    .stack {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 14px;
    }}
    .list {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.08);
      color: var(--text);
      font-size: 13px;
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }}
    .table th, .table td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      font-size: 14px;
    }}
    .banner {{
      margin-top: 14px;
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(111, 213, 255, 0.12);
      border: 1px solid rgba(111, 213, 255, 0.24);
      color: var(--text);
    }}
    .chart-box {{
      height: 360px;
    }}
    .auth-panel {{
      margin-bottom: 18px;
      padding: 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .auth-panel form {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 10px;
    }}
    .auth-panel input {{
      flex: 1 1 280px;
      min-width: 220px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(8, 17, 31, 0.75);
      color: var(--text);
      padding: 12px 14px;
    }}
    .auth-panel button {{
      border: 0;
      border-radius: 12px;
      padding: 12px 16px;
      background: linear-gradient(135deg, #6fd5ff, #90f0b6);
      color: #04111d;
      font-weight: 700;
      cursor: pointer;
    }}
    canvas {{ width: 100% !important; height: 100% !important; }}
    @media (max-width: 900px) {{
      .grid, .panels {{ grid-template-columns: 1fr; }}
      header {{ align-items: start; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>LISA Live Dashboard</h1>
        <div class="subtitle">Realtime metrics from the Message Hub, Task Conductor, and Notepad, rendered with Chart.js.</div>
      </div>
      <div class="status" id="hubStatus">Connecting...</div>
    </header>
    <div class="auth-panel" id="authPanel" hidden>
      <div class="metric-label">Session Authentication Required</div>
      <div class="metric-detail" id="authStatus">Enter the current admin token or bot security key to mint a short-lived dashboard session.</div>
      <form id="authForm">
        <input id="authCredential" type="password" autocomplete="current-password" placeholder="Admin token or bot security key" />
        <button type="submit">Unlock Dashboard</button>
      </form>
    </div>
      <div class="grid">
      <div class="card"><div class="metric-label">Active Tasks</div><div class="metric-value" id="activeTasks">0</div></div>
      <div class="card"><div class="metric-label">Token Consumption</div><div class="metric-value" id="tokens">0</div></div>
      <div class="card"><div class="metric-label">Evolution Rate</div><div class="metric-value" id="evolution">0.00 / min</div></div>
      <div class="card"><div class="metric-label">Dominant Persona</div><div class="metric-value" id="persona">-</div></div>
      <div class="card"><div class="metric-label">Evolution Status</div><div class="metric-value" id="evolutionStatus">idle</div><div class="metric-detail" id="lastSkill">No skill yet</div></div>
    </div>
    <div class="panels">
      <div class="card chart-box"><canvas id="timelineChart"></canvas></div>
      <div class="card chart-box"><canvas id="personaChart"></canvas></div>
    </div>
    <div class="stack">
      <div class="card">
        <div class="metric-label">Installed Skills and Capabilities</div>
        <div class="list" id="capabilitiesList"></div>
      </div>
      <div class="card">
        <div class="metric-label">Personal Context Summary</div>
        <div class="metric-detail" id="personalSummary">No personal context loaded.</div>
      </div>
    </div>
    <div class="wide card">
      <div class="metric-label">Active Tasks and Alerts</div>
      <div class="banner" id="alertBanner">System healthy.</div>
      <table class="table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id="statusTable"></tbody>
      </table>
    </div>
  </div>
  <script>
    const statusNode = document.getElementById("hubStatus");
    const activeTasksNode = document.getElementById("activeTasks");
    const tokensNode = document.getElementById("tokens");
    const evolutionNode = document.getElementById("evolution");
    const personaNode = document.getElementById("persona");
    const evolutionStatusNode = document.getElementById("evolutionStatus");
    const lastSkillNode = document.getElementById("lastSkill");
    const capabilitiesListNode = document.getElementById("capabilitiesList");
    const personalSummaryNode = document.getElementById("personalSummary");
    const alertBannerNode = document.getElementById("alertBanner");
    const statusTableNode = document.getElementById("statusTable");
    const authPanelNode = document.getElementById("authPanel");
    const authStatusNode = document.getElementById("authStatus");
    const authFormNode = document.getElementById("authForm");
    const authCredentialNode = document.getElementById("authCredential");
    const timelineCtx = document.getElementById("timelineChart");
    const personaCtx = document.getElementById("personaChart");
    let timelineChart;
    let personaChart;
    let reconnectTimer;

    function connect() {{
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const socketPath = {ws_path_literal};
      const legacySocketPath = {legacy_ws_path_literal};
      const socket = new WebSocket(`${{proto}}://${{location.host}}${{socketPath}}`);
      // Legacy endpoint kept for compatibility with older deployments: ${{legacySocketPath}}
      socket.addEventListener("open", () => {{
        statusNode.textContent = "Live";
        authPanelNode.hidden = true;
      }});
      socket.addEventListener("close", () => {{
        statusNode.textContent = "Session required";
        promptForSession("Your dashboard session expired or was rejected. Re-authenticate to reconnect.");
      }});
      socket.addEventListener("message", (event) => {{
        updateDashboard(JSON.parse(event.data));
      }});
    }}

    async function requestSession(credential) {{
      const response = await fetch("/auth/session", {{
        method: "POST",
        headers: {{
          "Content-Type": "application/json",
        }},
        body: JSON.stringify({{ credential }}),
      }});
      if (!response.ok) {{
        const body = await response.json().catch(() => ({{ detail: "Authentication failed." }}));
        throw new Error(body.detail || "Authentication failed.");
      }}
      return response.json();
    }}

    async function probeSession() {{
      const response = await fetch("/dashboard/snapshot", {{ credentials: "same-origin" }});
      if (response.ok) {{
        updateDashboard(await response.json());
        connect();
        return true;
      }}
      return false;
    }}

    function promptForSession(message) {{
      if (reconnectTimer) {{
        clearTimeout(reconnectTimer);
        reconnectTimer = undefined;
      }}
      authPanelNode.hidden = false;
      authStatusNode.textContent = message;
    }}

    function ensureCharts(snapshot) {{
      if (!timelineChart) {{
        timelineChart = new Chart(timelineCtx, {{
          type: "line",
          data: {{
            labels: snapshot.charts.timeline.labels,
            datasets: [
              {{ label: "Active tasks", data: snapshot.charts.timeline.active_tasks, borderColor: "#6fd5ff", tension: 0.3 }},
              {{ label: "Token consumption", data: snapshot.charts.timeline.token_consumption_total, borderColor: "#90f0b6", tension: 0.3 }},
              {{ label: "Evolution rate", data: snapshot.charts.timeline.evolution_rate, borderColor: "#f7c46c", tension: 0.3 }},
            ],
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ labels: {{ color: "#e6f1ff" }} }} }},
            scales: {{
              x: {{ ticks: {{ color: "#93a8c7" }}, grid: {{ color: "rgba(255,255,255,0.05)" }} }},
              y: {{ ticks: {{ color: "#93a8c7" }}, grid: {{ color: "rgba(255,255,255,0.05)" }} }},
            }},
          }},
        }});
      }} else {{
        timelineChart.data.labels = snapshot.charts.timeline.labels;
        timelineChart.data.datasets[0].data = snapshot.charts.timeline.active_tasks;
        timelineChart.data.datasets[1].data = snapshot.charts.timeline.token_consumption_total;
        timelineChart.data.datasets[2].data = snapshot.charts.timeline.evolution_rate;
        timelineChart.update("none");
      }}

      if (!personaChart) {{
        personaChart = new Chart(personaCtx, {{
          type: "doughnut",
          data: {{
            labels: snapshot.charts.personas.labels.length ? snapshot.charts.personas.labels : ["unassigned"],
            datasets: [{{
              data: snapshot.charts.personas.values.length ? snapshot.charts.personas.values : [1],
              backgroundColor: ["#6fd5ff", "#90f0b6", "#f7c46c", "#f28f8f", "#b28dff"],
            }}],
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ labels: {{ color: "#e6f1ff" }} }} }},
          }},
        }});
      }} else {{
        personaChart.data.labels = snapshot.charts.personas.labels.length ? snapshot.charts.personas.labels : ["unassigned"];
        personaChart.data.datasets[0].data = snapshot.charts.personas.values.length ? snapshot.charts.personas.values : [1];
        personaChart.update("none");
      }}
    }}

    function updateDashboard(snapshot) {{
      activeTasksNode.textContent = snapshot.active_tasks;
      tokensNode.textContent = snapshot.token_consumption.total.toLocaleString();
      evolutionNode.textContent = `${{snapshot.evolution_rate.toFixed(2)}} / min`;
      personaNode.textContent = snapshot.dominant_persona || "-";
      evolutionStatusNode.textContent = snapshot.last_evolution_status || "idle";
      lastSkillNode.textContent = snapshot.last_evolution_skill ? `Last skill: ${{snapshot.last_evolution_skill}}` : "No skill yet";
      capabilitiesListNode.innerHTML = "";
      (snapshot.capabilities || []).slice(0, 16).forEach((capability) => {{
        const pill = document.createElement("span");
        pill.className = "pill";
        pill.textContent = capability;
        capabilitiesListNode.appendChild(pill);
      }});
      if (!snapshot.capabilities || snapshot.capabilities.length === 0) {{
        capabilitiesListNode.innerHTML = '<span class="pill">No skills registered yet</span>';
      }}
      const personal = snapshot.personal_context || {{}};
      const reminders = (personal.reminders || []).length;
      const preferences = personal.preferences || {{}};
      personalSummaryNode.textContent = `Preferences: ${{Object.keys(preferences).length}} | Pending reminders: ${{reminders}}`;
      const alertParts = [];
      if (snapshot.last_evolution_status && snapshot.last_evolution_status !== "idle") {{
        alertParts.push(`Evolution: ${{snapshot.last_evolution_status}}`);
      }}
      if (reminders > 0) {{
        alertParts.push(`You have ${{reminders}} reminder(s) pending`);
      }}
      alertBannerNode.textContent = alertParts.length ? alertParts.join(" • ") : "System healthy.";
      statusTableNode.innerHTML = "";
      [
        ["Active tasks", snapshot.active_tasks],
        ["Token usage", snapshot.token_consumption.total.toLocaleString()],
        ["Dominant persona", snapshot.dominant_persona || "-"],
        ["Constitution", snapshot.personal_context?.constitution_state || "tracked by Notepad"],
      ].forEach(([label, value]) => {{
        const row = document.createElement("tr");
        row.innerHTML = `<td>${{label}}</td><td>${{value}}</td>`;
        statusTableNode.appendChild(row);
      }});
      ensureCharts(snapshot);
    }}

    authFormNode.addEventListener("submit", async (event) => {{
      event.preventDefault();
      const credential = authCredentialNode.value.trim();
      if (!credential) {{
        promptForSession("Enter the admin token or bot security key to continue.");
        return;
      }}
      authStatusNode.textContent = "Issuing short-lived session...";
      try {{
        await requestSession(credential);
        authCredentialNode.value = "";
        authStatusNode.textContent = "Authenticated. Connecting...";
        authPanelNode.hidden = true;
        connect();
      }} catch (error) {{
        promptForSession(error.message || "Authentication failed.");
      }}
    }});

    probeSession().then((authenticated) => {{
      if (!authenticated) {{
        promptForSession("Enter the current admin token or bot security key to mint a short-lived dashboard session.");
      }}
    }});
  </script>
</body>
</html>"""
