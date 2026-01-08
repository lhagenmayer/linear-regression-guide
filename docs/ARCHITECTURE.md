# 🏗️ Architektur - 100% Plattform-Agnostisch

Diese Anwendung ist **vollständig plattform-agnostisch** und kann mit **jedem Frontend** verwendet werden:

- ✅ Flask (Python Server-Rendered HTML)
- ✅ Streamlit (Python Interactive UI)
- ✅ Next.js / React
- ✅ Vite / Vue.js
- ✅ Angular / Svelte
- ✅ Mobile Apps (iOS/Android)
- ✅ Jeder HTTP-Client

## 📐 Architektur-Übersicht

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTENDS                                       │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │
│  │  Next.js  │  │   Vite    │  │   Vue     │  │ Angular   │  │  Mobile   │ │
│  │  /React   │  │  /React   │  │           │  │           │  │ (iOS/And) │ │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘ │
│        │              │              │              │              │        │
│        └──────────────┴──────────────┼──────────────┴──────────────┘        │
│                                      │ HTTP/JSON                            │
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                           REST API LAYER                                      │
│                        /src/api/ (Pure JSON)                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  /api/regression/simple      POST  → Run simple regression             │  │
│  │  /api/regression/multiple    POST  → Run multiple regression           │  │
│  │  /api/content/simple         POST  → Get educational content           │  │
│  │  /api/content/multiple       POST  → Get educational content           │  │
│  │  /api/content/schema         GET   → Get content structure schema      │  │
│  │  /api/ai/interpret           POST  → AI interpretation                 │  │
│  │  /api/datasets               GET   → List available datasets           │  │
│  │  /api/openapi.json           GET   → OpenAPI specification             │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                          CORE LAYER (Pure Python)                             │
│                         Framework-Agnostic Logic                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │
│  │    Pipeline     │  │    Content      │  │      AI         │               │
│  │   /pipeline/    │  │   /content/     │  │     /ai/        │               │
│  │                 │  │                 │  │                 │               │
│  │  • DataFetcher  │  │  • Structure    │  │  • Perplexity   │               │
│  │  • Calculator   │  │  • Builder      │  │    Client       │               │
│  │  • PlotBuilder  │  │  • Simple       │  │  • Response     │               │
│  │  • Serializers  │  │  • Multiple     │  │  • Caching      │               │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘               │
│                                │                                              │
│                     All outputs are JSON-serializable                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 🔌 Integration Beispiele

### Next.js / React

```typescript
// lib/api.ts
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function runSimpleRegression(params: {
  dataset?: string;
  n?: number;
  noise?: number;
  seed?: number;
}) {
  const response = await fetch(`${API_URL}/api/regression/simple`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return response.json();
}

export async function getEducationalContent(params: {
  dataset?: string;
  n?: number;
}) {
  const response = await fetch(`${API_URL}/api/content/simple`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return response.json();
}
```

```tsx
// components/RegressionChart.tsx
import Plotly from 'react-plotly.js';

export function RegressionChart({ plotData }: { plotData: any }) {
  return (
    <Plotly
      data={plotData.data}
      layout={plotData.layout}
    />
  );
}
```

### Vue.js / Vite

```typescript
// composables/useRegression.ts
import { ref } from 'vue';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export function useRegression() {
  const loading = ref(false);
  const result = ref(null);
  const content = ref(null);

  async function runAnalysis(params: any) {
    loading.value = true;
    try {
      const response = await fetch(`${API_URL}/api/content/simple`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      const data = await response.json();
      result.value = data.data?.stats;
      content.value = data.data?.content;
    } finally {
      loading.value = false;
    }
  }

  return { loading, result, content, runAnalysis };
}
```

### Vanilla JavaScript / Any Framework

```javascript
// Einfacher API-Aufruf mit fetch
async function analyze(dataset = 'electronics', n = 50) {
  const response = await fetch('http://localhost:8000/api/content/simple', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset, n }),
  });
  
  const { success, content, plots, stats } = await response.json();
  
  if (success) {
    // content.chapters - Educational content structure
    // plots.scatter    - Plotly figure (JSON)
    // stats            - Statistical results
    
    // Render plot with Plotly.js
    Plotly.newPlot('chart', plots.scatter.data, plots.scatter.layout);
    
    // Render content
    renderContent(content);
  }
}

function renderContent(content) {
  // Iterate through chapters and render based on element type
  content.chapters.forEach(chapter => {
    chapter.sections.forEach(section => {
      // Handle each element type: markdown, metric, formula, plot, etc.
      renderElement(section);
    });
  });
}
```

## 📦 Datenstrukturen

### Content Schema

Alle educational content Elemente folgen dieser Struktur:

```typescript
interface EducationalContent {
  title: string;
  subtitle: string;
  chapters: Chapter[];
}

interface Chapter {
  type: 'chapter';
  number: string;
  title: string;
  icon: string;
  sections: (Section | ContentElement)[];
}

interface Section {
  type: 'section';
  title: string;
  icon: string;
  content: ContentElement[];
}

// Content Element Types
type ContentElement = 
  | { type: 'markdown'; text: string }
  | { type: 'metric'; label: string; value: string; help_text?: string; delta?: string }
  | { type: 'metric_row'; metrics: Metric[] }
  | { type: 'formula'; latex: string; inline?: boolean }
  | { type: 'plot'; plot_key: string; title?: string; description?: string; height?: number }
  | { type: 'table'; headers: string[]; rows: string[][]; caption?: string }
  | { type: 'columns'; columns: ContentElement[][]; widths?: number[] }
  | { type: 'expander'; title: string; content: ContentElement[]; expanded?: boolean }
  | { type: 'info_box'; content: string }
  | { type: 'warning_box'; content: string }
  | { type: 'success_box'; content: string }
  | { type: 'code_block'; code: string; language?: string }
  | { type: 'divider' };
```

### API Response Format

```typescript
interface APIResponse {
  success: boolean;
  data?: {
    content: EducationalContent;
    plots: {
      scatter: PlotlyFigure;
      residuals: PlotlyFigure;
      diagnostics: PlotlyFigure;
      extra?: Record<string, PlotlyFigure>;
    };
    stats: {
      type: string;
      coefficients: { intercept: number; slope: number };
      model_fit: { r_squared: number; r_squared_adj: number };
      // ... more fields
    };
    data: {
      type: string;
      x: number[];
      y: number[];
      n: number;
      // ... more fields
    };
  };
  error?: string;
}
```

## 🚀 Server Starten

### REST API (für externe Frontends)

```bash
# Startet den API-Server auf Port 8000
python run.py --api

# Mit benutzerdefiniertem Port
python run.py --api --port 3001

# Mit FastAPI (falls installiert) für automatische OpenAPI Docs
pip install fastapi uvicorn
python run.py --api
# → Swagger UI: http://localhost:8000/docs
```

### Flask Web App (HTML Rendering)

```bash
# Startet Flask mit Server-Side Rendering
python run.py --flask --port 5000
```

### Streamlit (Interactive Python UI)

```bash
# Startet Streamlit
streamlit run run.py
```

## 🔧 Architektur-Prinzipien

### 1. Strikte Trennung von Concerns

```
Core Logic (Pure Python)     →  JSON-serialisierbar
     ↓
API Layer (REST)             →  Framework-agnostisch
     ↓
Adapters (Framework-spezifisch)  →  Flask/Streamlit/etc.
```

### 2. Alle Daten sind JSON-serialisierbar

- **Numpy Arrays** → Listen (`array.tolist()`)
- **Plotly Figures** → JSON (`fig.to_json()`)
- **Dataclasses** → Dict (`to_dict()` Methoden)
- **Content Elements** → Strukturierte Dicts

### 3. Keine Framework-Imports im Core

Der gesamte `/src/pipeline/`, `/src/content/`, und `/src/ai/` Code hat **keine** Imports von:
- `streamlit`
- `flask`
- `jinja2`
- Andere UI-Frameworks

### 4. Lazy Loading für Adapters

```python
# In endpoints.py
@property
def pipeline(self):
    """Lazy load to avoid import issues."""
    if self._pipeline is None:
        from ..pipeline import RegressionPipeline
        self._pipeline = RegressionPipeline()
    return self._pipeline
```

## 📊 Modul-Struktur

```
src/
├── api/                    # REST API Layer (100% agnostisch)
│   ├── __init__.py
│   ├── endpoints.py        # Business logic endpoints
│   ├── serializers.py      # JSON serialization
│   └── server.py           # Flask/FastAPI server
│
├── pipeline/               # Core Pipeline (100% agnostisch)
│   ├── get_data.py         # Data fetching
│   ├── calculate.py        # Statistics calculation
│   ├── plot.py             # Plotly figure generation
│   └── regression_pipeline.py
│
├── content/                # Content Layer (100% agnostisch)
│   ├── structure.py        # Content element dataclasses
│   ├── builder.py          # Abstract content builder
│   ├── simple_regression.py    # Simple regression content
│   └── multiple_regression.py  # Multiple regression content
│
├── ai/                     # AI Layer (100% agnostisch)
│   ├── perplexity_client.py    # API client
│   └── ui_components.py    # (optional, für Adapter)
│
└── adapters/               # Framework-spezifische Adapter
    ├── flask_app.py        # Flask mit HTML templates
    ├── streamlit/
    │   └── app.py          # Streamlit UI
    └── renderers/
        ├── html_renderer.py      # Content → HTML
        └── streamlit_renderer.py # Content → Streamlit
```

## 🌐 CORS Konfiguration

Der API-Server erlaubt standardmäßig alle Origins (`*`). Für Produktion:

```python
# In run.py oder beim Server-Start
from src.api import create_api_server

app = create_api_server(cors_origins=[
    "https://your-frontend.com",
    "http://localhost:3000",  # Next.js dev
    "http://localhost:5173",  # Vite dev
])
```

## 🧪 API Testen

```bash
# Health Check
curl http://localhost:8000/api/health

# Simple Regression
curl -X POST http://localhost:8000/api/regression/simple \
  -H "Content-Type: application/json" \
  -d '{"dataset": "electronics", "n": 50}'

# Educational Content
curl -X POST http://localhost:8000/api/content/simple \
  -H "Content-Type: application/json" \
  -d '{"dataset": "electronics", "n": 50}'

# AI Interpretation
curl -X POST http://localhost:8000/api/ai/interpret \
  -H "Content-Type: application/json" \
  -d '{"stats": {"intercept": 0.5, "slope": 0.3, "r_squared": 0.85}}'

# Available Datasets
curl http://localhost:8000/api/datasets
```
