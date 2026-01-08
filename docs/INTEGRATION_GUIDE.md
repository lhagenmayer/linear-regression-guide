# 🚀 Integration Guide

**5 Minuten zur Integration der Regression Analysis API in dein Projekt**

---

## 1️⃣ API Server starten

```bash
# Im Repository-Verzeichnis
python3 run.py --api --port 8000
```

Öffne http://localhost:8000/api/docs für die interaktive Dokumentation.

---

## 2️⃣ Schnelltest

```bash
# Health Check
curl http://localhost:8000/api/health

# Einfache Regression
curl -X POST http://localhost:8000/api/regression/simple \
  -H "Content-Type: application/json" \
  -d '{"dataset": "electronics", "n": 50}'
```

---

## 3️⃣ Integration nach Framework

### Next.js / React

```bash
npm install react-plotly.js plotly.js
```

```typescript
// lib/regression.ts
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function runRegression(params: { dataset?: string; n?: number }) {
  const res = await fetch(`${API_URL}/api/regression/simple`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}

export async function getContent(params: { dataset?: string; n?: number }) {
  const res = await fetch(`${API_URL}/api/content/simple`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}
```

```tsx
// components/Chart.tsx
'use client';
import dynamic from 'next/dynamic';
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export function RegressionChart({ plotData }: { plotData: any }) {
  return <Plot data={plotData.data} layout={plotData.layout} />;
}
```

```tsx
// app/page.tsx
import { getContent } from '@/lib/regression';
import { RegressionChart } from '@/components/Chart';

export default async function Page() {
  const { content, plots, stats } = await getContent({ n: 50 });
  
  return (
    <div>
      <h1>{content.title}</h1>
      <p>R² = {stats.model_fit.r_squared.toFixed(4)}</p>
      <RegressionChart plotData={plots.scatter} />
    </div>
  );
}
```

---

### Vue 3 / Nuxt

```bash
npm install vue-plotly.js
```

```vue
<!-- components/Regression.vue -->
<template>
  <div>
    <h2>{{ content?.title }}</h2>
    <p v-if="stats">R² = {{ stats.model_fit.r_squared.toFixed(4) }}</p>
    <div ref="plotEl" style="width: 100%; height: 400px;"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';
import Plotly from 'plotly.js-dist';

const props = defineProps(['dataset', 'n']);
const content = ref(null);
const stats = ref(null);
const plots = ref(null);
const plotEl = ref(null);

async function load() {
  const res = await fetch('http://localhost:8000/api/content/simple', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset: props.dataset, n: props.n }),
  });
  const data = await res.json();
  content.value = data.content;
  stats.value = data.stats;
  plots.value = data.plots;
  
  if (plotEl.value && plots.value?.scatter) {
    Plotly.newPlot(plotEl.value, plots.value.scatter.data, plots.value.scatter.layout);
  }
}

onMounted(load);
watch(() => [props.dataset, props.n], load);
</script>
```

---

### Vanilla JavaScript

```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
  <div id="chart"></div>
  <script>
    fetch('http://localhost:8000/api/content/simple', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset: 'electronics', n: 50 })
    })
    .then(r => r.json())
    .then(({ content, plots, stats }) => {
      document.body.innerHTML = `
        <h1>${content.title}</h1>
        <p>R² = ${stats.model_fit.r_squared.toFixed(4)}</p>
        <div id="chart"></div>
      `;
      Plotly.newPlot('chart', plots.scatter.data, plots.scatter.layout);
    });
  </script>
</body>
</html>
```

---

### Python

```python
import requests

API_URL = "http://localhost:8000"

# Regression
r = requests.post(f"{API_URL}/api/regression/simple", json={"n": 100})
data = r.json()["data"]
print(f"R² = {data['stats']['model_fit']['r_squared']:.4f}")

# AI Interpretation
r = requests.post(f"{API_URL}/api/ai/interpret", json={"stats": data["stats"]})
print(r.json()["interpretation"]["content"])
```

---

## 4️⃣ Error-Handling

### Error-Response-Struktur

Alle API-Endpunkte geben bei Fehlern folgendes Format zurück:

```json
{
  "success": false,
  "error": "Beschreibung des Fehlers",
  "error_id": "a1b2c3d4",
  "error_code": "VALIDATION_ERROR",
  "details": {
    "validation_errors": [
      {
        "loc": ["n"],
        "msg": "ensure this value is greater than or equal to 2",
        "type": "value_error.number.not_ge"
      }
    ]
  }
}
```

### Error-Handling in verschiedenen Frameworks

**Next.js / React:**

```typescript
async function runRegression(params: { dataset?: string; n?: number }) {
  try {
    const res = await fetch(`${API_URL}/api/regression/simple`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    
    const data = await res.json();
    
    if (!data.success) {
      // Log error for debugging
      console.error(`Error ID: ${data.error_id}`, data);
      
      // Show user-friendly message
      throw new Error(data.error || 'Unknown error occurred');
    }
    
    return data.data;
  } catch (error) {
    // Handle network errors
    console.error('Network error:', error);
    throw error;
  }
}
```

**Vue 3:**

```vue
<script setup>
import { ref } from 'vue';

const error = ref(null);
const errorId = ref(null);

async function loadData() {
  try {
    error.value = null;
    const res = await fetch('http://localhost:8000/api/regression/simple', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset: 'electronics', n: 50 })
    });
    
    const data = await res.json();
    
    if (!data.success) {
      error.value = data.error;
      errorId.value = data.error_id;
      console.error(`Error ID: ${errorId.value}`, data);
      return;
    }
    
    // Process data...
  } catch (e) {
    error.value = e.message;
  }
}
</script>

<template>
  <div v-if="error" class="error">
    <p>{{ error }}</p>
    <p v-if="errorId" class="text-sm">Error ID: {{ errorId }}</p>
  </div>
</template>
```

**Python:**

```python
import requests

def run_regression(dataset='electronics', n=50):
    try:
        response = requests.post(
            'http://localhost:8000/api/regression/simple',
            json={'dataset': dataset, 'n': n}
        )
        data = response.json()
        
        if not data.get('success'):
            error_id = data.get('error_id')
            error_code = data.get('error_code')
            print(f"Error ID: {error_id}, Code: {error_code}")
            print(f"Error: {data.get('error')}")
            raise Exception(data.get('error', 'Unknown error'))
        
        return data['data']
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        raise
```

## 5️⃣ Response-Struktur verstehen

### Stats

```json
{
  "coefficients": { "intercept": 0.66, "slope": 0.51 },
  "model_fit": { "r_squared": 0.91, "r_squared_adj": 0.90 },
  "t_tests": {
    "intercept": { "t_value": 4.80, "p_value": 0.00002 },
    "slope": { "t_value": 20.77, "p_value": 0.00000 }
  },
  "sample": { "n": 50, "df": 48 }
}
```

### Plots

Plotly JSON Format - direkt mit `Plotly.newPlot(el, data.data, data.layout)` rendern.

### Content

Strukturierte Kapitel mit Elements:
- `markdown` - Text
- `formula` - LaTeX
- `plot` - Plot-Referenz (key in plots)
- `metric` / `metric_row` - KPIs
- `table` - Tabellen
- `expander` - Aufklappbar
- `info_box` / `warning_box` / `success_box` - Hinweise

---

## 6️⃣ AI Interpretation

```bash
curl -X POST http://localhost:8000/api/ai/interpret \
  -H "Content-Type: application/json" \
  -d '{
    "stats": {
      "intercept": 0.66, "slope": 0.51, "r_squared": 0.91,
      "t_slope": 20.77, "p_slope": 0.00000, "n": 50,
      "x_label": "Verkaufsfläche", "y_label": "Umsatz"
    }
  }'
```

Ohne API-Key wird eine Fallback-Interpretation zurückgegeben.

---

## 📖 Weiterführende Dokumentation

- **[API.md](API.md)** - Vollständige API-Referenz
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Architektur-Details
- **[openapi.yaml](openapi.yaml)** - OpenAPI Specification
- **http://localhost:8000/api/docs** - Swagger UI (live)
