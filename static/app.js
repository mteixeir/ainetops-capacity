// ── Mostrar/ocultar el uploader de CSV según el modo seleccionado ──
document.querySelectorAll('input[name="modo"]').forEach(radio => {
  radio.addEventListener("change", e => {
    const csvDiv = document.getElementById("csv-upload");
    csvDiv.classList.toggle("oculto", e.target.value !== "csv");
  });
});

// ── Función principal: ejecuta el forecast ──
async function ejecutarForecast() {
  const modo        = document.querySelector('input[name="modo"]:checked').value;
  const btnForecast = document.getElementById("btn-forecast");
  const loading     = document.getElementById("loading");

  // Mostrar spinner, ocultar secciones previas
  btnForecast.disabled = true;
  loading.classList.remove("oculto");
  ["colapso-cards","graficas","seccion-ia"].forEach(id =>
    document.getElementById(id).classList.add("oculto")
  );

  try {
    // Construir el FormData (datos del formulario) para enviar al backend
    const form = new FormData();
    if (modo === "csv") {
      const archivo = document.getElementById("archivo-csv").files[0];
      if (!archivo) { alert("Por favor selecciona un archivo CSV."); return; }
      form.append("file", archivo);
      form.append("usar_sintetico", "false");
    } else {
      form.append("usar_sintetico", "true");
    }

    // Llamar al endpoint (punto de conexión) del backend
    const resp = await fetch("/api/forecast", { method: "POST", body: form });
    if (!resp.ok) throw new Error(`Error del servidor: ${resp.status}`);
    const data = await resp.json();

    if (data.error) { alert(data.error); return; }

    // Llenar tarjetas de colapso
    document.getElementById("fecha-cpu").textContent     = data.fechas_colapso.cpu;
    document.getElementById("fecha-ram").textContent     = data.fechas_colapso.ram;
    document.getElementById("fecha-trafico").textContent = data.fechas_colapso.trafico;
    document.getElementById("colapso-cards").classList.remove("oculto");

    // Dibujar las tres gráficas
    const colores = {
      cpu:     { hist: "#e5484d", pred: "#ff8080", banda: "rgba(229,72,77,0.15)" },
      ram:     { hist: "#f97316", pred: "#ffb07c", banda: "rgba(249,115,22,0.15)" },
      trafico: { hist: "#38bdf8", pred: "#93d9f7", banda: "rgba(56,189,248,0.15)" }
    };
    const titulos = {
      cpu: "CPU (Unidad Central de Procesamiento) – Utilización %",
      ram: "RAM (Memoria de Acceso Aleatorio) – Utilización %",
      trafico: "Tráfico de Red – Utilización del enlace %"
    };

    ["cpu","ram","trafico"].forEach(metrica => {
      const hist = data.historico;
      const pred = data.forecast[metrica];
      const c    = colores[metrica];

      // Separar datos históricos de los pronosticados
      const fechaCorte = hist[hist.length - 1].ds;
      const predFuturo = pred.filter(r => r.ds > fechaCorte);

      const trazas = [
        // Banda de confianza al 95%
        {
          x: [...predFuturo.map(r => r.ds), ...predFuturo.map(r => r.ds).reverse()],
          y: [...predFuturo.map(r => r.yhat_upper), ...predFuturo.map(r => r.yhat_lower).reverse()],
          fill: "toself", fillcolor: c.banda,
          line: { color: "transparent" },
          name: "Intervalo de confianza 95%",
          type: "scatter", mode: "lines", hoverinfo: "skip"
        },
        // Línea histórica
        {
          x: hist.map(r => r.ds),
          y: hist.map(r => r[metrica]),
          name: "Histórico",
          type: "scatter", mode: "lines+markers",
          line: { color: c.hist, width: 2.5 },
          marker: { size: 5 }
        },
        // Línea de pronóstico
        {
          x: predFuturo.map(r => r.ds),
          y: predFuturo.map(r => r.yhat),
          name: "Pronóstico",
          type: "scatter", mode: "lines",
          line: { color: c.pred, width: 2, dash: "dash" }
        },
        // Línea de umbral crítico 90%
        {
          x: [hist[0].ds, predFuturo.at(-1)?.ds ?? hist.at(-1).ds],
          y: [90, 90],
          name: "Umbral crítico 90%",
          type: "scatter", mode: "lines",
          line: { color: "#facc15", width: 1.5, dash: "dot" }
        }
      ];

      const layout = {
        title: { text: titulos[metrica], font: { color: "#e6edf3", size: 14 } },
        paper_bgcolor: "#161b22", plot_bgcolor: "#0d1117",
        font: { color: "#8b949e" },
        xaxis: { gridcolor: "#21262d", showgrid: true },
        yaxis: { gridcolor: "#21262d", range: [0, 105], ticksuffix: "%" },
        legend: { orientation: "h", y: -0.2, font: { size: 11 } },
        margin: { t: 50, b: 60, l: 50, r: 20 }
      };

      Plotly.newPlot(`grafica-${metrica}`, trazas, layout, { responsive: true });
    });

    document.getElementById("graficas").classList.remove("oculto");

    // Mostrar resumen ejecutivo de la IA
    document.getElementById("resumen-ia").textContent = data.resumen_ia;
    document.getElementById("seccion-ia").classList.remove("oculto");

  } catch (err) {
    alert("Error al generar el forecast: " + err.message);
    console.error(err);
  } finally {
    loading.classList.add("oculto");
    btnForecast.disabled = false;
  }
}