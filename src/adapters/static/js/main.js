document.addEventListener('DOMContentLoaded', () => {
    // === Theme Handling ===
    const initTheme = () => {
        const savedTheme = localStorage.getItem('theme') || 'dark'; // Default to dark for "premium" feel
        document.documentElement.setAttribute('data-bs-theme', savedTheme);
        return savedTheme;
    };

    // === Smooth Scroll ===
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // === Plot Resizing ===
    const resizeObserver = new ResizeObserver((entries) => {
        for (let entry of entries) {
            const plotDiv = entry.target.querySelector('.js-plotly-plot');
            if (plotDiv) {
                Plotly.Plots.resize(plotDiv);
            }
        }
    });

    document.querySelectorAll('.plot-container').forEach(el => {
        resizeObserver.observe(el);
    });
});

// === Plotly Layout Helper ===
// Exposed to global scope for templates to use
window.getPlotlyLayout = () => {
    const isDark = document.documentElement.getAttribute('data-bs-theme') !== 'light';

    // Midnight Theme Colors
    const bgColor = isDark ? 'rgba(0,0,0,0)' : '#ffffff';
    const textColor = isDark ? '#f8fafc' : '#0f172a';
    const gridColor = isDark ? 'rgba(148, 163, 184, 0.1)' : 'rgba(0, 0, 0, 0.05)';

    return {
        paper_bgcolor: bgColor,
        plot_bgcolor: bgColor,
        font: {
            family: 'Inter, sans-serif',
            color: textColor
        },
        xaxis: {
            gridcolor: gridColor,
            zerolinecolor: gridColor
        },
        yaxis: {
            gridcolor: gridColor,
            zerolinecolor: gridColor
        },
        margin: { t: 40, r: 20, b: 40, l: 60 }
    };
};
