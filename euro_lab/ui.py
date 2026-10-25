from html import escape
import streamlit as st

COLORS = ["#2846a6", "#b1573b", "#637857", "#95769a", "#a08337"]


def style():
    st.html("""<style>
    :root{--navy:#18263c;--paper:#f7f5ef;--rule:#d8d7cf;--blue:#2846a6}
    .block-container{max-width:1320px;padding-top:4.5rem;padding-bottom:4rem}
    h1,h2,h3{font-family:Georgia,'Times New Roman',serif!important;font-weight:400!important;color:var(--navy)}
    h1{font-size:clamp(2.7rem,5vw,4.6rem)!important;line-height:1.03!important;letter-spacing:-.045em!important}
    h2{font-size:2rem!important;letter-spacing:-.025em!important}
    h3{font-size:1.45rem!important}
    p{line-height:1.6} [data-testid="stSidebar"]{border-right:1px solid var(--rule)}
    [data-testid="stMetric"]{border-top:2px solid var(--navy);padding-top:14px}
    [data-testid="stMetricValue"]{font-family:Georgia,serif;font-size:2.4rem}
    [data-testid="stMetricLabel"]{font-size:.8rem}
    .eyebrow{font:600 .72rem/1.5 ui-monospace,monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--blue);margin-bottom:.75rem}
    .intro{font-size:1.12rem;line-height:1.65;max-width:760px;color:#4c5665;margin-bottom:2rem}
    .masthead{padding-bottom:20px;border-bottom:1px solid var(--rule);margin-bottom:24px;display:flex;justify-content:space-between;gap:1rem;font:500 .72rem ui-monospace,monospace;text-transform:uppercase;letter-spacing:.08em}
    .note{padding:1.1rem 1.25rem;background:#eeece4;border-left:3px solid var(--blue);font-size:.94rem;line-height:1.6;margin:1rem 0 1.5rem}
    .side-title{font:400 2.3rem Georgia,serif;letter-spacing:-.05em;line-height:1.05;margin:.5rem 0 .8rem}
    .side-copy{font-size:.82rem;color:#536071;line-height:1.55;margin-bottom:1.5rem}
    .stButton button,.stDownloadButton button{min-height:44px} :focus-visible{outline:2px solid #2846a6!important;outline-offset:3px}
    @media(max-width:640px){.block-container{padding-top:1.5rem}.masthead{flex-wrap:wrap}h1{font-size:2.7rem!important}}
    @media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;animation:none!important;transition:none!important}}
    </style>""")


def header(number, title, description):
    st.html(
        '<div class="masthead"><span>EURO 2024 / Model notes</span><span>Stephen Ahiabah · International football</span></div>'
    )
    st.html(f'<div class="eyebrow">{escape(number)} / Research notebook</div>')
    st.title(title)
    st.html(f'<div class="intro">{escape(description)}</div>')


def note(text):
    st.html(f'<div class="note">{escape(text)}</div>')


def chart(fig, key=None, height=380):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#18263c", size=12),
        colorway=COLORS,
        margin=dict(l=12, r=15, t=45, b=20),
        height=height,
        legend_title_text="",
        title_font=dict(size=17),
        hoverlabel=dict(bgcolor="#ffffff"),
    )
    fig.update_xaxes(gridcolor="#e4e2da", zerolinecolor="#d8d7cf")
    fig.update_yaxes(gridcolor="#e4e2da", zerolinecolor="#d8d7cf")
    st.plotly_chart(
        fig,
        width="stretch",
        key=key,
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
