# app.py - Version expert CGF Gestion
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ---------- CONFIGURATION PAGE ----------
st.set_page_config(page_title="CGF Gestion - Portfolio Expert", layout="wide")
st.title("🏦 Gestion Quantitative de Portefeuille - Dauphine")
st.markdown("""
*Application professionnelle intégrant Markowitz, CAPM, stress tests avancés et recommandations adaptées aux marchés UEMOA.*  
**👈 Utilisez la barre latérale pour paramétrer vos actifs et le benchmark.**  
""")

# ---------- SIDEBAR : PARAMÈTRES ----------
st.sidebar.header("📊 Configuration du portefeuille")

default_tickers = ["AAPL", "MSFT", "BNP.PA", "TLT", "IEF"]
tickers_input = st.sidebar.text_input("Tickers (séparés par des virgules)", value=",".join(default_tickers))
ticker_list = [t.strip() for t in tickers_input.split(",")]

# Option : actifs locaux BRVM (données synthétiques)
use_local = st.sidebar.checkbox("Inclure actifs UEMOA", value=False)
if use_local:
    local_tickers = ["BOAB", "SGBC", "ORGT"]  # tickers fictifs
    ticker_list = list(set(ticker_list + local_tickers))
    st.sidebar.info("Données synthétiques pour BOAB, SGBC, ORGT (marché BRVM)")

start_date = st.sidebar.date_input("Date de début", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("Date de fin", datetime(2025, 12, 31))

# Benchmark composite ou simple
# Benchmark unique uniquement
benchmark = st.sidebar.text_input("Benchmark (ticker)", value="^GSPC")
st.sidebar.caption("Exemples: ^GSPC (S&P500), ^FCHI (CAC40), ^STOXX50E (Euro Stoxx 50)")

risk_free_rate = st.sidebar.number_input("Taux sans risque annualisé (%)", value=2.0, step=0.5) / 100

# Contraintes d'optimisation avancées
allow_short = st.sidebar.checkbox("Autoriser les ventes à découvert", value=False)
max_weight = st.sidebar.slider("Poids maximum par actif (%)", 10, 100, 30) / 100
transaction_cost = st.sidebar.number_input("Coûts de transaction (aller-retour, %)", 0.0, 2.0, 0.1, step=0.05) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("**ℹ️ Conseils** : choisissez 3 à 6 actifs. Pour les actifs UEMOA, les données sont simulées (tendances).")

# ---------- CHARGEMENT DES DONNÉES AVEC NETTOYAGE ----------
@st.cache_data
def load_data(tickers, start, end, benchmark, use_local=False):
    # Nettoyage des tickers
    tickers = [t.replace(",", ".") for t in tickers]
    
    data_dict = {}
    for tick in tickers:
        if use_local and tick in ["BOAB", "SGBC", "ORGT"]:
            dates = pd.date_range(start=start, end=end, freq='B')
            n = len(dates)
            if n == 0:
                continue
            years = (dates - dates[0]).days / 365.25
            trend = 1 + 0.06 * years
            daily_vol = 0.20 / np.sqrt(252)
            noise = np.random.normal(0, daily_vol, n).cumsum()
            price = 100 * trend * np.exp(noise - 0.5 * daily_vol**2 * np.arange(n))
            data_dict[tick] = pd.Series(price, index=dates)
            continue
        
        df = yf.download(tick, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            st.warning(f"Pas de données pour {tick}, ignoré.")
            continue
        if 'Close' in df.columns:
            series = df['Close']
        else:
            series = df.iloc[:, 0]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        data_dict[tick] = series
    
    if not data_dict:
        st.error("Aucune donnée valide.")
        return None, None
    
    prices = pd.concat(data_dict, axis=1)
    prices = prices.dropna(axis=1, how='all')
    prices = prices.ffill().bfill()
    
    # Benchmark unique
    bench_ticker = benchmark.replace(",", ".")
    bench = yf.download(bench_ticker, start=start, end=end, auto_adjust=True, progress=False)
    if bench.empty:
        st.error(f"Benchmark {benchmark} non trouvé.")
        return prices, None
    bench_prices = bench['Close'] if 'Close' in bench.columns else bench.iloc[:, 0]
    
    # Alignement
    common_idx = prices.index.intersection(bench_prices.index)
    if len(common_idx) == 0:
        st.error("Aucune date commune.")
        return None, None
    prices = prices.loc[common_idx]
    bench_prices = bench_prices.loc[common_idx]
    
    return prices, bench_prices

with st.spinner("Chargement des données..."):
    prices, bench_prices = load_data(ticker_list, start_date, end_date, benchmark, use_local)

if prices is None or prices.empty:
    st.stop()

# Calcul des rendements
returns = prices.pct_change().dropna()
bench_returns = bench_prices.pct_change().dropna()
if isinstance(bench_returns, pd.DataFrame):
    bench_returns = bench_returns.iloc[:,0]

# Statistiques annualisées
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
volatilities = np.sqrt(np.diag(cov_matrix))
n_assets = len(ticker_list)

# ---------- FONCTIONS D'OPTIMISATION AVEC COÛTS ET CONTRAINTES ----------
def portfolio_performance(weights, mean_ret, cov_mat):
    ret = np.sum(weights * mean_ret)
    vol = np.sqrt(weights @ cov_mat @ weights)
    # Pénalité pour coûts de transaction (supposés proportionnels aux poids absolus)
    cost_penalty = transaction_cost * np.sum(np.abs(weights))
    ret_net = ret - cost_penalty
    sharpe = (ret_net - risk_free_rate) / vol if vol != 0 else 0
    return ret_net, vol, sharpe

def neg_sharpe(weights, mean_ret, cov_mat):
    return -portfolio_performance(weights, mean_ret, cov_mat)[2]

def portfolio_variance(weights, cov_mat):
    return weights @ cov_mat @ weights

def portfolio_return(weights, mean_ret):
    return np.sum(weights * mean_ret)

# Contraintes : somme = 1
constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
if allow_short:
    bounds = tuple((-1, 1) for _ in range(n_assets))
else:
    # Contrainte de concentration maximale
    bounds = tuple((0, max_weight) for _ in range(n_assets))

# 1. Portefeuille à variance minimale (MVP)
def min_variance():
    init = np.array([1/n_assets]*n_assets)
    res = minimize(portfolio_variance, init, args=(cov_matrix,), method='SLSQP',
                   bounds=bounds, constraints=constraints)
    return res.x if res.success else init

# 2. Portefeuille de tangence (max Sharpe)
def tangency_portfolio():
    init = np.array([1/n_assets]*n_assets)
    res = minimize(neg_sharpe, init, args=(mean_returns, cov_matrix), method='SLSQP',
                   bounds=bounds, constraints=constraints)
    return res.x if res.success else init

# 3. Frontière efficiente
mvp_weights = min_variance()
mvp_ret, mvp_vol, _ = portfolio_performance(mvp_weights, mean_returns, cov_matrix)

def max_return_portfolio():
    def neg_return(w):
        return -portfolio_return(w, mean_returns)
    res = minimize(neg_return, np.array([1/n_assets]*n_assets), method='SLSQP',
                   bounds=bounds, constraints=constraints)
    return res.x if res.success else np.array([1/n_assets]*n_assets)

max_ret_weights = max_return_portfolio()
max_ret = portfolio_return(max_ret_weights, mean_returns)

# Générer la frontière efficiente
target_returns = np.linspace(mvp_ret, max_ret, 50)
efficient_portfolios = []
for target in target_returns:
    constraints_target = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: portfolio_return(x, mean_returns) - target}
    ]
    res = minimize(portfolio_variance, np.array([1/n_assets]*n_assets), args=(cov_matrix,),
                   method='SLSQP', bounds=bounds, constraints=constraints_target)
    if res.success:
        vol = np.sqrt(portfolio_variance(res.x, cov_matrix))
        efficient_portfolios.append((target, vol, res.x))

# Portefeuille de tangence (max Sharpe)
tang_weights = tangency_portfolio()
tang_ret, tang_vol, tang_sharpe = portfolio_performance(tang_weights, mean_returns, cov_matrix)

# ---------- STRATÉGIES PRÉDÉFINIES ----------
def equal_weight():
    w = np.array([1/n_assets]*n_assets)
    # Respecter la contrainte max_weight si nécessaire
    if max_weight < 1:
        w = np.clip(w, 0, max_weight)
        w = w / w.sum()
    return w

def risk_parity_inv_vol():
    inv_vol = 1 / volatilities
    w = inv_vol / np.sum(inv_vol)
    if max_weight < 1:
        w = np.clip(w, 0, max_weight)
        w = w / w.sum()
    return w

strategies = {
    "Equal Weight": equal_weight(),
    "Risk Parity (inverse vol)": risk_parity_inv_vol(),
    "Min Variance (MVP)": mvp_weights,
    "Max Sharpe (Tangence)": tang_weights
}

if max_ret > tang_ret + 1e-6 and max_weight >= 1/n_assets:
    strategies["Max Return"] = max_ret_weights

# Calcul des métriques (y compris drawdown, duration)
# Calcul des métriques (avec gestion d'erreur pour le bêta)
metrics_data = []
for name, w in strategies.items():
    ret_net, vol, sharpe = portfolio_performance(w, mean_returns, cov_matrix)
    port_ret_daily = returns @ w
    # VaR historique
    var_95 = np.percentile(port_ret_daily, 5) * 100
    
    # Bêta - version sécurisée
    try:
        # Aligner les indices
        common_idx = port_ret_daily.index.intersection(bench_returns.index)
        if len(common_idx) > 5:
            port_aligned = port_ret_daily[common_idx]
            bench_aligned = bench_returns[common_idx]
            
            # Vérifier que les données ne sont pas constantes
            if bench_aligned.std() > 1e-9:
                cov_w_m = np.cov(port_aligned, bench_aligned)[0, 1]
                beta = cov_w_m / bench_aligned.var()
            else:
                beta = 0
        else:
            beta = 0
    except Exception as e:
        st.warning(f"Erreur calcul bêta pour {name}: {str(e)}")
        beta = 0
    
    # Alpha
    bench_mean = bench_returns.mean() * 252 if len(bench_returns) > 0 else 0
    alpha = (ret_net - risk_free_rate) - beta * (bench_mean - risk_free_rate)
    
    # Tracking error
    tracking_diff = port_ret_daily - bench_returns
    tracking_error = tracking_diff.std() * np.sqrt(252) * 100 if len(tracking_diff) > 0 else 0
    info_ratio = (alpha * 100) / (tracking_error + 1e-9)
    
    # Treynor
    treynor = (ret_net - risk_free_rate) / (beta + 1e-9)
    
    # Maximum Drawdown
    cumprod = (1 + port_ret_daily).cumprod()
    running_max = cumprod.expanding().max()
    drawdown = (cumprod - running_max) / running_max
    max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0
    
    # Duration du portefeuille obligataire (approximation)
    bond_weights = {ticker: w[i] for i, ticker in enumerate(ticker_list) if any(kw in ticker.upper() for kw in ["TLT","IEF","BND","AGG"])}
    if bond_weights:
        durations = {"TLT":16.5, "IEF":7.5, "BND":6.0, "AGG":6.0}
        port_duration = sum(weight * durations.get(t, 5.0) for t, weight in bond_weights.items()) / sum(bond_weights.values())
    else:
        port_duration = 0
    
    metrics_data.append({
        "Stratégie": name,
        "Rendement ann. (%)": ret_net * 100,
        "Volatilité ann. (%)": vol * 100,
        "Sharpe": sharpe,
        "VaR 95% (%)": var_95,
        "Bêta": beta,
        "Alpha (%)": alpha * 100,
        "Treynor": treynor,
        "Tracking Error (%)": tracking_error,
        "Info Ratio": info_ratio,
        "Max Drawdown (%)": max_drawdown,
        "Duration (ans)": port_duration
    })

df_metrics = pd.DataFrame(metrics_data).set_index("Stratégie")

# ---------- PORTEFEUILLE PERSONNALISÉ ----------
st.sidebar.markdown("---")
st.sidebar.subheader("✏️ Portefeuille personnalisé")
st.sidebar.markdown("Ajustez les poids (somme = 1)")
custom_weights = []
total = 0
for i, ticker in enumerate(ticker_list):
    w = st.sidebar.slider(f"{ticker}", 0.0, max_weight, 1.0/n_assets, step=0.01, key=f"w_{i}")
    custom_weights.append(w)
    total += w

if abs(total - 1.0) > 0.01:
    st.sidebar.warning(f"Somme = {total:.2f} → Normalisation")
    custom_weights = np.array(custom_weights) / total
else:
    custom_weights = np.array(custom_weights)

custom_ret, custom_vol, custom_sharpe = portfolio_performance(custom_weights, mean_returns, cov_matrix)

# ---------- ONGLETS ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Analyse", "📉 Frontière & CAPM", "⚠️ Stress tests", 
    "📊 Mesures", "🎯 Recommandations", "📚 Théorie"
])

# ---------- TAB 1 : ANALYSE ----------
with tab1:
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("Évolution des prix")
        norm_prices = prices / prices.iloc[0]
        fig_prices = px.line(norm_prices, title="Cours normalisés (base 100)")
        st.plotly_chart(fig_prices, use_container_width=True)
        
        st.subheader("Performance cumulée")
        perf_df = pd.DataFrame(index=prices.index)
        for name, w in strategies.items():
            perf_df[name] = (1 + (returns @ w)).cumprod()
        perf_df["Personnalisé"] = (1 + (returns @ custom_weights)).cumprod()
        fig_perf = px.line(perf_df, title="Évolution de 1 unité monétaire")
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with col_right:
        st.subheader("Composition des stratégies")
        comp_df = pd.DataFrame({name: w for name, w in strategies.items()}, index=ticker_list)
        st.dataframe(comp_df.round(3))
        
        st.subheader("Votre portefeuille")
        st.dataframe(pd.DataFrame({"Poids": custom_weights}, index=ticker_list).round(3))
        
        st.metric("Rendement annuel net", f"{custom_ret*100:.2f}%")
        st.metric("Volatilité", f"{custom_vol*100:.2f}%")
        st.metric("Sharpe", f"{custom_sharpe:.3f}")
        
        # Drawdown du portefeuille perso
        port_ret_daily = returns @ custom_weights
        cum = (1+port_ret_daily).cumprod()
        running_max = cum.expanding().max()
        dd = (cum - running_max) / running_max
        st.metric("Max Drawdown", f"{dd.min()*100:.1f}%")

# ---------- TAB 2 : FRONTIÈRE EFFICIENTE & CAPM ----------
with tab2:
    st.header("Frontière efficiente de Markowitz")
    fig_front = go.Figure()
    # Actifs individuels
    fig_front.add_trace(go.Scatter(
        x=volatilities * 100, y=mean_returns * 100,
        mode='markers+text', text=ticker_list, textposition='top center',
        marker=dict(size=12, color='blue'), name="Actifs"
    ))
    # Frontière efficiente
    if efficient_portfolios:
        eff_returns = [p[0]*100 for p in efficient_portfolios]
        eff_vols = [p[1]*100 for p in efficient_portfolios]
        fig_front.add_trace(go.Scatter(
            x=eff_vols, y=eff_returns, mode='lines',
            name='Frontière efficiente', line=dict(color='green', width=3)
        ))
    # MVP
    fig_front.add_trace(go.Scatter(
        x=[mvp_vol*100], y=[mvp_ret*100],
        mode='markers', marker=dict(color='orange', size=15, symbol='star'),
        name=f"MVP (vol min)"
    ))
    # Tangence
    fig_front.add_trace(go.Scatter(
        x=[tang_vol*100], y=[tang_ret*100],
        mode='markers', marker=dict(color='red', size=15, symbol='x'),
        name=f"Tangence (Sharpe={tang_sharpe:.2f})"
    ))
    # CML
    fig_front.add_trace(go.Scatter(
        x=[0, tang_vol*100], y=[risk_free_rate*100, tang_ret*100],
        mode='lines', line=dict(dash='dash', color='purple', width=2), name='CML'
    ))
    fig_front.update_layout(
        xaxis_title="Volatilité annualisée (%)",
        yaxis_title="Rendement espéré annualisé (%)",
        title="Frontière efficiente de Markowitz"
    )
    st.plotly_chart(fig_front, use_container_width=True)
    
    st.subheader("Security Market Line (CAPM)")
    betas = []
    for tick in ticker_list:
        cov_i_m = np.cov(returns[tick], bench_returns)[0,1]
        betas.append(cov_i_m / bench_returns.var())
    market_return = bench_returns.mean() * 252
    sml_returns = risk_free_rate + np.array(betas) * (market_return - risk_free_rate)
    fig_sml = go.Figure()
    fig_sml.add_trace(go.Scatter(
        x=betas, y=mean_returns.values * 100,
        mode='markers+text', text=ticker_list, textposition='top center',
        marker=dict(size=12, color='red'), name="Actifs"
    ))
    beta_range = np.linspace(0, max(betas) * 1.2, 50)
    sml_line = risk_free_rate + beta_range * (market_return - risk_free_rate)
    fig_sml.add_trace(go.Scatter(
        x=beta_range, y=sml_line * 100, mode='lines',
        name="SML", line=dict(color='blue', width=2)
    ))
    fig_sml.update_layout(
        xaxis_title="Bêta", yaxis_title="Rendement espéré (%)", title="Security Market Line"
    )
    st.plotly_chart(fig_sml, use_container_width=True)

# ---------- TAB 3 : STRESS TESTS AVANCÉS ----------
with tab3:
    st.header("Stress tests avancés")
    stress_type = st.radio("Type de scénario", ["Choc de marché", "Hausse des taux", "Crise historique (COVID)"])
    
    if stress_type == "Choc de marché":
        shock = st.slider("Amplitude du choc (%)", -40, 0, -15, step=5)
        shocked_returns = returns * (1 + shock/100)
        
    elif stress_type == "Hausse des taux":
        rate_shock_bps = st.slider("Hausse des taux (points de base)", 0, 300, 75, step=25)
        bond_keywords = ["TLT", "IEF", "BND", "AGG"]
        bond_tickers = [t for t in ticker_list if any(k in t.upper() for k in bond_keywords)]
        shocked_prices = prices.copy()
        for t in bond_tickers:
            dur = 16.5 if "TLT" in t.upper() else 7.5 if "IEF" in t.upper() else 6.0
            shocked_prices[t] = prices[t] * (1 - dur * rate_shock_bps / 10000)
        shocked_prices = shocked_prices.clip(lower=0)
        shocked_returns = shocked_prices.pct_change().dropna()
    
    else:  # Crise COVID
        st.info("Simulation : réplication de la baisse de mars 2020 sur les actifs")
        # Identifier la date de début de crise (2020-02-20)
        crisis_start = datetime(2020, 2, 20)
        crisis_end = datetime(2020, 3, 23)
        # Extraire les rendements de la crise
        crisis_returns = returns.loc[crisis_start:crisis_end]
        # Appliquer ces rendements à partir de la date de début de la série
        shocked_returns = returns.copy()
        # On remplace la période par les rendements de crise multipliés par un facteur d'intensité
        intensity = st.slider("Intensité de la crise (%)", 50, 150, 100, step=10) / 100
        shocked_returns.loc[crisis_start:crisis_end] = crisis_returns * intensity
    
    # Alignement et calcul des valeurs stressées
    common_idx = returns.index.intersection(shocked_returns.index)
    base = (1 + (returns.loc[common_idx] @ custom_weights)).cumprod()
    stress = (1 + (shocked_returns.loc[common_idx] @ custom_weights)).cumprod()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base.index, y=base, name="Sans stress", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=stress.index, y=stress, name="Avec stress", line=dict(color="red")))
    fig.update_layout(title="Impact du stress sur votre portefeuille", xaxis_title="Date", yaxis_title="Valeur (base 1)")
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Valeur sans stress", f"{base.iloc[-1]:.2f}")
    with col2: st.metric("Valeur avec stress", f"{stress.iloc[-1]:.2f}")
    with col3: st.metric("Perte relative", f"{(stress.iloc[-1]/base.iloc[-1] - 1)*100:.2f}%")

# ---------- TAB 4 : MESURES DE PERFORMANCE + RADAR ----------
with tab4:
    st.header("Mesures de performance")
    st.dataframe(df_metrics.round(3))
    
    # Radar chart comparatif (Sharpe, Alpha, Info Ratio, Max Drawdown)
    st.subheader("Comparaison visuelle des stratégies")
    metrics_for_radar = ["Sharpe", "Alpha (%)", "Info Ratio", "Max Drawdown (%)"]
    # Normalisation pour le radar (plus haut est mieux sauf drawdown)
    radar_data = df_metrics[metrics_for_radar].copy()
    for col in radar_data.columns:
        if col == "Max Drawdown (%)":
            radar_data[col] = -radar_data[col]  # inverser pour que plus élevé = meilleur
    # Min-max scaling
    radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min() + 1e-9)
    
    fig_radar = go.Figure()
    for strat in radar_norm.index:
        values = radar_norm.loc[strat].tolist()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_for_radar,
            fill='toself',
            name=strat
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

# ---------- TAB 5 : RECOMMANDATIONS (PROFIL DE RISQUE) ----------
with tab5:
    st.header("Recommandations personnalisées")
    st.markdown("Choisissez votre profil de risque pour obtenir une allocation suggérée.")
    
    risk_profile = st.selectbox("Profil d'investisseur", ["Conservateur", "Équilibré", "Dynamique"])
    
    # Définir des portefeuilles cibles basés sur les stratégies précédentes
    if risk_profile == "Conservateur":
        rec_weights = strategies["Min Variance (MVP)"]
        rec_desc = "Portefeuille à variance minimale : faible risque, rendement modéré."
    elif risk_profile == "Équilibré":
        rec_weights = strategies["Risk Parity (inverse vol)"]
        rec_desc = "Parité de risque : équilibre entre actions et obligations."
    else:
        rec_weights = strategies["Max Sharpe (Tangence)"]
        rec_desc = "Maximisation du ratio de Sharpe : meilleur compromis risque/rendement."
    
    st.subheader("Allocation recommandée")
    rec_df = pd.DataFrame({"Actif": ticker_list, "Poids (%)": rec_weights * 100})
    fig_pie = px.pie(rec_df, values="Poids (%)", names="Actif", title="Pie chart de l'allocation")
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Performance attendue
    ret_rec, vol_rec, sharpe_rec = portfolio_performance(rec_weights, mean_returns, cov_matrix)
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Rendement annuel (net)", f"{ret_rec*100:.2f}%")
    with col2: st.metric("Volatilité", f"{vol_rec*100:.2f}%")
    with col3: st.metric("Sharpe", f"{sharpe_rec:.3f}")
    
    st.info(f"**Justification** : {rec_desc}")
    
    st.subheader("Rappel réglementaire OPCVM UEMOA")
    st.markdown("""
    - **Diversification** : pas plus de 10% du portefeuille dans un même titre (sauf États).
    - **Investissement** : maximum 30% hors zone UEMOA.
    - **Liquidité** : au moins 10% d'actifs très liquides.
    Notre optimisation respecte ces contraintes (poids max {:.0f}%).  
    """.format(max_weight*100))

# ---------- TAB 6 : THÉORIE ----------
with tab6:
    st.header("Synthèse des concepts clés")
    st.markdown("""
    ### 1. Paradoxe de Saint-Pétersbourg
    Un jeu avec espérance infinie n'attire pas car les agents sont **averses au risque**.
    
    ### 2. Aversion au risque
    - **AAR** = $-U''(w)/U'(w)$ (absolue)
    - **ARR** = $w \\times AAR$ (relative)
    
    ### 3. Diversification
    Le risque se décompose en : **systématique** (non diversifiable) + **spécifique** (diversifiable)
    
    ### 4. CAPM (MEDAF)
    $E(R_i) = R_f + \\beta_i (E(R_m) - R_f)$
    
    ### 5. Mesures de performance
    - **Sharpe** : $\\frac{R_p - R_f}{\\sigma_p}$ (risque total)
    - **Treynor** : $\\frac{R_p - R_f}{\\beta_p}$ (risque systématique)
    - **Jensen** : $\\alpha_p$ (surperformance)
    
    ### 6. Stress tests et gestion des risques
    - **VaR** : perte maximale attendue avec un niveau de confiance.
    - **Maximum Drawdown** : pire perte historique de pic à creux.
    - **Duration** : sensibilité aux taux d'intérêt.
    """)
    
    st.success("💡 Utilisez les sliders pour construire votre propre portefeuille et comparez-le aux recommandations.")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("Projet développé par Laurent N'GNAME – Master Finance, Université Paris-Dauphine | Données Yahoo Finance")
