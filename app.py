from datetime import date, timedelta
import json
import pandas as pd
import streamlit as st
from predictor import ROOT, load_matches, predict
from ui_tables import metric_table

st.set_page_config(page_title='Premier League Predictor', page_icon='⚽', layout='wide')
st.title('Premier League match predictor')
st.caption('Historical team form · Home and away performance · Elo strength')
if not (ROOT / 'artifacts/model.joblib').exists():
    st.info('Run python predictor.py download, then python predictor.py train to prepare the model.')
    st.stop()
matches = load_matches()
report = json.loads((ROOT / 'artifacts/evaluation.json').read_text())
latest = matches.Date.max().date()
st.caption(f'Data through {latest} · Model: {report["selected_model"]} · Premier League matches only')
teams = sorted(set(matches.HomeTeam) | set(matches.AwayTeam))
with st.form('fixture'):
    left, right, when = st.columns(3)
    home = left.selectbox('Home team', teams, index=teams.index('Arsenal'))
    away = right.selectbox('Away team', teams, index=teams.index('Chelsea'))
    fixture_date = when.date_input('Match date', value=max(date.today(), latest+timedelta(days=1)), min_value=latest+timedelta(days=1))
    submitted = st.form_submit_button('Predict match', type='primary')
if submitted:
    try:
        result = predict(home, away, fixture_date)
        if result['data_age_days'] > 14:
            st.warning(f'The latest results are {result["data_age_days"]} days before this fixture. Refresh data and retrain for current form.')
        if min(result['recent_matches'].values()) < 5:
            st.warning('At least one team has fewer than five matches in the preceding year; its form estimate is limited.')
        for col, (label, probability) in zip(st.columns(3), result['probabilities'].items()):
            col.metric(label, f'{probability:.1%}')
        st.bar_chart(pd.Series(result['probabilities'], name='Probability'))
        s = result['predicted_stats']
        st.subheader('Predicted match statistics')
        st.dataframe(pd.DataFrame({'Team': [home, away], 'Goals': [s['FTHG'], s['FTAG']],
            'Shots': [s['HS'], s['AS']], 'Shots on target': [s['HST'], s['AST']]}).round(2), hide_index=True)
        st.caption('Statistical averages, not an exact score forecast. This model does not include injuries, lineups, possession or cup fixtures.')
        if result.get('poisson'):
            with st.expander('Goal-based model comparison'):
                st.write('Independent Poisson goal distributions include all scorelines when calculating win, draw and loss probabilities.')
                st.dataframe(pd.DataFrame({'Selected model': result['probabilities'], 'Poisson model': result['poisson']['probabilities']}).style.format('{:.1%}'))
                st.write(f"Poisson goal averages: {home} {result['poisson']['goal_rates'][0]:.2f}, {away} {result['poisson']['goal_rates'][1]:.2f}")
    except ValueError as error:
        st.error(str(error))
with st.expander('Held-out evaluation and model selection', expanded=True):
    st.write('Model selected by pooled validation log loss across expanding windows for 2021/22, 2022/23 and 2023/24. Benchmark: 2024/25–2025/26. Earlier results update form, while model weights stay fixed. The app uses a separate model refitted on all available matches.')
    st.caption('This benchmark was inspected during the first milestone; it is not a fresh holdout for this iteration.')
    metrics = metric_table(report['test'], ['accuracy', 'macro_f1', 'log_loss', 'brier_multiclass'])
    st.dataframe(metrics.style.format('{:.3f}'))
    st.caption('Lower log loss and Brier score are better. Higher accuracy and macro-F1 are better. Probabilities have not been post-hoc calibrated.')
    st.write('Linear-regression test MAE', report['stats_mae'])
    if 'recall_by_outcome' in report['test'][report['selected_model']]:
        st.write('Selected model recall by outcome', report['test'][report['selected_model']]['recall_by_outcome'])
    if report.get('schema_version') == 2:
        st.subheader('Chronological validation leaderboard')
        leaderboard = metric_table(report['validation'], ['accuracy', 'macro_f1', 'log_loss']).sort_values('log_loss')
        st.dataframe(leaderboard.style.format('{:.4f}'))
