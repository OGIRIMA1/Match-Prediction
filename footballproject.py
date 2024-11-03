import streamlit as st
import random
import pandas as pd
import plotly.express as px
import joblib
import sklearn
import numpy as np



# Load the dataset
df = pd.read_csv(r"C:\Users\DELL\Documents\python_class_it\Projects\Match Prediction\readingmatch_stats.csv")


#load the model
with open ('mymodel.pkl','rb') as file:
    log = joblib.load(file)

with open ('scoresmodel.pkl','rb') as file:
    Rad = joblib.load(file)

# DISPLAY FIELD FOR TEAM DATA IN THE SIDEBAR
with st.sidebar:
    team1 = st.selectbox("SELECT TEAM 1", df["Team"].sort_values().unique())
    team2 = st.selectbox("SELECT TEAM 2", df["Team"].sort_values().unique())
    show_data = st.button("RESULTING DATA")
    show_result=st.button('Predict Result')

container = st.container()
data1, data2 = container.columns(2)

# Title and description
st.title("FOOTBALL ANALYSIS")
st.markdown("ALL PREMIER LEAGUE TEAMS VISUALIZATIONS CHARTS ")

# Function to create visualizations for a selected team
def create_visualizations(team_data, team_name):
    if team_data.empty:
        st.warning(f"No data available for {team_name}.")
        return

    team_data['Date'] = pd.to_datetime(team_data['Date'])  # Ensure 'Date' is in datetime format

    # Create a line chart to show Goals For (GF) over time
    fig_gf = px.line(team_data, x='Date', y='GF', title=f'Goals For Over Time for {team_name}')
    st.plotly_chart(fig_gf)

    # Create a line chart to show Goals Against (GA) over time
    fig_ga = px.line(team_data, x='Date', y='GA', title=f'Goals Against Over Time for {team_name}')
    st.plotly_chart(fig_ga)

    # Create a scatter plot to show the relationship between Shots (Sh) and Goals For (GF)
    fig_sh_gf = px.scatter(team_data, x='Sh', y='GF', title=f'Relationship between Shots and Goals For for {team_name}')
    st.plotly_chart(fig_sh_gf)

    # Create a scatter plot to show the relationship between Shots on Target (SoT) and Goals For (GF)
    fig_sot_gf = px.scatter(team_data, x='SoT', y='GF', title=f'Relationship between Shots on Target and Goals For for {team_name}')
    st.plotly_chart(fig_sot_gf)

    # Create a pie chart to show the distribution of match results for the selected team
    match_results = team_data['Result'].value_counts()
    fig_results = px.pie(values=match_results.values, names=match_results.index, title=f'Distribution of Match Results for {team_name}')
    st.plotly_chart(fig_results)

    # Create a bar chart for total Goals For and Goals Against
    fig_bar = px.bar(team_data, x='Date', y=['GF', 'GA'], title=f'Total Goals For and Against for {team_name}')
    st.plotly_chart(fig_bar)

    # Create a half pie chart to show the distribution of wins and losses for the selected team
    fig_half_pie = px.pie(values=match_results.values, names=match_results.index, title=f'Wins and Losses for {team_name}', hole=0.5)
    st.plotly_chart(fig_half_pie)

    # Create a sunburst chart to show the distribution of match results by team Result
    fig_sunburst = px.sunburst(team_data, path=['Result'], values='GF', title=f'Sunburst Chart for {team_name}')
    st.plotly_chart(fig_sunburst)

    # Create a tree map to show the distribution of match results by team Result
    fig_tree_map = px.treemap(team_data, path=['Result'], values='GA', title=f'Tree Map for {team_name}')
    st.plotly_chart(fig_tree_map)

# Filter the data for both teams and create visualizations
if show_data:
    team1_data = df[df['Team'] == team1]
    team2_data = df[df['Team'] == team2]

    with data1:
        st.subheader(f"Visualizations for {team1}")
        create_visualizations(team1_data, team1)

    with data2:
        st.subheader(f"Visualizations for {team2}")
        create_visualizations(team2_data, team2)

if show_result:
        
    team1_stats=df.loc[df["Team"]== team1].select_dtypes(include=[np.number]).mean()
    team2_stats=df.loc[df["Team"]== team2].select_dtypes(include=[np.number]).mean()
    # Prepare input data
    def safe_get_feature(stats, feature):
        return stats[feature] if feature in stats.index else 0

    features = ['GF', 'Round', 'xGA', 'Poss', 'SoT', 'Sh', 'GA', 'xG']

    input_data = np.array([safe_get_feature(team1_stats, feature) - safe_get_feature(team2_stats, feature) for feature in features])
    input_data = input_data.reshape(1, -1)

    # Make prediction
    prediction = log.predict(input_data)
    prediction_proba = log.predict_proba(input_data)

    # Show results
    st.subheader('Prediction Results:')

    # Convert numeric prediction to result
    result_mapping = {0: 'LOSE', 1: 'DRAW', 2: 'WIN'}
    (f"{team1} IS PREDICTED TO {result_mapping[prediction[0] ]} AGAINST {team2}")
    predicted_result = result_mapping[prediction[0]]

    st.write(f'Predicted Outcome: **{predicted_result}**')

    # Display probabilities
    st.write('Probability Distribution:')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Loss Probability', f'{prediction_proba[0][0]:.2%}')
    with col2:
        st.metric('Draw Probability', f'{prediction_proba[0][1]:.2%}')
    with col3:
        st.metric('Win Probability', f'{prediction_proba[0][2]:.2%}')

   
    team1_stats=df.loc[df["Team"]== team1].select_dtypes(include=[np.number]).mean()
    team2_stats=df.loc[df["Team"]== team2].select_dtypes(include=[np.number]).mean()




    # Use the result from the first prediction model
    predicted_result = result_mapping[prediction[0]]

    # Generate scores based on the predicted result
    if predicted_result == 'WIN':
        score1 = random.randint(1, 5)  # Team 1 scores 1-5 goals
        score2 = random.randint(0, score1 - 1)  # Team 2 scores 0 to (Team 1's score - 1) goals
    elif predicted_result == 'LOSE':
        score2 = random.randint(1, 5)  # Team 2 scores 1-5 goals
        score1 = random.randint(0, score2 - 1)  # Team 1 scores 0 to (Team 2's score - 1) goals
    else:  # DRAW
        score1 = score2 = random.randint(0, 3)  # Both teams score the same (0-3 goals)

    st.write(f'Predicted Score: {team1} {score1} - {score2} {team2}')
    