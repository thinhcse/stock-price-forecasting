import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

def return_plot(model, data_test_iter, time_stamps_test, configs):
    times = time_stamps_test[configs["model"]["window_size"]:]
    preds = []
    groundtruths = []
    for input, groundtruth in data_test_iter:
        pred = model(input).ravel().detach().numpy()
        preds.append(pred)
        groundtruths.append(groundtruth)

    selects = range(0, len(preds), 7)
    times = [times[i] for i in selects]
    preds = [preds[i] for i in selects]
    groundtruths = [groundtruths[i] for i in selects]

    df_preds = pd.DataFrame(data = preds, columns = ['ScaledReturn'], index = times)
    df_groundtruths = pd.DataFrame(data = groundtruths, columns = ['ScaledReturn'], index = times)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df_preds.index, y = df_preds.ScaledReturn, name = 'predict'))
    fig.add_trace(go.Scatter(x = df_groundtruths.index, y = df_groundtruths.ScaledReturn, name = 'groundtruth'))
    fig.update_layout(title = f'S&P500 ScaledReturn Forecasting', font_family = 'Courier New')
    fig.show()