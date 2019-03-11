from flask import Flask, flash, redirect, render_template, request, session, abort
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('home.html')


@app.route("/train", methods=['POST', 'GET'])
def train():
    if request.method == 'POST':
        result = request.form
        pred_size = int(result['predsize'])
        step_size = pred_size
        pred_size = 1
        raw_data = pd.read_csv('static/house-sales.csv')
        timeseries = np.asarray(raw_data['sales'])
        buffer = timeseries.tolist()
        pred = []
        for t in range(pred_size):
            print(t)
            model = ARIMA(buffer, order=(8, 1, 0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast(steps=step_size)
            print(output)
            y = output[0]
            pred.extend(y)
            buffer.extend(y)

        plot = figure(plot_width=900, plot_height=500)
        plot.multi_line([list(range(len(timeseries))),
                         list(range(len(buffer)))],
                        [timeseries, buffer],
                        color=["red", "blue"], alpha=[0.8, 0.3], line_width=4)

        script, div = components(plot)

        return render_template('result.html', div=div, script=script)
    else:
        return "<h1>Direct access not allowed</h1>"


def generate_input_and_output(timeseries, stepsize=3):
    x = []
    y = []
    i = 0
    while i + stepsize < len(timeseries):
        x.append(timeseries[i: i + stepsize])
        y.append(timeseries[i + stepsize])
        i += 1

    return np.asarray(x), np.asarray(y)


if __name__ == "__main__":
    app.run(debug=True)
