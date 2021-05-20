import os
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, mean_squared_error


class Regressor(BaseEstimator):
    """Creates the regerssion model object.

    The object performs curve fitting to the provided data using the `reg_func` target function with the ordinary least square method.

    Attributes
    ----------
    data: DataFrame
        The pd.DataFrame object containing the pump and motor speed and mass data.
    fitted: boolean
        The flag to indicate whether fitting was performed.
    machine_type: {'pump', 'motor'}, optional
        The string specifying a machine type under consideration, default 'pump'.
    data_type: {'speed', 'mass'}, optional
        The string specifying a data type, deafult 'speed'.
    """
    def __init__(self, machine_type, data_type):
        self.machine_type = machine_type
        self.data_type = data_type
        self.fitted_ = False

    def get_params(self, deep=True):
        """A service function to build a custom estimator"""
        return {"machine_type": self.machine_type, "data_type": self.data_type}

    def set_params(self, **parameters):
        """A service function to build a custom estimator"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def reg_func(self, x, *args):
        """Specifies the regression target function.

        Based on the data_type in the `data_type` attribute, the function chooses either a linear-exponential or simple linear model.

        Parameters
        ----------
        x: ndarray
            The function variable.
        *args: tuple
            The tuple with the coefficients of a regression model.

        Returns
        -------
        function
            The target regression function.
        """
        if self.data_type == 'speed':
            return args[0] * np.exp(-args[1] * x) + args[2]
        if self.data_type == 'mass':
            return args[0] * x + args[1]

    def fit(self, x, y):
        """Fits the target regression model to the provided data with the ordinary least square method and updates the class attributes accordingly

        Parameters
        ----------
        x: ndarray
            The feature values
        y: ndarray
            Ground truth target values
        """
        guess_speed = [1e3, 1e-3, 1e3]
        guess_mass = [1, 1]
        if self.data_type == 'speed':
            self.coefs_, self.cov_ = curve_fit(self.reg_func, x, y,
                                               guess_speed)
        elif self.data_type == 'mass':
            self.coefs_, self.cov_ = curve_fit(self.reg_func, x, y, guess_mass)

    def predict(self, x):
        """Calculates the regression function for a given input variable x and regression coefficients (taken from the class attributes).

        Parameters
        ----------
        x: ndarray
            The function variable.
        """
        return self.reg_func(x, *self.coefs_)

    def eval(self, x, y):
        """Calculates the metrics of the estimator (root-mean-square-error and r2_score).

        Parameters
        ----------
        # x: ndarray
            The test feature values
        y: ndarray
            Ground truth target values
        """
        return mean_squared_error(y, self.predict(x),
                                  squared=False), r2_score(y, self.predict(x))

    def plot(self, data, show_figure=False, save_figure=False, format='pdf'):
        """Plots and optionally saves the catalogue data alone or the catalogue data and the regression model if fitting was performed.

        Parameters
        ----------
        data: pd.DataFrame
            The dataframe containing `manufacturer` pd.Series and self.data_type pd.Series.
        show_figure: bool, optional
            The flag for saving the figure, default False.
        save_figure: bool, optional
            The flag for saving the figure, default False.
        format : str, optional
            The file extension in which the figure will be saved, default 'pdf'.

        Returns
        -------
        fig: streamlit figure object
            The streamlit figure.
        """
        fig = go.Figure()
        for idx, i in enumerate(data['manufacturer'].unique()):
            fig.add_scatter(x=data['displacement'][data['manufacturer'] == i],
                            y=data[self.data_type][data['manufacturer'] == i],
                            mode='markers',
                            name=i,
                            marker_symbol=idx,
                            marker=dict(size=7,
                                        line=dict(color='black', width=.5)))
        fig.update_layout(
            title=f'{self.machine_type.capitalize()} {self.data_type} data',
            width=800,
            height=600,
            xaxis=dict(
                title=f'{self.machine_type.capitalize()} displacement, cc/rev',
                showline=True,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.5,
                range=[0, round(1.1 * max(data['displacement']), -2)]),
            yaxis=dict(
                title=f'{self.machine_type.capitalize()} {self.data_type}, rpm'
                if self.data_type == 'speed' else
                f'{self.machine_type.capitalize()} {self.data_type}, kg',
                showline=True,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.5,
                range=[0, round(1.1 * max(data[self.data_type]), -2)]
                if self.data_type == 'mass' else [
                    round(.9 * min(data[self.data_type]), -2),
                    round(1.1 * max(data[self.data_type]), -2)
                ]),
            plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,0)',
            showlegend=True,
        )
        if self.fitted_:
            data = data[data['type'] == f'{self.machine_type.capitalize()}']
            x = data['displacement'].values
            x_cont = np.linspace(.2 * np.amin(x), 1.2 * np.amax(x), num=100)
            for i in zip(('Regression model', 'Upper limit', 'Lower limit'),
                         (0, self.rmse_, -self.rmse_)):
                fig.add_scatter(
                    x=x_cont,
                    y=self.predict(x_cont) + i[1],
                    mode='lines',
                    name=i[0],
                    line=dict(width=1, dash='dash'),
                )
        if save_figure:
            if not os.path.exists('images'):
                os.mkdir('images')
            fig.write_image(
                f'images/{self.machine_type}_{self.data_type}.{format}')
        if show_figure:
            fig.show()
        return fig
