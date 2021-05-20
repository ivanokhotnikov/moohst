import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.figure_factory as ff
from io import BytesIO
from joblib import dump, load
from hst.hst import HST
from hst.regressor import Regressor
from sklearn.model_selection import KFold, train_test_split, cross_validate


def set_defaults(analysis_type):
    """Assigns the default values of oil, its parameters and initial design parameters to initialize the HST, to plot its efficiency map and to conduct other calculations.

    Paramters
    ---
    analysis_type: str, 'sizing', 'performance', 'map' or 'comparison
        A string flag defining a type of callation to customize.
    """
    if analysis_type == 'sizing':
        max_swash_angle = 18
        pistons = 9
        return max_swash_angle, pistons
    if analysis_type == 'performance':
        input_speed = 2160
        pressure_charge = 25
        pressure_discharge = 475
        return input_speed, pressure_charge, pressure_discharge
    if analysis_type == 'map':
        max_speed = 2500
        max_pressure = 500
        max_power = 580
        gear_ratio = 0.8
        return max_speed, max_pressure, max_power, gear_ratio
    if analysis_type == 'comparison':
        displ_1 = 350
        displ_2 = 330
        speed = 2160
        pressure = 475
        oil_temp = 100
        return displ_1, displ_2, speed, pressure, oil_temp


def fit_catalogues(data_in):
    """
    Fits the custom Regressor model to the catalogue data. Each modelis cross_validated with the metrics being saved in the model's properties.

    Parameters:
    ---
    data_in: pd.DataFrame
        Catalogue data

    Returns:
    ---
    models: dict
        Dictionary of the fitted models. Keys: 'pump_mass', 'pump_speed', 'motor_mass', 'motor_speed'. Values: Regressor objects
    """
    models = {}
    if not os.path.exists('models'):
        os.mkdir('models')
    for machine_type in ('pump', 'motor'):
        for data_type in ('speed', 'mass'):
            data = data_in[data_in['type'] == f'{machine_type.capitalize()}']
            model = Regressor(machine_type=machine_type, data_type=data_type)
            x_full = data['displacement'].to_numpy(dtype='float64')
            y_full = data[data_type].to_numpy(dtype='float64')
            x_train, x_test, y_train, y_test = train_test_split(x_full,
                                                                y_full,
                                                                test_size=0.2,
                                                                random_state=0)
            strat_k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_results = cross_validate(
                model,
                x_train,
                y_train,
                cv=strat_k_fold,
                scoring=['neg_root_mean_squared_error', 'r2'],
                return_estimator=True,
                n_jobs=-1,
                verbose=0)
            model.r2_ = np.mean([k for k in cv_results['test_r2']])
            model.cv_rmse_ = -np.mean(
                [k for k in cv_results['test_neg_root_mean_squared_error']])
            model.cv_r2std_ = np.std([k for k in cv_results['test_r2']])
            model.cv_rmsestd_ = np.std(
                [k for k in cv_results['test_neg_root_mean_squared_error']])
            model.coefs_ = np.mean([k.coefs_ for k in cv_results['estimator']],
                                   axis=0)
            model.test_rmse_, model.test_r2_ = model.eval(x_test, y_test)
            model.fitted_ = True
            dump(model,
                 os.path.join('models', f'{machine_type}_{data_type}.joblib'))
            models['_'.join((machine_type, data_type))] = model
    return models


def load_catalogues(github=False):
    """
    Load and returns the regression models as well as the catalogue data.

    Parameters:
    ---
    github: bool, default False
        A flag to read the catalogues from the github repository if set to True and there is no local `models` folder containing fir models. If github=False and the `models` folder contains nothing, `fir_catalogues` is performed.

    Returns:
    ---
    models: dict
        Dictionary of four regression models with te following keys: 'pump_mass', 'pump_speed', 'motor_mass', 'motor_speed'
    data: pd.DataFrame
        Catalogues data
    """
    models = {}
    data = pd.read_csv(
        'https://raw.githubusercontent.com/ivanokhotnikov/effmap_demo/master/data/data.csv',
        index_col='#')
    if os.path.exists('.\\models') and len(os.listdir('.\\models')):
        for file in os.listdir('.\\models'):
            models[file[:-7]] = load(os.path.join(os.getcwd(), 'models', file))
    elif github:
        for machine_type in ('pump', 'motor'):
            for data_type in ('mass', 'speed'):
                link = f'https://github.com/ivanokhotnikov/effmap_demo/blob/master/models/{machine_type}_{data_type}.joblib?raw=true'
                mfile = BytesIO(requests.get(link).content)
                models['_'.join((machine_type, data_type))] = load(mfile)
    else:
        models = fit_catalogues(data)
    return models, data


def plot_catalogue_data(models,
                        data_in,
                        show_figure=False,
                        save_figure=False,
                        format='pdf'):
    for i in models:
        model = models[i]
        data_type = model.data_type
        machine_type = model.machine_type
        data = data_in[data_in['type'] == f'{machine_type.capitalize()}']
        x = data['displacement'].to_numpy(dtype='float64')
        x_cont = np.linspace(.2 * np.amin(x), 1.2 * np.amax(x), num=100)
        sns.set_style('ticks', {
            'spines.linewidth': .25,
        })
        sns.set_palette('Set1',
                        n_colors=len(data['manufacturer'].unique()),
                        desat=.9)
        plot = sns.JointGrid(
            x='displacement',
            y=data_type,
            data=data,
            hue='manufacturer',
        )
        plot.plot_joint(sns.scatterplot, edgecolor='.2', linewidth=.5)
        plot.plot_marginals(sns.kdeplot, fill=True)
        if data_type == 'speed':
            plot.ax_joint.set_ylabel(f'{machine_type.capitalize()} speed, rpm')
            plot.ax_marg_y.set_ylim(500, 5000)
        if data_type == 'mass':
            plot.ax_joint.set_ylabel(f'{machine_type.capitalize()} mass, kg')
            if machine_type == 'pump': plot.ax_marg_y.set_ylim(0, 700)
            if machine_type == 'motor': plot.ax_marg_y.set_ylim(0, 500)
        plot.ax_joint.set_xlabel(
            f'{machine_type.capitalize()} displacement, cc/rev')
        plot.ax_marg_x.set_xlim(0, 900)
        for l in zip(('Fit + RMSE', 'Fit', 'Fit - RMSE'),
                     (+model.test_rmse_, 0, -model.test_rmse_),
                     ('crimson', 'gold', 'seagreen')):
            plot.ax_joint.plot(
                x_cont,
                model.predict(x_cont) + l[1],
                color=l[2],
                linestyle='--',
                label=f'{l[0]}',
                linewidth=1,
            )
        plot.ax_joint.legend()
        if show_figure:
            plt.show()
        if save_figure:
            if not os.path.exists('images'): os.mkdir('images')
            plot.savefig(f'images/{machine_type}_{data_type}.{format}')
        plt.clf()
        plt.close('all')


def plot_catalogue_data_plotly(models,
                               data_in,
                               show_figure=False,
                               save_figure=False,
                               format='pdf'):
    for i in models:
        model = models[i]
        data_type = model.data_type
        machine_type = model.machine_type
        data = data_in[data_in['type'] == f'{machine_type.capitalize()}']
        x = data['displacement'].to_numpy(dtype='float64')
        x_cont = np.linspace(.2 * np.amin(x), 1.2 * np.amax(x), num=100)
        fig_scatter = go.Figure()
        for l in zip(('Fit + SD', 'Fit', 'Fit - SD'),
                     (+model.test_rmse_, 0, -model.test_rmse_),
                     ('--r', '--y', '--g')):
            fig_scatter.add_scatter(
                x=x_cont,
                y=model.predict(x) + l[1],
                name=l[0],
                color=l[2],
            )
        for idx, k in enumerate(data['manufacturer'].unique()):
            fig_scatter.add_scatter(
                x=data['displacement'][data['manufacturer'] == k],
                y=data[data_type][data['manufacturer'] == k],
                mode='markers',
                name=k,
                marker_symbol=idx,
                marker=dict(size=7, line=dict(color='black', width=.5)),
            )
        fig_scatter.update_xaxes(
            title_text=f'{machine_type.capitalize()} displacement, cc/rev',
            linecolor='black',
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=0.25,
            linewidth=0.25,
            mirror='all',
            range=[0, round(1.1 * max(data['displacement']), -2)],
        )
        fig_scatter.update_yaxes(
            title_text=f'{machine_type.capitalize()} {data_type}, rpm'
            if data_type == 'speed' else
            f'{machine_type.capitalize()} {data_type}, kg',
            linecolor='black',
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=0.25,
            linewidth=0.25,
            mirror='all',
            range=[0, round(1.2 * max(data[data_type]), -2)]
            if data_type == 'mass' else [
                round(.7 * min(data[data_type]), -2),
                round(1.2 * max(data[data_type]), -2)
            ],
        )
        fig_scatter.update_layout(
            # title=f'Catalogue data for {machine_type} {data_type}',
            template='none',
            width=700,
            height=600,
            showlegend=True,
            legend=dict(orientation='v',
                        x=.99 if data_type == 'speed' else .01,
                        y=1,
                        xanchor='right' if data_type == 'speed' else 'left',
                        yanchor='top'),
            font=dict(size=14, color='black'),
        )
        if save_figure:
            if not os.path.exists('images'): os.mkdir('images')
            fig_scatter.write_image(
                f'images/{machine_type}_{data_type}.{format}')
        if show_figure:
            fig_scatter.show()


def plot_histograms(models,
                    data_in,
                    show_figure=True,
                    save_figure=False,
                    format='pdf'):
    sns.set_style('ticks', {
        'palette': 'Set1',
        'spines.linewidth': .25,
    })
    for data_type in ('speed', 'mass'):
        for machine_type in ('pump', 'motor'):
            ind = data_in.index[data_in['type'] ==
                                f'{machine_type.capitalize()}']
            predictions = models[f'{machine_type}_{data_type}'].predict(
                data_in.loc[ind, 'displacement'])
            data_in.loc[ind, f'{data_type}_residuals'] = data_in.loc[
                ind, data_type] - predictions
        plot_originals = sns.histplot(
            data=data_in,
            x=data_type,
            kde=True,
            hue='type',
            element='step',
        )
        sns.despine()
        plot_originals.legend_.set_title(None)
        if data_type == 'speed': plt.xlabel('Speed, rpm')
        if data_type == 'mass': plt.xlabel('Mass, kg')
        if show_figure: plt.show()
        if save_figure:
            if not os.path.exists('images'): os.mkdir('images')
            plt.savefig(f'images/hist_originals_{data_type}.{format}')
        plt.clf()
        plot_originals.cla()
        plot_residuals = sns.histplot(
            data=data_in,
            x=f'{data_type}_residuals',
            kde=True,
            hue='type',
            element='step',
        )
        sns.despine()
        plot_residuals.legend_.set_title(None)
        if data_type == 'speed': plt.xlabel('Speed residual, rpm')
        if data_type == 'mass': plt.xlabel('Mass residual, kg')
        if show_figure: plt.show()
        if save_figure:
            if not os.path.exists('images'): os.mkdir('images')
            plt.savefig(f'images/hist_residuals_{data_type}.{format}')
        plt.clf()
        plt.close('all')


def plot_histograms_plotly(models,
                           data_in,
                           show_figure=True,
                           save_figure=False,
                           format='pdf'):
    for data_type in ('speed', 'mass'):
        fig_hist_original = go.Figure()
        fig_hist_residuals = go.Figure()
        for machine_type in ('pump', 'motor'):
            fig_hist_original.add_trace(
                go.Histogram(
                    x=data_in[data_in['type'] ==
                              f'{machine_type.capitalize()}']
                    [data_type].to_numpy(dtype='float64'),
                    name=f'{machine_type.capitalize()}',
                    xbins=dict(size=100 if data_type == 'speed' else 10),
                    opacity=0.75,
                    # histnorm='probability density',
                ))
            fig_hist_residuals.add_trace(
                go.Histogram(
                    x=data_in[data_in['type'] ==
                              f'{machine_type.capitalize()}']
                    [data_type].to_numpy(dtype='float64') -
                    models[f'{machine_type}_{data_type}'].predict(
                        data_in[data_in['type'] ==
                                f'{machine_type.capitalize()}']
                        ['displacement'].to_numpy(dtype='float64')),
                    name=f'{machine_type.capitalize()}',
                    xbins=dict(size=100 if data_type == 'speed' else 10),
                    opacity=0.75,
                    # histnorm='probability density',
                ))
            fig_hist_original.update_layout(
                # title=f'Distribution of the original {data_type} data',
                width=700,
                height=600,
                template='none',
                barmode='overlay',
                xaxis=dict(
                    title=f'{data_type.capitalize()}, rpm' if data_type
                    == 'speed' else f'{data_type.capitalize()}, kg',
                    showline=True,
                    linecolor='black',
                    showgrid=True,
                    gridcolor='LightGray',
                    gridwidth=0.25,
                    linewidth=0.25,
                    mirror='all',
                ),
                yaxis=dict(
                    title=f'Count',
                    showline=True,
                    linecolor='black',
                    showgrid=True,
                    gridcolor='LightGray',
                    gridwidth=0.25,
                    linewidth=0.25,
                    mirror='all',
                ),
                showlegend=True,
                legend=dict(orientation='v',
                            x=.99,
                            y=1,
                            xanchor='right',
                            yanchor='top'),
                font=dict(size=14, color='black'),
            )
            fig_hist_residuals.update_layout(
                # title=f'Distribution of the residuals of the {data_type} data',
                width=700,
                height=600,
                template='none',
                barmode='overlay',
                xaxis=dict(
                    title=f'{data_type.capitalize()} models residuals, rpm'
                    if data_type == 'speed' else
                    f'{data_type.capitalize()} models residuals, kg',
                    showline=True,
                    linecolor='black',
                    showgrid=True,
                    gridcolor='LightGray',
                    gridwidth=0.25,
                    linewidth=0.25,
                    mirror='all',
                ),
                yaxis=dict(
                    title=f'Frequency',
                    showline=True,
                    linecolor='black',
                    showgrid=True,
                    gridcolor='LightGray',
                    gridwidth=0.25,
                    linewidth=0.25,
                    mirror='all',
                ),
                showlegend=True,
                legend=dict(orientation='v',
                            x=.99,
                            y=1,
                            xanchor='right',
                            yanchor='top'),
                font=dict(size=14, color='black'),
            )
        if save_figure:
            if not os.path.exists('images'): os.mkdir('images')
            fig_hist_original.write_image(
                f'images/hist_originals_{data_type}.{format}')
            fig_hist_residuals.write_image(
                f'images/hist_residuals_{data_type}.{format}')
        if show_figure:
            fig_hist_original.show()
            fig_hist_residuals.show()


def plot_distributions_plotly(models,
                              data_in,
                              show_figure=True,
                              save_figure=False,
                              format='pdf'):
    for data_type in ('speed', 'mass'):
        print_data_originals = []
        print_data_residuals = []
        for machine_type in ('pump', 'motor'):
            print_data_originals.append(
                data_in[data_in['type'] == f'{machine_type.capitalize()}']
                [data_type].to_numpy(dtype='float64'))
            print_data_residuals.append(
                data_in[data_in['type'] == f'{machine_type.capitalize()}']
                [data_type].to_numpy(dtype='float64') -
                models[f'{machine_type}_{data_type}'].predict(
                    data_in[data_in['type'] == f'{machine_type.capitalize()}']
                    ['displacement'].to_numpy(dtype='float64')))
        fig_dist_originals = ff.create_distplot(
            print_data_originals,
            ['Pump', 'Motor'],
            show_hist=True,
            bin_size=100 if data_type == 'speed' else 10,
            show_rug=False,
        )
        fig_dist_originals.update_layout(
            # title=f'Distribution of the original {data_type} data',
            width=700,
            height=600,
            template='none',
            xaxis=dict(
                title=f'{data_type.capitalize()}, rpm'
                if data_type == 'speed' else f'{data_type.capitalize()}, kg',
                showline=True,
                linecolor='black',
                showgrid=True,
                gridcolor='LightGray',
                linewidth=0.25,
                gridwidth=0.25,
                mirror='all',
            ),
            yaxis=dict(
                title=f'Probability density',
                showline=True,
                linecolor='black',
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.25,
                mirror='all',
            ),
            showlegend=True,
            legend=dict(orientation='v',
                        x=.99,
                        y=1,
                        xanchor='right',
                        yanchor='top'),
            font=dict(size=14, color='black'))
        fig_dist_residuals = ff.create_distplot(
            print_data_residuals,
            ['Pump', 'Motor'],
            show_hist=True,
            bin_size=100 if data_type == 'speed' else 10,
            show_rug=False,
        )
        fig_dist_residuals.update_layout(
            # title=f'Distribution of the residuals of the {data_type} data',
            width=700,
            height=600,
            template='none',
            xaxis=dict(
                title=f'{data_type.capitalize()} models residuals, rpm'
                if data_type == 'speed' else
                f'{data_type.capitalize()} models residuals, kg',
                showline=True,
                linecolor='black',
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.25,
                mirror='all',
            ),
            yaxis=dict(
                title=f'Probability density',
                showline=True,
                linecolor='black',
                showgrid=True,
                gridcolor='LightGray',
                gridwidth=0.25,
                linewidth=0.25,
                mirror='all',
            ),
            showlegend=True,
            legend=dict(orientation='v',
                        x=.99,
                        y=1,
                        xanchor='right',
                        yanchor='top'),
            font=dict(size=14, color='black'))
        if save_figure:
            if not os.path.exists('images'): os.mkdir('images')
            fig_dist_originals.write_image(
                f'images/dist_originals_{data_type}.{format}')
            fig_dist_residuals.write_image(
                f'images/dist_residuals_{data_type}.{format}')
        if show_figure:
            fig_dist_originals.show()
            fig_dist_residuals.show()


def plot_validation(show_figure=False, save_figure=False, format='pdf'):
    data = pd.read_csv(
        'https://raw.githubusercontent.com/ivanokhotnikov/effmap_demo/master/data/test_data.csv'
    )
    data.dropna(
        subset=['Forward Speed', 'Reverse Speed', 'Volumetric at 1780RPM'],
        inplace=True)
    speeds = data[['Forward Speed', 'Reverse Speed']].astype(float)
    speeds = speeds.stack()
    vol_eff = speeds / 1780 * 1e2
    piston_max = 1.1653 * 25.4 * 1e-3
    piston_min = 1.1650 * 25.4 * 1e-3
    bore_max = 1.1677 * 25.4 * 1e-3
    bore_min = 1.1671 * 25.4 * 1e-3
    rad_clearance_max = (bore_max - piston_min) / 2
    rad_clearance_min = (bore_min - piston_max) / 2
    benchmark = HST(swash=15, oil='SAE 30', oil_temp=60)
    benchmark.compute_sizes(displ=196,
                            k1=.7155,
                            k2=.9017,
                            k3=.47,
                            k4=.9348,
                            k5=.9068)
    benchmark.load_oil()
    eff_min = benchmark.compute_eff(speed_pump=1780,
                                    pressure_discharge=207,
                                    pressure_charge=14,
                                    h3=rad_clearance_max)[0]
    eff_max = benchmark.compute_eff(speed_pump=1780,
                                    pressure_discharge=207,
                                    pressure_charge=14,
                                    h3=rad_clearance_min)[0]
    sns.set_style('ticks', {
        'spines.linewidth': .25,
    })
    sns.set_palette('Set1')
    plot = sns.histplot(
        data=vol_eff,
        kde=True,
        element='step',
        label='Test data',
        color='steelblue',
    )
    sns.despine()
    plot.axes.axvline(
        eff_max['hst']['volumetric'],
        label=
        f"Prediction at min clearance, {round(eff_max['hst']['volumetric'], 2)}%",
        linestyle='--',
        color='crimson',
        linewidth=1,
    )
    plot.axes.axvline(
        eff_min['hst']['volumetric'],
        label=
        f"Prediction at max clearance, {round(eff_min['hst']['volumetric'], 2)}%",
        linestyle='--',
        color='seagreen',
        linewidth=1,
    )
    plot.axes.axvline(
        vol_eff.mean(),
        label=f"Test mean, {round(vol_eff.mean(), 2)}%",
        linestyle='--',
        color='steelblue',
        linewidth=1,
    )
    plot.axes.axvline(
        vol_eff.mean() + vol_eff.std(),
        label=f"Test mean + SD, {round(vol_eff.mean() + vol_eff.std(),2)}%",
        linestyle='--',
        color='darkorange',
        linewidth=1,
    )
    plot.axes.axvline(
        vol_eff.mean() - vol_eff.std(),
        label=f"Test mean - SD, {round(vol_eff.mean() - vol_eff.std(),2)}%",
        color='slateblue',
        linestyle='--',
        linewidth=1,
    )
    plt.xlim(84, 95)
    plt.xlabel('HST volumetric efficiency, %')
    plt.legend(loc='upper right', fontsize='x-small')
    if save_figure:
        if not os.path.exists('images'): os.mkdir('images')
        plt.savefig(f'images/validation.{format}')
    if show_figure: plt.show()
    plt.clf()
    plt.close('all')


def plot_validation_plotly(show_figure=False, save_figure=False, format='pdf'):
    """Performs validation of the efficiency model by computing the volumetric efficiency of the benchmark HST, for which the test data is available containing results of measurements of the input/output speeds, and comparing the computed efficiency with the test data.
    """
    data = pd.read_csv(
        'https://raw.githubusercontent.com/ivanokhotnikov/effmap_demo/master/data/test_data.csv'
    )
    data.dropna(
        subset=['Forward Speed', 'Reverse Speed', 'Volumetric at 1780RPM'],
        inplace=True)
    speeds = data[['Forward Speed', 'Reverse Speed']].astype(float)
    speeds = speeds.stack()
    vol_eff = speeds / 1780 * 1e2
    piston_max = 1.1653 * 25.4 * 1e-3
    piston_min = 1.1650 * 25.4 * 1e-3
    bore_max = 1.1677 * 25.4 * 1e-3
    bore_min = 1.1671 * 25.4 * 1e-3
    rad_clearance_max = (bore_max - piston_min) / 2
    rad_clearance_min = (bore_min - piston_max) / 2
    benchmark = HST(swash=15, oil='SAE 30', oil_temp=60)
    benchmark.compute_sizes(displ=196,
                            k1=.7155,
                            k2=.9017,
                            k3=.47,
                            k4=.9348,
                            k5=.9068)
    benchmark.load_oil()
    eff_min = benchmark.compute_eff(speed_pump=1780,
                                    pressure_discharge=207,
                                    pressure_charge=14,
                                    h3=rad_clearance_max)[0]
    eff_max = benchmark.compute_eff(speed_pump=1780,
                                    pressure_discharge=207,
                                    pressure_charge=14,
                                    h3=rad_clearance_min)[0]
    fig = ff.create_distplot(
        [vol_eff],
        [
            f"Test data. Mean = {round(vol_eff.mean(),2)}%, SD = {round(vol_eff.std(),2)}%"
        ],
        show_hist=True,
        bin_size=.3,
        show_rug=False,
    )
    fig.add_scatter(
        x=[eff_max['hst']['volumetric'], eff_max['hst']['volumetric']],
        y=[0, .6],
        mode='lines',
        name=
        f"Prediction at min clearance, {round(eff_max['hst']['volumetric'],2)}%",
        line=dict(width=1.5, ),
    )
    fig.add_scatter(
        x=[eff_min['hst']['volumetric'], eff_min['hst']['volumetric']],
        y=[0, .6],
        mode='lines',
        name=
        f"Prediction at max clearance, {round(eff_min['hst']['volumetric'],2)}%",
        line=dict(width=1.5, ),
    )
    fig.add_scatter(
        x=[vol_eff.mean(), vol_eff.mean()],
        y=[0, .6],
        mode='lines',
        name=f"Test mean, {round(vol_eff.mean(),2)}%",
        line=dict(width=1.5, dash='dash'),
    )
    fig.add_scatter(
        x=[vol_eff.mean() + vol_eff.std(),
           vol_eff.mean() + vol_eff.std()],
        y=[0, .6],
        mode='lines',
        name='Test mean + SD',
        line=dict(width=1.5, dash='dash'),
    )
    fig.add_scatter(
        x=[vol_eff.mean() - vol_eff.std(),
           vol_eff.mean() - vol_eff.std()],
        y=[0, .6],
        mode='lines',
        name='Test mean - SD',
        line=dict(width=1.5, dash='dash'),
    )
    fig.update_layout(
        # title=
        # f'Sample of {len(vol_eff)} measurements of the {benchmark.displ} cc/rev HST with {benchmark.oil} at {benchmark.oil_temp}C',
        template='none',
        width=1000,
        height=600,
        xaxis=dict(
            title='HST volumetric efficiency, %',
            showline=True,
            linecolor='black',
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=0.25,
            linewidth=0.25,
            range=[84, 94],
            dtick=2,
            mirror='all',
        ),
        yaxis=dict(
            title='Probability density',
            showline=True,
            linecolor='black',
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=0.25,
            linewidth=0.25,
            range=[0, .6],
            mirror='all',
        ),
        showlegend=True,
        legend_orientation='v',
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top'),
        font=dict(size=14, color='black'))
    if save_figure:
        if not os.path.exists('images'): os.mkdir('images')
        fig.write_image(f'images/validation.{format}')
    if show_figure: fig.show()


def plot_hst_comparison(displ_1,
                        displ_2,
                        speed,
                        pressure,
                        temp,
                        show_figure=True,
                        save_figure=False,
                        format='pdf'):
    """
    Prints a bar plot to compare total efficiencies of two HSTs.

    Parameters:
    ---
    displ_1, displ_2: float
        Displacements of the HSTs to be comapred
    speed, pressure, temp, charge: floats
        Operational parameters for the comparison
    """
    effs_1, effs_2 = [], []
    motor_pows_1, motor_pows_2 = [], []
    pump_pows_1, pump_pows_2 = [], []
    oils = ('SAE 15W40', 'SAE 10W40', 'SAE 10W60', 'SAE 5W40', 'SAE 0W30',
            'SAE 30')
    hst_1, hst_2 = HST(oil_temp=temp), HST(oil_temp=temp)
    hst_1.compute_sizes(displ_1)
    hst_2.compute_sizes(displ_2)
    for oil in oils:
        hst_1.oil, hst_2.oil = oil, oil
        hst_1.load_oil()
        hst_2.load_oil()
        eff_1 = hst_1.compute_eff(speed, pressure)[0]
        eff_2 = hst_2.compute_eff(speed, pressure)[0]
        effs_1.append(eff_1['hst']['total'])
        effs_2.append(eff_2['hst']['total'])
        motor_pows_1.append(hst_1.performance['motor']['power'])
        motor_pows_2.append(hst_2.performance['motor']['power'])
        pump_pows_1.append(hst_1.performance['pump']['power'])
        pump_pows_2.append(hst_2.performance['pump']['power'])
    fig_eff = go.Figure()
    fig_eff.add_trace(
        go.Bar(
            x=oils,
            y=effs_1,
            text=[f'{eff:.2f}' for eff in effs_1],
            textposition='auto',
            name=f'{displ_1} cc/rev',
            marker_color='steelblue',
        ))
    fig_eff.add_trace(
        go.Bar(x=oils,
               y=effs_2,
               text=[f'{eff:.2f}' for eff in effs_2],
               textposition='auto',
               name=f'{displ_2} cc/rev',
               marker_color='indianred'))
    fig_eff.update_layout(
        # title=
        # f'Total efficiency of {displ_1} and {displ_2} cc/rev HSTs at {speed} rpm, {pressure} bar, {temp}C oil',
        yaxis=dict(
            title='Total HST efficiency, %',
            range=[50, 90],
        ),
        template='none',
        showlegend=True,
        legend_orientation='h',
        width=800,
        height=500,
        font=dict(size=14),
    )
    fig_pow = go.Figure()
    fig_pow.add_trace(
        go.Bar(
            x=oils,
            y=pump_pows_1,
            text=[f'{pow:.2f}' for pow in pump_pows_1],
            textposition='auto',
            name=f'{displ_1} cc/rev in',
            marker_color='steelblue',
        ))
    fig_pow.add_trace(
        go.Bar(
            x=oils,
            y=motor_pows_1,
            text=[f'{pow:.2f}' for pow in motor_pows_1],
            textposition='auto',
            name=f'{displ_1} cc/rev  out',
            marker_color='lightblue',
        ))
    fig_pow.add_trace(
        go.Bar(
            x=oils,
            y=pump_pows_2,
            text=[f'{pow:.2f}' for pow in pump_pows_2],
            textposition='auto',
            name=f'{displ_2} cc/rev  in',
            marker_color='indianred',
        ))
    fig_pow.add_trace(
        go.Bar(x=oils,
               y=motor_pows_2,
               text=[f'{pow:.2f}' for pow in motor_pows_2],
               textposition='auto',
               name=f'{displ_2} cc/rev out',
               marker_color='pink'))
    fig_pow.update_layout(
        # title=
        # f'Power balance of {displ_1} and {displ_2} cc/rev HSTs at {speed} rpm, {pressure} bar, {temp}C oil',
        yaxis=dict(title='Power, kW', ),
        template='none',
        showlegend=True,
        legend_orientation='h',
        width=900,
        height=600,
        font=dict(size=14, color='black'),
    )
    if save_figure:
        if not os.path.exists('images'):
            os.mkdir('images')
        fig_eff.write_image(f'images/hst_eff_comparison.{format}')
        fig_pow.write_image(f'images/hst_power_comparison.{format}')
    if show_figure:
        fig_eff.show()
        fig_pow.show()


def plot_engines_comparison(hst1,
                            hst4,
                            show_figure=True,
                            save_figure=False,
                            format='pdf'):
    engine_1 = hst1.load_engines()['engine_1']
    engine_4 = hst4.load_engines()['engine_4']
    fig_comparison = go.Figure()
    fig_comparison.add_scatter(
        x=engine_4['speed'],
        y=engine_4['torque'],
        name='Engine 4',
        mode='lines+markers',
        marker=dict(size=3),
        line=dict(color='indianred', width=1.5),
    )
    fig_comparison.add_scatter(
        x=engine_1['speed'],
        y=engine_1['torque'],
        name='Engine 1',
        mode='lines+markers',
        marker=dict(size=3),
        line=dict(color='steelblue', width=1.5),
    )
    fig_comparison.update_xaxes(
        title_text=f'Engine speed, rpm',
        linecolor='black',
        showgrid=True,
        gridcolor='LightGray',
        gridwidth=0.25,
        linewidth=0.25,
        mirror='all',
    )
    fig_comparison.update_yaxes(
        title_text=f'Engine torque, Nm',
        linecolor='black',
        showgrid=True,
        gridcolor='LightGray',
        gridwidth=0.25,
        linewidth=0.25,
        mirror='all',
    )
    fig_comparison.update_layout(
        # title=f'Engines comparison',
        template='none',
        width=900,
        height=600,
        showlegend=True,
        font=dict(size=14, color='black'),
    )
    if save_figure:
        if not os.path.exists('images'):
            os.mkdir('images')
        fig_comparison.write_image(f'images/engines_comparison.{format}')
    if show_figure:
        fig_comparison.show()


def print_to_moohst(show=True, save=False):
    if show:
        shows = True
        saves = False
    elif save:
        shows = False
        saves = True
    #*  Catalogue data
    models, data = load_catalogues()
    plot_catalogue_data(
        models,
        data,
        show_figure=shows,
        save_figure=saves,
    )
    plot_histograms(
        models,
        data,
        show_figure=shows,
        save_figure=saves,
    )
    #*  Validation
    plot_validation(show_figure=shows, save_figure=saves)
    #*  HST initialization
    hst = HST(*set_defaults('sizing'))
    hst.compute_sizes(displ=500)
    hst.compute_speed_limit(models['pump_speed'])
    #*  Oil setting
    hst.oil = 'SAE 15W40'
    hst.oil_temp = 100
    hst.load_oil()
    hst.plot_oil(show_figure=shows, save_figure=saves)
    #* Maps preparation
    input_speed, pressure_charge, pressure_discharge = set_defaults(
        'performance')
    max_speed, max_pressure, hst.max_power_input, hst.input_gear_ratio = set_defaults(
        'map')
    hst.engine = None
    #* Maps plotting
    hst.plot_eff_map(
        max_speed,
        max_pressure,
        pressure_charge=pressure_charge,
        show_figure=shows,
        save_figure=saves,
    )
    hst.plot_power_map(
        max_speed,
        max_pressure,
        pressure_charge=pressure_charge,
        show_figure=shows,
        save_figure=saves,
    )
    #* Comparisons
    # engine_1 = HST(440, engine='engine_1')
    # engine_4 = HST(350,
    #                engine='engine_4',
    #                max_power_input=580,
    #                input_gear_ratio=.8)
    # plot_engines_comparison(engine_1,
    #                         engine_4,
    #                         show_figure=shows,
    #                         save_figure=saves,
    #                         format='pdf')
    # plot_hst_comparison(*set_defaults('comparison'),
    #                     show_figure=shows,
    #                     save_figure=saves,
    #                     format='pdf')


if __name__ == '__main__':
    print_to_moohst(show=True, save=False)