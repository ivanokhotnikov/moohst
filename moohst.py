import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hst_analysis as ha
import plotly.graph_objects as go
from hst.hst import HST
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_decision_making


class MOOHST(Problem):
    def __init__(self):
        super().__init__(n_var=3,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array([200, 1000, 100]),
                         xu=np.array([800, 3000, 700]),
                         elementwise_evaluation=True)
        self.hst = HST()
        self.hst.oil = 'SAE 15W40'
        self.hst.oil_temp = 100
        self.hst.load_oil()
        self.reg_models, _ = ha.load_catalogues()

    def _evaluate(self, x, out, *args, **kwargs):
        self.hst.compute_sizes(x[0])
        f1 = np.dot(
            self.reg_models['pump_mass'].coefs_ +
            self.reg_models['motor_mass'].coefs_, [x[0], 1])
        f2 = -self.hst.compute_eff(x[1], x[2])[0]['hst']['total']
        f3 = -self.hst.compute_eff(x[1], x[2])[1]['motor']['power']
        g1 = x[1] - self.reg_models['pump_speed'].coefs_[0] * np.exp(
            -self.reg_models['pump_speed'].coefs_[1] *
            x[0]) - self.reg_models['pump_speed'].coefs_[2]
        out["F"] = [f1, f2, f3]
        out["G"] = [g1]


def plot_scatter_front_3d_plotly(res,
                                 show_figure=True,
                                 save_figure=False,
                                 format='pdf'):
    fig = go.Figure(data=[
        go.Scatter3d(
            x=res.F[:, 0],
            y=-res.F[:, 1],
            z=-res.F[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='indianred',
                opacity=1,
                line=dict(color='black', width=.5),
            ),
        )
    ])
    fig.update_layout(width=700,
                      height=700,
                      template='simple_white',
                      margin=dict(l=0, r=0, b=0, t=0),
                      scene_camera=dict(
                          eye=dict(x=1.5, y=1.5, z=1.5),
                          center=dict(x=0, y=0, z=0),
                      ))
    fig.update_traces(name='Pareto front',
                      projection_x_show=True,
                      projection_y_show=True,
                      projection_z_show=True,
                      projection_x_opacity=.6,
                      projection_y_opacity=.6,
                      projection_z_opacity=.6,
                      projection_x_scale=.3,
                      projection_y_scale=.3,
                      projection_z_scale=.3,
                      textfont_size=14,
                      surfacecolor='indianred')
    fig.update_scenes(xaxis=go.layout.scene.XAxis(title='Mass, kg',
                                                  linecolor='black',
                                                  showgrid=True,
                                                  linewidth=0.5),
                      yaxis=go.layout.scene.YAxis(title='Total efficiency, %',
                                                  linecolor='black',
                                                  showgrid=True,
                                                  linewidth=0.5),
                      zaxis=go.layout.scene.ZAxis(title='Absorbed power, kW',
                                                  linecolor='black',
                                                  showgrid=True,
                                                  linewidth=0.5))
    if save_figure:
        if not os.path.exists('images'): os.mkdir('images')
        fig.write_image(f'images/scatter_front_3d.{format}')
    if show_figure: fig.show()


def plot_scatter_set_3d_plotly(res,
                               show_figure=True,
                               save_figure=False,
                               format='pdf'):
    fig = go.Figure(data=[
        go.Scatter3d(
            x=res.X[:, 0],
            y=res.X[:, 1],
            z=res.X[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='navy',
                opacity=1,
                line=dict(color='black', width=.5),
            ),
        )
    ])
    fig.update_layout(width=700,
                      height=700,
                      template='simple_white',
                      margin=dict(l=0, r=0, b=0, t=0),
                      scene_camera=dict(
                          eye=dict(x=1.6, y=1.6, z=1.6),
                          center=dict(x=0, y=0, z=0),
                      ))
    fig.update_traces(name='Pareto set',
                      projection_x_show=True,
                      projection_y_show=True,
                      projection_z_show=True,
                      projection_x_opacity=.6,
                      projection_y_opacity=.6,
                      projection_z_opacity=.6,
                      projection_x_scale=.3,
                      projection_y_scale=.3,
                      projection_z_scale=.3,
                      textfont_size=14,
                      surfacecolor='indianred')
    fig.update_scenes(xaxis=go.layout.scene.XAxis(title='Displacement, cc/rev',
                                                  linecolor='black',
                                                  showgrid=True,
                                                  linewidth=0.5),
                      yaxis=go.layout.scene.YAxis(title='Pump speed, rpm',
                                                  linecolor='black',
                                                  showgrid=True,
                                                  linewidth=0.5),
                      zaxis=go.layout.scene.ZAxis(
                          title='Pressure differential, bar',
                          linecolor='black',
                          showgrid=True,
                          linewidth=0.5))
    if save_figure:
        if not os.path.exists('images'): os.mkdir('images')
        fig.write_image(f'images/scatter_set_3d.{format}')
    if show_figure: fig.show()


def plot_scatter_front_3d(res,
                          I,
                          show_figure=True,
                          save_figure=False,
                          format='pdf'):
    sns.set_style('ticks', {
        'spines.linewidth': .25,
    })
    fig = plt.figure()
    X = res.F[:, 0]
    Y = -res.F[:, 1]
    Z = -res.F[:, 2]
    ax = plt.subplot(projection='3d')
    ax.scatter(X, Y, Z, s=5, marker='.', c='steelblue', alpha=1)
    ax.scatter(X[I], Y[I], Z[I], marker='o', s=10, c='darkorange', alpha=1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel('HST mass, kg')
    ax.set_ylabel('Total HST efficiency, %')
    ax.set_zlabel('Transmitted power, kW')
    ax.view_init(elev=30, azim=60)

    #*Projections
    ax1 = plt.subplot(projection='3d')
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    cx = np.ones_like(X) * ax.get_xlim3d()[0]
    cy = np.ones_like(X) * ax.get_ylim3d()[0]
    cz = np.ones_like(Z) * ax.get_zlim3d()[0]

    ax1.scatter(X, Y, cz, c='steelblue', marker='.', s=2, alpha=.2)  #c=Z
    ax1.scatter(X, cy, Z, c='steelblue', marker='.', s=2, alpha=.2)  #c=-Y
    ax1.scatter(cx, Y, Z, c='steelblue', marker='.', s=2, alpha=.2)  #c=X

    ax1.scatter(X[I], Y[I], cz[I], c='darkorange', marker='o', s=5,
                alpha=.2)  #c=Z
    ax1.scatter(X[I], cy[I], Z[I], c='darkorange', marker='o', s=5,
                alpha=.2)  #c=-Y
    ax1.scatter(cx[I], Y[I], Z[I], c='darkorange', marker='o', s=5,
                alpha=.2)  #c=X

    ax1.set_xlim3d(ax.get_xlim3d())
    ax1.set_ylim3d(ax.get_ylim3d())
    ax1.set_zlim3d(ax.get_zlim3d())

    if save_figure:
        if not os.path.exists('images'): os.mkdir('images')
        plt.savefig(f'images/scatter_front_3d.{format}',
                    bbox_inches='tight',
                    orientation='landscape',
                    pad_inches=.1)
    if show_figure: plt.show()
    plt.clf()
    plt.close('all')


def plot_scatter_set_3d(res,
                        I,
                        show_figure=True,
                        save_figure=False,
                        format='pdf'):
    sns.set_style('ticks', {
        'spines.linewidth': .25,
    })
    fig = plt.figure()
    X = res.X[:, 0]
    Y = res.X[:, 1]
    Z = res.X[:, 2]
    ax = plt.subplot(projection='3d')
    ax.scatter(X, Y, Z, s=5, marker='.', c='steelblue', alpha=1)
    ax.scatter(X[I], Y[I], Z[I], s=10, c='darkorange', alpha=1)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel('Displacement, cc/rev')
    ax.set_ylabel('Pump speed, rpm')
    ax.set_zlabel('Pressure differential, bar')
    ax.view_init(elev=30, azim=60)

    #* Projections
    ax1 = plt.subplot(projection='3d')
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    cx = np.ones_like(X) * ax.get_xlim3d()[0]
    cy = np.ones_like(X) * ax.get_ylim3d()[0]
    cz = np.ones_like(Z) * ax.get_zlim3d()[0]

    ax1.scatter(X, Y, cz, c='steelblue', marker='.', s=2, alpha=.2)  #c=Z
    ax1.scatter(X, cy, Z, c='steelblue', marker='.', s=2, alpha=.2)  #c=-Y
    ax1.scatter(cx, Y, Z, c='steelblue', marker='.', s=2, alpha=.2)  #c=X

    ax1.scatter(X[I], Y[I], cz[I], c='darkorange', marker='o', s=5,
                alpha=.2)  #c=Z
    ax1.scatter(X[I], cy[I], Z[I], c='darkorange', marker='o', s=5,
                alpha=.2)  #c=-Y
    ax1.scatter(cx[I], Y[I], Z[I], c='darkorange', marker='o', s=5,
                alpha=.2)  #c=X

    ax1.set_xlim3d(ax.get_xlim3d())
    ax1.set_ylim3d(ax.get_ylim3d())
    ax1.set_zlim3d(ax.get_zlim3d())

    if save_figure:
        if not os.path.exists('images'): os.mkdir('images')
        plt.savefig(f'images/scatter_set_3d.{format}',
                    bbox_inches='tight',
                    orientation='landscape',
                    pad_inches=.1)
    if show_figure: plt.show()
    plt.clf()
    plt.close('all')


def plot_scatter_matrix_front(res,
                              I,
                              show_figure=True,
                              save_figure=False,
                              format='pdf'):
    sns.set_style('ticks', {
        'spines.linewidth': .25,
    })
    front_df = pd.DataFrame(res.F,
                            columns=[
                                'HST mass, kg', 'Total HST efficiency, %',
                                'Transmitted power, kW'
                            ]).abs()
    knee = []
    for i in front_df.index:
        if i in I:
            knee.append('High')
        else:
            knee.append('Low')
    front_df['Trade-off'] = pd.Series(knee)
    g = sns.pairplot(
        front_df,
        diag_kind='kde',
        hue='Trade-off',
        markers=['.', 'o'],
        plot_kws={'s': 20},
    )
    g.map_lower(sns.kdeplot, levels=4, linewidths=.5)
    g.map_upper(sns.kdeplot, levels=4, linewidths=.5)
    g._legend.remove()
    if save_figure:
        if not os.path.exists('images'): os.mkdir('images')
        plt.savefig(f'images/scatter_matrix_front.{format}',
                    bbox_inches='tight',
                    orientation='landscape',
                    pad_inches=.1)
    if show_figure: plt.show()
    plt.clf()
    plt.close('all')


def plot_scatter_matrix_set(res,
                            I,
                            show_figure=True,
                            save_figure=False,
                            format='pdf'):
    set_df = pd.DataFrame(res.X,
                          columns=[
                              'Displacement, cc/rev', 'Pump speed ,rpm',
                              'Pressure differential, bar'
                          ])
    knee = []
    for i in set_df.index:
        if i in I:
            knee.append('High')
        else:
            knee.append('Low')
    set_df['Trade-off'] = pd.Series(knee)
    sns.set_style('ticks', {
        'spines.linewidth': .25,
    })
    g = sns.pairplot(
        set_df,
        diag_kind='kde',
        hue='Trade-off',
        markers=['.', 'o'],
        plot_kws={'s': 20},
    )
    g.map_upper(sns.kdeplot, levels=4, linewidths=.5)
    g.map_lower(sns.kdeplot, levels=4, linewidths=.5)
    g._legend.remove()
    if save_figure:
        if not os.path.exists('images'): os.mkdir('images')
        plt.savefig(f'images/scatter_matrix_set.{format}',
                    bbox_inches='tight',
                    orientation='landscape',
                    pad_inches=.1)
    if show_figure: plt.show()
    plt.clf()
    plt.close('all')


def plot_convergence(res, show_figure=True, save_figure=False, format='pdf'):
    n_evals = np.array([e.evaluator.n_eval for e in res.history])
    val = np.array([e.pop.get("F") for e in res.history])
    opt = val.min(axis=1)[:]
    sns.set_style('ticks', {
        'spines.linewidth': .25,
    })
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Evaluations')
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    for i, ttl, axis, col in zip(
            range(3),
        ['HST mass, kg', 'Total HST efficiency, %', 'Transmitted power, kW'],
        (ax1, ax2, ax3), ('steelblue', 'crimson', 'gold')):
        axis.set_ylabel(ttl)
        axis.spines['left'].set_position(('axes', -.15 * i))
        axis.yaxis.set_label_position('left')
        axis.yaxis.set_ticks_position('left')
        sns.lineplot(x=n_evals, y=abs(opt[:, i]), ax=axis, color=col)
    sns.despine()
    fig.legend(
        labels=[
            'HST mass, kg', 'Total HST efficiency, %', 'Transmitted power, kW'
        ],
        loc='upper right',
        bbox_to_anchor=(.9, .9),
    )
    if save_figure:
        if not os.path.exists('images'): os.mkdir('images')
        plt.savefig(f'images/convergence.{format}',
                    bbox_inches='tight',
                    orientation='landscape',
                    pad_inches=.1)
    if show_figure: plt.show()
    plt.clf()
    plt.close('all')


def main():
    show = True
    save = False
    ha.print_to_moohst(show=show, save=save)
    problem = MOOHST()
    algorithm = NSGA2(pop_size=450)
    res = minimize(problem,
                   algorithm, ("n_gen", 25),
                   verbose=True,
                   seed=3,
                   save_history=True)
    dm = get_decision_making("high-tradeoff")
    I = dm.do(res.F)
    #* Plotting
    plot_scatter_front_3d(res, I, show_figure=show, save_figure=save)
    plot_scatter_set_3d(res, I, show_figure=show, save_figure=save)
    plot_scatter_matrix_front(res, I, show_figure=show, save_figure=save)
    plot_scatter_matrix_set(res, I, show_figure=show, save_figure=save)
    plot_convergence(res, show_figure=show, save_figure=save)


if __name__ == '__main__':
    main()