from utils import *
import utils
import os
import sys
import glob
import inspect
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly.offline as plot
import numpy as np
import random

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
# User's modules from another directory
sys.path.append(os.path.join(parentdir, "bases"))
COLOR_PALETTE = [(255, 23, 68), (240, 98, 146), (170, 0, 255), (124, 67, 189), (48, 79, 254), (26, 35, 126), (41, 121, 255),
                 (0, 145, 234), (24, 255, 255), (0, 230,
                                                 118), (50, 203, 0), (0, 200, 83), (255, 255, 0),
                 (255, 111, 0), (172, 25, 0), (84, 110, 122), (213, 0, 0), (250, 27, 27)]


def showLineChart(list_axes: list, names: list, title=None, x_title=None, y_title=None, special_points=[]):

    fig = go.Figure()
    if len(list_axes) != len(names):
        raise Exception("Names list length doesn't match axises list length.")
    colors_set = random.sample(COLOR_PALETTE, len(list_axes))
    frames=[]
    x_size=len(list_axes[0][0])
    for i in range(x_size):
        frame={'data': [], 'name': 'frame'+str(i)}
        dic=lambda k,l,x,y:{'type':'scatter','name':names[l],'x':x[:k], 'y':y[:k]}
        
        # l=[go.Frame(data=[go.Scatter(x=x[:i], y=y[:i])]) for (x,y) in list_axes]
        frame['data']=[dic(i,j,x,y) for j,(x,y) in enumerate(list_axes)]
        frames.append(go.Frame(frame.copy()))

    for i, (x, y) in enumerate(list_axes):
    
        fig.add_trace(go.Scatter(x=x, y=y, name=names[i], line=dict(color='rgb'+str(colors_set[i])),
                                 hovertemplate='<b>Epoch:</b> %{x} <br><b>Error:</b> %{y:.2f}',
                                marker={'size':[6 if (x[j],y[j]) not in [(a,b) for (a,b,c) in special_points] else 15 for j in range(len(x))]}
                                )
           )
 
    fig.frames=frames.copy()
    fig.update_layout(title={'text': title if title is not None else '',
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'
                             },
                      xaxis_title=x_title if x_title is not None else '',
                      yaxis_title=y_title if y_title is not None else '',
                      hoverlabel=dict(
        font_size=16,
        font_family="Rockwell"
    ),
        font=dict(
                          family="Courier New, monospace",
                          size=18,
                          color="#7f7f7f"
    ),
    updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None,dict(frame={'duration':1},fromcurrent=True,
                                                         mode='immediate')])])
                ],
        annotations=[
        dict(
            x=pt[0],
            y=pt[1],
            xref="x",
            yref="y",
            text=pt[2],
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#636363"
            ),
            align="center",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=0,
            ay=-50,
            opacity=0.8
        ) for i,pt in enumerate(special_points)
    ],
    transition={'duration': 1,'easing': 'bounce-in-out'}
    )
    fig.update_traces(mode="markers+lines")
    # plot.iplot(fig, filename='jupyter-basic_bar',animation_opts={'frame':{'duration':1}})
    fig.show(animation_opts={'frame':{'duration':100}})
    return fig


if __name__ == "__main__":
    # x1 = [2, 4, 6, 8, 9, 11, 15, 19]
    # x2 = [3, 4, 6, 7, 10, 12, 16, 20]
    # x3 = [2, 4, 7, 8, 9, 11, 15, 22]
    # x4 = [2, 4, 6, 8, 9, 11, 15, 19]
    # y1 = [2.12, 4.12, 6.12, 8.12, 9.12, 11.12, 15.12, 19]
    # y2 = [30.1, 4.8, 1.8, 12.8, 4.8, 22.8, 20.8, 19.2]
    # y3 = [6.5, 1.5, 32.5, 4.5, 5.5, 22.5, 10.5, 19]
    # y4 = [22.12, 14.12, 46.12, 38.12, 9.12, 13.2, 15.2, 19]
    # names = ['D1', 'D6', 'D5', 'D4']
    # liste = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    # showLineChart(liste, names, title="Line plot", special_points=[
    #               (2, 2.12, 'min error'), (19, 19, 'minimum_error')])
    ###################################################################################""""
    # x = np.array(range(0, 30, 1))
    # y = np.array(range(15, 45, 1))

    # l = [go.Frame(data=[go.Scatter(x=x[:i], y=y[:i])]) for i in range(1, 30)]

    # fig = go.Figure(
    #     data=[go.Scatter(x=[x[0]], y=[y[0]])],
    #     layout=go.Layout(
    #         xaxis=dict(range=[0, 5], autorange=False),
    #         yaxis=dict(range=[0, 5], autorange=False),
    #         title="Start Title",
    #         updatemenus=[dict(
    #             type="buttons",
    #             buttons=[dict(label="Play",
    #                           method="animate",
    #                           args=[None,{'frame':{'duration':1}}])])],
    #         transition={'duration': 0.00002}
            
    #     ),
    #     frames=l,

    # )

    # # fig.show()
    # # fig.layout.updatemenus[0].buttons[0].click()
    # plot.iplot(fig,animation_opts={'frame':{'duration':1}})



    # test_data=np.random.randint(25, 100, (24, 31))
    # data=[dict(type='scatter',
    #       x=list(range(24)),
    #       y=test_data[:0],
    #  mode='markers',
    #  marker=dict(size=10, color='red'))
    #  ]
    # frames=[dict(data=[dict(y=test_data[:,k])],
    #          traces=[0],
    #          name=f'{k+1}') for k in range(31)]

    # fig=dict(data=data, frames=frames)
    # plot.iplot(fig)         
    pass


# fig = dict(
# layout = dict(
#     xaxis1 = {'domain': [0.0, 0.44], 'anchor': 'y1', 'title': '1', 'range': [-2.25, 3.25]},
#     yaxis1 = {'domain': [0.0, 1.0], 'anchor': 'x1', 'title': 'y', 'range': [-1, 11]},
#     xaxis2 = {'domain': [0.56, 1.0], 'anchor': 'y2', 'title': '2', 'range': [-2.25, 3.25]},
#     yaxis2 = {'domain': [0.0, 1.0], 'anchor': 'x2', 'title': 'y', 'range': [-1, 11]},
#     title  = '',
#     margin = {'t': 50, 'b': 50, 'l': 50, 'r': 50},
#     updatemenus = [{'buttons': [{'args': [['0', '1', '2', '3'], {'frame': {'duration': 500.0, 'redraw': False}, 'fromcurrent': True, 'transition': {'duration': 0, 'easing': 'linear'}}], 'label': 'Play', 'method': 'animate'}, {'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}], 'label': 'Pause', 'method': 'animate'}], 'direction': 'left', 'pad': {'r': 10, 't': 85}, 'showactive': True, 'type': 'buttons', 'x': 0.1, 'y': 0, 'xanchor': 'right', 'yanchor': 'top'}],
#     sliders = [{'yanchor': 'top', 'xanchor': 'left', 'currentvalue': {'font': {'size': 16}, 'prefix': 'Frame: ', 'visible': True, 'xanchor': 'right'}, 'transition': {'duration': 500.0, 'easing': 'linear'}, 'pad': {'b': 10, 't': 50}, 'len': 0.9, 'x': 0.1, 'y': 0, 
#                 'steps': [{'args': [['0'], {'frame': {'duration': 500.0, 'easing': 'linear', 'redraw': False}, 'transition': {'duration': 0, 'easing': 'linear'}}], 'label': '0', 'method': 'animate'}, 
#                           {'args': [['1'], {'frame': {'duration': 500.0, 'easing': 'linear', 'redraw': False}, 'transition': {'duration': 0, 'easing': 'linear'}}], 'label': '1', 'method': 'animate'}, 
#                           {'args': [['2'], {'frame': {'duration': 500.0, 'easing': 'linear', 'redraw': False}, 'transition': {'duration': 0, 'easing': 'linear'}}], 'label': '2', 'method': 'animate'},
#                           {'args': [['3'], {'frame': {'duration': 500.0, 'easing': 'linear', 'redraw': False}, 'transition': {'duration': 0, 'easing': 'linear'}}], 'label': '3', 'method': 'animate'}, 
#                 ]}]
# ),
# data = [
#     {'type': 'scatter', 'name': 'f1', 'x': [-2.  , -1.  ,  0.01,  1.  ,  2.  ,  3.  ], 'y': [  4,   1,   1, 1,   4,   9], 'hoverinfo': 'name+text', 'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}}, 'line': {'color': 'rgba(255,79,38,1.000000)'}, 'mode': 'markers+lines', 'fillcolor': 'rgba(255,79,38,0.600000)', 'legendgroup': 'f1', 'showlegend': True, 'xaxis': 'x1', 'yaxis': 'y1'},
# {'type': 'scatter', 'name': 'f2', 'x': [-2.  , -1.  ,  0.01,  1.  ,  2.  ,  3.  ], 'y': [  2.5,   1,   1, 1,   2.5,   1], 'hoverinfo': 'name+text', 'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}}, 'line': {'color': 'rgba(79,102,165,1.000000)'}, 'mode': 'markers+lines', 'fillcolor': 'rgba(79,102,165,0.600000)', 'legendgroup': 'f2', 'showlegend': True, 'xaxis': 'x2', 'yaxis': 'y2'},
# ],
# frames = [
#     {'name' : '0', 'layout' : {},
#      'data': [
#          {'type': 'scatter', 'name': 'f1', 'x': [-2.  , -1.  ,  0.01,  1.  ,  2.  ,  3.  ], 'y': [  5,   8,   3, 2,   4,   0], 'hoverinfo': 'name+text', 'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}}, 'line': {'color': 'rgba(255,79,38,1.000000)'}, 'mode': 'markers+lines', 'fillcolor': 'rgba(255,79,38,0.600000)', 'legendgroup': 'f1', 'showlegend': True, 'xaxis': 'x1', 'yaxis': 'y1'}, 
# #              {'type': 'scatter', 'name': 'f1', 'x': [-2.  , -1.  ,  0.01,  1.  ,  2.  ,  3.  ], 'y': [  4,   7,   2, 1,   3,   0], 'hoverinfo': 'name+text', 'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}}, 'line': {'color': 'rgba(255,79,38,1.000000)'}, 'mode': 'markers+lines', 'fillcolor': 'rgba(255,79,38,0.600000)', 'legendgroup': 'f1', 'showlegend': True, 'xaxis': 'x1', 'yaxis': 'y1'},
#          {'type': 'scatter', 'name': 'f2', 'x': [-2.  , -1.  ,  0.01,  1.  ,  2.  ,  3.  ], 'y': [  3,   7,   4, 8,   5,   9], 'hoverinfo': 'name+text', 'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}}, 'line': {'color': 'rgba(79,102,165,1.000000)'}, 'mode': 'markers+lines', 'fillcolor': 'rgba(79,102,165,0.600000)', 'legendgroup': 'f2', 'showlegend': True, 'xaxis': 'x2', 'yaxis': 'y2'},
# #              {'type': 'scatter', 'name': 'f2', 'x': [-2.  , -1.  ,  0.01,  1.  ,  2.  ,  3.  ], 'y': [  2,   8,   3, 9,   4,   9], 'hoverinfo': 'name+text', 'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}}, 'line': {'color': 'rgba(79,102,165,1.000000)'}, 'mode': 'markers+lines', 'fillcolor': 'rgba(79,102,165,0.600000)', 'legendgroup': 'f2', 'showlegend': True, 'xaxis': 'x2', 'yaxis': 'y2'}
#      ],
#     },
#     {'name' : '1', 'layout' : {},
#      'data': [
#          {'type': 'scatter', 'name': 'f1', 'x': [-2.  , -1.  ,  0.01,  1.  ,  2.  ,  3.  ], 'y': [  4,   1,   1, 1,   4,   9], 'hoverinfo': 'name+text', 'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}}, 'line': {'color': 'rgba(255,79,38,1.000000)'}, 'mode': 'markers+lines', 'fillcolor': 'rgba(255,79,38,0.600000)', 'legendgroup': 'f1', 'showlegend': True, 'xaxis': 'x1', 'yaxis': 'y1'}, 
#          {'type': 'scatter', 'name': 'f2', 'x': [-2.  , -1.  ,  0.01,  1.  ,  2.  ,  3.  ], 'y': [  2.5,   1,   1, 1,   2.5,   1], 'hoverinfo': 'name+text', 'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}}, 'line': {'color': 'rgba(79,102,165,1.000000)'}, 'mode': 'markers+lines', 'fillcolor': 'rgba(79,102,165,0.600000)', 'legendgroup': 'f2', 'showlegend': True, 'xaxis': 'x2', 'yaxis': 'y2'}],
#     },
#     {'name' : '2', 'layout' : {},
#      'data': [
#          {'type': 'scatter', 'name': 'f1', 'x': [-2.  , -1.  ,  0.01,  1.  ,  2.  ,  3.  ], 'y': [  5,   8,   3, 2,   4,   0], 'hoverinfo': 'name+text', 'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}}, 'line': {'color': 'rgba(255,79,38,1.000000)'}, 'mode': 'markers+lines', 'fillcolor': 'rgba(255,79,38,0.600000)', 'legendgroup': 'f1', 'showlegend': True, 'xaxis': 'x1', 'yaxis': 'y1'}, 
#          {'type': 'scatter', 'name': 'f2', 'x': [-2.  , -1.  ,  0.01,  1.  ,  2.  ,  3.  ], 'y': [  3,   7,   4, 8,   5,   9], 'hoverinfo': 'name+text', 'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}}, 'line': {'color': 'rgba(79,102,165,1.000000)'}, 'mode': 'markers+lines', 'fillcolor': 'rgba(79,102,165,0.600000)', 'legendgroup': 'f2', 'showlegend': True, 'xaxis': 'x2', 'yaxis': 'y2'}],
#     },
#     {'name' : '3', 'layout' : {},
#      'data': [
#          {'type': 'scatter', 'name': 'f1', 'x': [-2.  , -1.  ,  0.01,  1.  ,  2.  ,  3.  ], 'y': [  4,   1,   1, 1,   4,   9], 'hoverinfo': 'name+text', 'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}}, 'line': {'color': 'rgba(255,79,38,1.000000)'}, 'mode': 'markers+lines', 'fillcolor': 'rgba(255,79,38,0.600000)', 'legendgroup': 'f1', 'showlegend': True, 'xaxis': 'x1', 'yaxis': 'y1'}, 
#          {'type': 'scatter', 'name': 'f2', 'x': [-2.  , -1.  ,  0.01,  1.  ,  2.  ,  3.  ], 'y': [  2.5,   1,   1, 1,   2.5,   1], 'hoverinfo': 'name+text', 'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}}, 'line': {'color': 'rgba(79,102,165,1.000000)'}, 'mode': 'markers+lines', 'fillcolor': 'rgba(79,102,165,0.600000)', 'legendgroup': 'f2', 'showlegend': True, 'xaxis': 'x2', 'yaxis': 'y2'}],
#     }
# ]
# )
# plot.iplot(fig)