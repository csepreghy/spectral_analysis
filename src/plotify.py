from matplotlib import rcParams
import matplotlib.font_manager
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np


class Plotify:
  def __init__(self):
    # Basic configuration
    self.use_grid = True

    plt.style.use('dark_background')

    # Color Constants
    self.background_color = '#1C2024'
    self.grid_color = '#444444'
    self.legend_color = '#282D33'
    self.c_white = '#FFFFFF'
    self.c_cyan = '#4FB99F'
    self.c_orange = '#F2B134'
    self.c_red = '#ED553B'
    self.c_green = '#62BF04'
    self.c_blue = '#189BF2'
    self.c_pink = '#FF697C'
    self.c_purple = '#EEA5FF'

    self.plot_colors = [self.c_orange, self.c_cyan, self.c_red, self.c_green, self.c_blue]

    rcParams.update({'font.sans-serif': 'Arial'})

  def get_colors(self):
    colors = {
      'orange': self.c_orange,
      'cyan': self.c_cyan,
      'red': self.c_red,
      'blue': self.c_blue,
      'green': self.c_green,
      'pink': self.c_pink,
      'purple': self.c_purple,
      'white': self.c_white
    }

    return colors

  def boxplot(self, data, labels, title, ylabel):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(self.background_color)
    ax.set_facecolor(self.background_color)

    bplot = ax.boxplot(
      data,
      vert=True,
      patch_artist=True,
      labels=labels,
      boxprops=dict(facecolor=self.c_white, color=self.c_white),
      capprops=dict(color=self.c_white),
      whiskerprops=dict(color=self.c_white),
      flierprops=dict(markeredgecolor=self.c_white),
      medianprops=dict(color=self.c_white)
    )

    for patch, color in zip(bplot['boxes'], self.plot_colors):
      patch.set_facecolor(color)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend((bplot['boxes']), labels, loc=2, facecolor=self.legend_color)

    plt.subplots_adjust(top=0.85)
    plt.grid(self.use_grid, color=self.grid_color)

    plt.show()

  def scatter_plot(
    self,
    x_list,
    y_list,
    linewidth=0.5,
    alpha=1,
    xlabel='X label',
    ylabel='Y label',
    title='Title',
    legend_labels=(''),
    arrows=[],
    equal_axis=False,
    tickfrequencyone=True,
    show_plot=True,
    ax=None
  ):
    if ax == None: _, ax = self.get_figax()

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if tickfrequencyone == True:
      ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    for i, x in enumerate(x_list):
      ax.scatter(
        x,
        y_list[i],
        linewidths=linewidth,
        alpha=alpha,
        c=self.plot_colors[i],
        edgecolor='#333333'
      )

    if len(arrows) > 0:
      for arrow in arrows:
        plt.arrow(
          x=arrow['x'],
          y=arrow['y'],
          dx=arrow['dx'],
          dy=arrow['dy'],
          width=arrow['width'],
          color=arrow['color'],
          alpha=0.8
        )

    ax.grid(self.use_grid, color=self.grid_color)
    ax.legend(legend_labels, facecolor=self.legend_color)

    if equal_axis == True:
      plt.axis('equal')

    #plt.savefig((title + str(np.random.rand(1)[0]) + '.png'),
    #            facecolor=self.background_color, dpi=180)

    if show_plot == True:
      plt.show()

  def scatter3d(
    self,
    x,
    y,
    z,
    linewidth=0.5,
    alpha=1,
    xlabel='X label',
    ylabel='Y label',
    zlabel='Z label',
    title='Title',
    arrows=[],
    equal_axis=False,
    show=True
  ):
    _, ax = self.get_figax3d()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.scatter(
      x,
      y,
      z,
      alpha=alpha,
      c=self.c_orange,
      edgecolor='#555555'
    )

    ax.grid(self.use_grid, color=self.grid_color)

    if equal_axis == True:
      plt.axis('equal')

    # plt.savefig((title + str(np.random.rand(1)[0]) + '.png'), facecolor=self.background_color, dpi=180)
    if show == True: plt.show()

  def histogram(
    self,
    x_list,
    ylabel='Y label',
    xlabel='X label',
    title='Title',
    labels=('Label 1', 'Label 2')
  ):
    fig, ax = self.get_figax()

    for i, x in enumerate(x_list):
      ax.hist(x, int(np.max(x) - np.min(x)), facecolor=self.plot_colors[i])

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(labels, facecolor=self.legend_color)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

  def bar(
    self,
    x_list,
    y_list,
    ylabel='Y label',
    xlabel='X label',
    title='Title',
    ymin=0,
    ymax=None,
    linewidth=0.8,
    use_x_list_as_xticks=False,
    xticks=[],
    rotation=0,
    show=True
  ):
    _, ax = self.get_figax()

    ax.bar(x_list, height=y_list, width=linewidth, color=self.c_orange)
    ax.set_ylim(ymin=ymin)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    if ymax != None:
      ax.set_ylim(ymax=ymax)

    if use_x_list_as_xticks == True:
      plt.xticks(x_list)

    if len(xticks) > 0:
      ax.set_xticklabels(xticks)
      ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.xticks(rotation=rotation)
    plt.tight_layout()
    if show == True:
      plt.show()

    # plt.savefig((title + str(np.random.rand(1)[0]) + '.png'), facecolor=self.background_color, dpi=120)

    return ax

  def plot(
    self,
    y,
    ylabel='Y label',
    xlabel='X label',
    title='Title',
    show_plot=True,
    use_x_list_as_xticks=True,
    tickfrequencyone=False,
    equal_axis=False,
    x=[],
    figsize=(8,6),
    filename='filename',
    ymin=None,
    ymax=None,
    xmin=None,
    xmax=None,
    save=False,
    label='',
    color='orange'
  ):
    colors = self.get_colors()
    fig, ax = self.get_figax(figsize=figsize)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if ymin is not None: ax.set_ylim(ymin=ymin)
    if ymax is not None: ax.set_ylim(ymax=ymax)

    if xmin is not None: ax.set_xlim(xmin=xmin)
    if xmax is not None: ax.set_xlim(xmax=xmax)
    
    if equal_axis == True:
      plt.axis('equal')

    if tickfrequencyone == True:
      ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    if len(x) == 0: plt.plot(x, color=colors[color], label=label)
    if len(x) > 0: 
      print('x', x)
      print('y', y)
      plt.plot(x=x, y=y, color=colors[color], label=label)
    
    if save == True: plt.savefig(('plots/' + filename), facecolor=self.background_color, dpi=180)

    if show_plot == True:
      plt.show()

    return fig, ax

  def get_figax(self, is3d=False, figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(self.background_color)

    ax.set_facecolor(self.background_color)
    ax.tick_params(colors=self.c_white)
    ax.xaxis.label.set_color(self.c_white)
    ax.yaxis.label.set_color(self.c_white)
    ax.grid(self.use_grid, color=self.grid_color)

    return fig, ax

  def get_figax3d(self):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    fig.patch.set_facecolor(self.background_color)

    ax.set_facecolor(self.background_color)
    ax.tick_params(colors=self.c_white)
    ax.xaxis.label.set_color(self.c_white)
    ax.yaxis.label.set_color(self.c_white)
    ax.grid(self.use_grid, color=self.grid_color)

    return fig, ax
