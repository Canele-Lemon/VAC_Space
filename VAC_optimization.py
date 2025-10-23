def MatFormat_AxisTitle(ax, axis_title, font=None, color='#595959', fontsize=9, fontweight='normal', axis='x', show_labels=True):
    title_font = {
        'family': font,
        'color': color,
        'weight': fontweight,
        'size': fontsize,
    }
    if axis == 'x':
        if show_labels:
            ax.set_xlabel(axis_title, fontdict=title_font)
            ax.tick_params(axis='x', labelbottom=True)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False)

    elif axis == 'y':
        ax.set_ylabel(axis_title, fontdict=title_font)
    
    return ax

## 축 서식 ##
def MatFormat_Axis(ax, min_val, max_val, tick_interval=None, tick_color='#bfbfbf', label_font=None, label_color='#595959', label_fontsize=9, axis='x', formatter=None):
    if axis == 'x':
        ax.set_xlim(min_val, max_val)
        if tick_interval is not None:
            ax.set_xticks(np.arange(min_val, max_val + tick_interval, tick_interval))
        else:
            print("inside xtick=none")
            pass
            # ax.set_xticks(np.arange(min_val, max_val))
        ax.tick_params(axis='x', labelsize=label_fontsize, labelcolor=label_color, colors=tick_color, direction='in')
    elif axis == 'y':
        ax.set_ylim(min_val, max_val)
        if tick_interval is not None:
            ax.set_yticks(np.arange(min_val, max_val + tick_interval, tick_interval))
        else:
            pass
            # ax.set_yticks(np.arange(min_val, max_val))
        # ax.set_yticks(np.arange(min_val, max_val + tick_interval, tick_interval))
        ax.tick_params(axis='y', labelsize=label_fontsize, labelcolor=label_color, colors=tick_color, direction='in')
        if formatter:
            ax.yaxis.set_major_formatter(formatter)
    return ax
