b1 = self.ax.bar(
    x - width/2, data_off, width,
    label=self.series_labels[0],
    color='gray'
)
b2 = self.ax.bar(
    x + width/2, data_on,  width,
    label=self.series_labels[1],
    color='red'
)