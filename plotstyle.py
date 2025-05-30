import matplotlib.pyplot as plt
from cycler import cycler

# define an enviroment class to be used like tis:
# with PlotStyle():

class PlotStyle:
    def __init__(self, figsize=(10, 7), font_size=14, dpi=300):
        self.figsize = figsize
        self.font_size = font_size
        self.dpi = dpi

    def __enter__(self):
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['font.size'] = self.font_size
        plt.rcParams['figure.dpi'] = self.dpi
        # Use latex for text rendering
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern']
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        # style
        plt.style.use('seaborn-v0_8-bright')
        plt.rcParams['axes.prop_cycle'] = cycler(color=['#ff9900', '#cc3300', '#00cc99', '#339933'])

    def __exit__(self, exc_type, exc_value, traceback):
        # Reset to default
        plt.rcdefaults()
        plt.close('all')
        

        if exc_type is not None:
            print("An error occurred:", exc_value)
