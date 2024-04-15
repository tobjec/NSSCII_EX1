import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_speedup(df):
    # convert dict to dataframe

    df['Resolution 125 speedup'] = df['time 125'][0] / df['time 125']
    df['Resolution 250 speedup'] = df['time 250'][0] / df['time 250']
    df['Resolution 1000 speedup'] = df['time 1000'][0] / df['time 1000']
    df['Resolution 2000 speedup'] = df['time 2000'][0] / df['time 2000']

    df_plot = df[['cores','Resolution 125 speedup','Resolution 250 speedup','Resolution 1000 speedup','Resolution 2000 speedup']]

    print(df_plot)

    plt.figure()
    ax = df_plot.plot(x='cores')
    plt.suptitle(r'$\bf{Benchmark\ 1D-Decomposition}$', fontsize=12)
    plt.title('Parallel Speedup', fontsize=10)
    ax.set_xlabel('# of CPU cores')
    ax.set_ylabel(r'$Speedup\ (\frac{T_1}{T_n})$')

    ax.figure.savefig("speedup.png")

def plot_efficiency(df):
    # convert dict to dataframe
    df = pd.read_table("results.csv", sep=',')

    df['Resolution 125 efficiency'] = df['time 125'][0] / (df['cores'] * df['time 125'])
    df['Resolution 250 efficiency'] = df['time 250'][0] / (df['cores'] * df['time 250'])
    df['Resolution 1000 efficiency'] = df['time 1000'][0] / (df['cores'] * df['time 1000'])
    df['Resolution 2000 efficiency'] = df['time 2000'][0] / (df['cores'] * df['time 2000'])

    df_plot = df[['cores','Resolution 125 efficiency','Resolution 250 efficiency','Resolution 1000 efficiency','Resolution 2000 efficiency']]

    print(df_plot)

    plt.figure()
    ax = df_plot.plot(x='cores')
    plt.suptitle(r'$\bf{Benchmark\ 1D-Decomposition}$', fontsize=12)
    plt.title('Parallel Efficiency', fontsize=10)
    ax.set_xlabel('# of CPU cores')
    ax.set_ylabel(r'$Efficiency\ (\frac{T_1}{n*T_n})$')

    ax.figure.savefig("efficiency.png")



df = pd.read_table("results.csv", sep=',')
print(df)

plot_speedup(df)
plot_efficiency(df)