import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_correlation_heatmap():
    
    numeric_data = np.array([
        [-0.3307, 0.3922, 0.4611, 0.4631, 0.4652],
        [-0.1142, 0.4476, 0.3918, 0.1578, 0.1587],
        [-0.1714, 0.4149, 0.2125, 0.2734, 0.3787],
        [-0.1747, 0.1580, 0.1334, 0.0432, 0.1359],
        [-0.0584, 0.1523, 0.1771, -0.0131, 0.1125]
    ])

   
    annot_data = np.array([
        ['-0.3307***', '0.3922***', '0.4611***', '0.4631***', '0.4652***'],
        ['-0.1142***', '0.4476***', '0.3918***', '0.1578***', '0.1587***'],
        ['-0.1714***', '0.4149***', '0.2125***', '0.2734***', '0.3787***'],
        ['-0.1747***', '0.1580***', '0.1334***', '0.0432***', '0.1359***'],
        ['-0.0584***', '0.1523***', '0.1771***', '-0.0131 (n.s)', '0.1125***']
    ])

   
    movies = ['Chatter', 'BigBuckBunny', 'AfterTheRain', 'Between Viewings', 'Superhero']
    parameters = ['Predictable', 'Alert', 'Attention', 'HeartRate', 'IntenseEmotion']

    
    df = pd.DataFrame(numeric_data, index=movies, columns=parameters)

    
    plt.figure(figsize=(10, 6))
    
    
    ax = sns.heatmap(
        df, 
        annot=annot_data, 
        fmt="",                  
        cmap="RdYlGn",              
        center=0,                
        vmin=-0.5, vmax=0.5,     
        cbar_kws={'label': 'Adjusted Partial Correlation (r)'},
        linewidths=1,             
        linecolor='white'
    )

    plt.title('Figure 2:Correlation between  SVR predictions of arousal and the 5 emotional annotations', pad=20, fontsize=12, fontweight='bold', y=-0.45)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    
    
    plt.show()

if __name__ == "__main__":
    generate_correlation_heatmap()