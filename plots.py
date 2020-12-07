import numpy as np
import matplotlib.pyplot as plt



def plot_train_val(m_train, m_val, period, metric):
    
    #plt.figure(figsize=(8,8.5))
    plt.title('Evolution of the '+metric+ ' with respect to the number of epochs',fontsize=14)

    plt.plot(np.array(range(1,len(m_train)+1))*period, m_train, color='blue', marker='*', ls=':', label=metric+' train')
    plt.plot(np.array(range(1,len(m_val)+1))*period, m_val, color='red', marker='*', ls=':', label=metric+' val')

    plt.xlabel('Number of Epochs')
    plt.ylabel(metric)
    plt.legend(loc = 'upper right')
    plt.show()
    