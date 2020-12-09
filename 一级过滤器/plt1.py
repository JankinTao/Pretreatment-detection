import matplotlib.pyplot as plt
for i in [1,2,3,4,5,6]:
    a=2*i+1
    plt.plot(i,a,color="red",linewidth=1,linestyle='-', marker='*')
plt.show()