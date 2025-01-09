from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import matplotlib.pyplot as plt

x_data=np.array([0.0809533, 0.0919485,
 0.114347, 0.129381, 0.134471, 0.425716, 0.440864,0.477756, 0.478615,  0.485316, 0.486947
        ]).reshape(-1, 1)                       #the x value here must be reshaped so we only input a flat array

y_data=np.array([0.215936,0.248158, 0.332308, 0.423743, 0.462652, 3.78217, 3.84422, 3.97498,
 3.97773, 3.99871, 4.00371
])

kernel = RBF(length_scale=0.001)


random_curve=GaussianProcessRegressor(kernel=kernel,alpha=1e-2)
random_curve.fit(x_data,y_data)

x_new=np.linspace(0,0.5,1000).reshape(-1,1) #the start and stop functions will create 1000 points (NOT DATA POINTS but GPR generated points)
y_mean, y_std = random_curve.predict(x_new, return_std=True)



plt.figure(figsize=(10,5))

plt.xlabel("x--[Temperature (GEV)]")
plt.ylabel("y--[P$T^{-4}$]")
plt.scatter(x_data,y_data, marker='o', color='r', s=15,label='X=Data Points',edgecolors='black',linewidths=1.5)
i=np.random.seed(10) #this from the random library allows us to create ANYN set of n lines (we set n to be 3 for now)
for n in range (10): #we can change the range value here to incoporate curves 0 to n-1, in this case n=3
    plt.plot(x_new, random_curve.sample_y(x_new, 1, random_state=i), lw=1, ls='--', label=f'Predictive Random Curve {n+1}')


plt.fill_between(x_new[:,0], y_mean - 1 * y_std, y_mean + 1 * y_std, color='red', label='68% confidence level',alpha=0.2)
plt.fill_between(x_new[:,0], y_mean-2 * y_std,y_mean+2 * y_std, color='orange', label='95% confidence level',alpha=0.2)
plt.fill_between(x_new[:,0], y_mean-3 * y_std,y_mean+3 * y_std, color='yellow', label='99.7% confidence level',alpha=0.2)

plt.legend(title="Legend" ,loc='lower center', fontsize='large' )
legend = plt.legend()


for text in legend.get_texts():
    text.set_color("Purple")
plt.show()