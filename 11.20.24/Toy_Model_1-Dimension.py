from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, CompoundKernel,RationalQuadratic, Matern, DotProduct,ConstantKernel,ExpSineSquared  #try out some different kernels
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel("data.xlsx")
print(data.shape)

np_data = data.to_numpy()
x_data=np_data[:,0]
y_data=np_data[:,1]


print("X data:", x_data)
print("Standard deviation of data:", x_data.std())
print("Mean of data:", x_data.mean())

print("Y data:", y_data)
print("Standard deviation of data:", y_data.std())
print("Mean of data:", y_data.mean())

#simulate randomness within regions between data points
#product and sum are both viable, however, we depend on the kernel we are using; combined kernel formally does so
#kernel = RBF(length_scale=0.0001483459) #reduce to have more rough#SPIKE/OSCILLATORY motion for 0.0001483459 and smaller----subset of Matern but much smoother for even very small scales
#kernel = Matern(length_scale=0.00050853459) #MANY interstections between different curves more rough edges between data points--general for RBF same oscillatory for small: the second derivitive is near undefined
#kernel = DotProduct(sigma_0=3.1001483459)#goes to a line for small and large values...,
#kernel = WhiteKernel(length_scale=0.0001483459) #allows for adding NOISE to the dataset, or uncertainty (MUST be added)
#kernel = RBF(length_scale=0.1001483459)*ConstantKernel(constant_value=2)
kernel = ExpSineSquared(length_scale=0.1001483459)
#ConstantKernel(constant_value=2) #the constant kernel will add a constant to previous kernels (must be added to something)
#kernel = RationalQuadratic(length_scale=0.0001483459)


random_curve = GaussianProcessRegressor(kernel=kernel,alpha=1e-2)
random_curve.fit(x_data.reshape(-1,1),y_data)
x_new=np.linspace(0,0.5,1000).reshape(-1,1) #the start and stop functions will create 1000 points (NOT DATA POINTS but GPR generated points)
y_new=random_curve.sample_y(x_new, 1, random_state=1000)
plt.figure(figsize=(10,5))

plt.xlabel("x--[Temperature (GEV)]")
plt.ylabel("y--[P$T^{-4}$]")
i=np.random.seed(12) #this from the random library allows us to create ANYN set of n lines (we set n to be 3 for now)

plt.plot(x_data,y_data, lw=1, ls='--', label=f'Predictive Random Curve 1')



for n in range (3): #we can change the range value here to incoporate curves 0 to n-1, in this case n=3
    plt.plot(x_new,random_curve.sample_y(x_new, 1, random_state=i), lw=1, ls='--', label=f'Predictive Random Curve {n+1}')

plt.scatter(x_data,y_data, marker='x', color='r', s=10,label='X=Data Points')
plt.legend(title="Legend" ,loc='lower center', fontsize='large' )
legend = plt.legend()
# Changing the color of each text in the legend
for text in legend.get_texts():
    text.set_color("Purple")
plt.show()


