import numpy as np
import streamlit as st 
import matplotlib.pyplot as plt
import math
import seaborn as sn

st.set_option('deprecation.showPyplotGlobalUse', False)

# Write a title
st.title('Sigmoid Function')
# define sidebar
st.sidebar.markdown("Choose options here:")

def update():
    alpha = st.session_state.alpha

alpha = st.sidebar.slider('alpha', 0., 10., step=0.01, value=5., key='alpha',on_change=update)

def sigmoid(x, alpha):
    return 1. / (1. + math.exp(-alpha*x))

def sigmoid_2(x, y, alpha):
    return 1. / (1. + np.exp(-alpha*(x + y)))

fig, ax = plt.subplots(figsize=(16,8))

## first figure
x = np.linspace(-10, 10, 200)
vfunc = np.vectorize(sigmoid)
y = vfunc(x,alpha)

ax.set_title('1-dim Sigmoid')
ax.plot(x,y)
st.pyplot(fig)

#second figure
fig, axs = plt.subplots(figsize=(16,8))
#sn.set_style('whitegrid')
X_VAL = np.linspace(-10, 10, 100)
Y_VAL = np.linspace(-10, 10, 100)

# Create a grid
X, Y = np.meshgrid(X_VAL, Y_VAL)
# Compute the sigmoid function for each pair of x and y coordinates
Z = sigmoid_2(X, Y, alpha)

axs = plt.axes(projection='3d')
axs.plot_surface(X, Y, Z,cmap='viridis')
# Set labels and title
axs.set_xlabel('X')
axs.set_ylabel('Y')
axs.set_zlabel('Sigmoid Value')
axs.set_title('3D Sigmoid Function')
print('Hi')
st.pyplot(fig)


