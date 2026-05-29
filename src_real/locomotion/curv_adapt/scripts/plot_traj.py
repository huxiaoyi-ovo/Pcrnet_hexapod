#绘制曲线
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
# ax1 = fig.add_subplot(121)

# ax2 = fig.add_subplot(122,projection='3d')
ax1=plt.subplot()
def plot_traj(y,z,y_r,z_r,t,t_r):
    ax1.plot(t, y, label='pitch(t)')
    ax1.plot(t_r, y_r, label='pitch_r(t)')
    ax1.plot(t, z, label='roll(t)')
    ax1.plot(t_r, z_r, label='roll_r(t)')
    ax1.set_xlabel('time(s)')
    ax1.set_ylabel('position(m)')
    ax1.legend()
# def plot_3d_traj(x, y, z, t):
#     ax2.plot(x, y, z, label='trajectory')
#     ax2.set_xlabel('x')
#     ax2.set_ylabel('y')
#     ax2.set_zlabel('z')
#     ax2.legend()

#从../data/B_e_traj_axes.txt中读取每一行，第一个元素为时间，后面为x,y,z坐标
def read_traj(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        traj = []
        for line in lines:
            data = line.strip().split()
            # if float(data[0])>1715131233.591608 and float(data[0])<1715131234:
            if float(data[0])>0.1:
                t = float(data[0])
                x = float(data[1])
                y = float(data[2])
                z = float(data[3])
                traj.append((t, x, y, z))
        return traj

if __name__ == '__main__':
    traj = read_traj('/home/val/BIH_ws/hexapod_sim/bag/body_pos_des.txt')
    traj_2=read_traj('/home/val/BIH_ws/hexapod_sim/bag/body_pos_real.txt')
    t = [x[0] for x in traj]
    x = [x[1] for x in traj]
    y = [x[2] for x in traj]
    z = [x[3] for x in traj]
    
    t_r=[x[0] for x in traj_2]
    x_r = [x[1] for x in traj_2]
    y_r = [x[2] for x in traj_2]
    z_r = [x[3] for x in traj_2]
    plot_traj(y,z,y_r,z_r,t,t_r)
    # plot_3d_traj(x, y, z, t)
    plt.show()


