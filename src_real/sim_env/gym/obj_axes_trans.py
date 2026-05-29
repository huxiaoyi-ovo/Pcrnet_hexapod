# isaac gym读取.obj时，会将原有的坐标系绕x轴旋转-90°
# 将obj文件的坐标系绕x轴旋转90°，计算点在新坐标系下的表示
# 也就是将.obj文件的所有点绕x轴旋转-90度，并保存为新的.obj文件，抵消影响
# 遍历input_dir下所有.obj文件，将其坐标系绕x轴旋转-90°，并保存在output_dir下新的.obj文件
import os
import numpy as np

def rotate_vertices(vertices):
    rot_mat=np.array([
        [1,0,0],
        [0,0,1],
        [0,-1,0]])
    col_vertices=vertices.T
    return (np.dot(rot_mat,col_vertices)).T

def modify_obj_file(input_file,output_file):
    with open(input_file,'r') as infile:
        lines=infile.readlines()
    vertices=[]
    normals=[]
    for i,line in enumerate(lines):
        if line.startswith('v '):
            parts=line.split()
            if len(parts)==4:
                x,y,z=map(float,parts[1:4])
                vertices.append([x,y,z])
        elif line.startswith('vn '):
            parts=line.split()
            if len(parts)==4:
                x,y,z=map(float,parts[1:4])
                normals.append([x,y,z])

    vertices=np.array(vertices)
    normals=np.array(normals)

    rotated_vertices=rotate_vertices(vertices)
    rotated_normals=rotate_vertices(normals)
    with open(output_file,'w') as outfile:
        vertex_index=0
        normals_index=0
        for i,line in enumerate(lines):
            if line.startswith('v '):
                outfile.write(f"v {rotated_vertices[vertex_index][0]} {rotated_vertices[vertex_index][1]} {rotated_vertices[vertex_index][2]}\n")
                vertex_index+=1
            elif line.startswith('vn '):
                outfile.write(f"vn {rotated_normals[normals_index][0]} {rotated_normals[normals_index][1]} {rotated_normals[normals_index][2]}\n")
                normals_index+=1
            else:
                outfile.write(line)

def process_all_obj_files(input_dir,output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.obj'):
            input_file=os.path.join(input_dir,filename)
            # filename=filename.replace('.obj','_rot.obj')
            output_file=os.path.join(output_dir,filename)
            modify_obj_file(input_file,output_file)

if __name__=='__main__':
    # input_dir='/home/val/BIH_ws/hex_sim2/src/sim_env/model/hex_v4/meshes'
    # input_dir='/home/val/BIH_ws/hex_sim2/src/sim_env/model/hex_v4/two_set_meshes/collision'
    input_dir='/home/val/BIH_ws/hex_sim2/src/sim_env/model/hex_v4/two_set_meshes/visual'
    # output_dir='/home/val/BIH_ws/hex_sim2/src/sim_env/model/hex_v4/meshes/rot'
    # output_dir='/home/val/BIH_ws/hex_sim2/src/sim_env/model/hex_v4/two_set_meshes/collision/rot'
    output_dir='/home/val/BIH_ws/hex_sim2/src/sim_env/model/hex_v4/two_set_meshes/visual/rot'
    process_all_obj_files(input_dir,output_dir)