#pragma once
#ifndef ROBOT_POLYHEDRA_H
#define ROBOT_POLYHEDRA_H

#include<Eigen/Dense>
// #include<Eigen/Geometry>
#include<vector>
#include<map>
#include<string>
#include"utils/ColorCout.h"
using namespace Eigen;
using namespace std;

//构建左右工作空间（多个凸包构成），身体凸包的v和h表达，检测一条线是否穿过工作空间或身体
//实例化需要输入string参数 “normal”或者“shrink”
class ROBOT_POLYHEDRA2
{
public:
    //polys_vertices[0]=[x1 y1;
    //                   x2 y2;...]
    
    vector<MatrixX2d> r_vertices,l_vertices,body_vertices;  //机器人真实的凸包边界
    vector<vector<MatrixX2d>*> lr_vertices_ptr;
    // MatrixX2d body_vertices;
    //r_coeff=[a11 a12
    //         a21 a22...]
    //r_b=[b1 b2 ...]
    //a11x+a22z>=b1
    vector<MatrixX2d> r_coeff,l_coeff,body_coeff;
    vector<vector<MatrixX2d>*> lr_coeff_ptr;
    // MatrixX2d body_coeff;
    vector<VectorXd> r_b, l_b, body_b;
    vector<vector<VectorXd>*> lr_b_ptr;
    // VectorXd body_b;
    string hedra_type;
    //type用于指定是真实的凸包边界还是向内缩水的边界
    ROBOT_POLYHEDRA2(string type):hedra_type(type){
        int polys_num=5;
        //初始化凸包的顶点
        r_vertices.resize(polys_num);
        l_vertices.resize(polys_num);
        body_vertices.resize(1);

        r_coeff.resize(polys_num);
        l_coeff.resize(polys_num);
        body_coeff.resize(1);

        r_b.resize(polys_num);
        l_b.resize(polys_num);
        body_b.resize(1);
        //将分配好空间的系数填入指针    
        lr_coeff_ptr.push_back(&l_coeff);
        lr_coeff_ptr.push_back(&r_coeff);
        lr_b_ptr.push_back(&l_b);
        lr_b_ptr.push_back(&r_b);
        lr_vertices_ptr.push_back(&l_vertices);
        lr_vertices_ptr.push_back(&r_vertices);


        //初始化身体凸包顶点
        int rows=8;
        for(int i=0;i<1;i++){
            body_vertices[i].resize(rows,2);
            body_coeff[i].resize(rows,2);
            body_b[i].resize(rows);
        }
        rows=4;
        for (int i=0;i<polys_num;i++){
            r_vertices[i].resize(rows,2);
            l_vertices[i].resize(rows,2);
            r_coeff[i].resize(rows,2);
            l_coeff[i].resize(rows,2);
            r_b[i].resize(rows);
            l_b[i].resize(rows);
        }
        if(hedra_type =="normal"){
            r_vertices[0]<<0.18,0.2, 0.27,0.2, 0.34,0, 0.15,0;
            r_vertices[1]<<0.15,0, 0.34,0, 0.26,-0.2, 0.135,-0.035;
            r_vertices[2]<<0.135,-0.035, 0.26,-0.2, 0.12,-0.27, 0.08,-0.075;
            r_vertices[3]<<0.08,-0.075, 0.12,-0.27, 0,-0.27, 0,-0.075;
            r_vertices[4]<<0,-0.075, 0,-0.27,-0.15,-0.18, -0.15,-0.075;          
            // body_vertices[0].block(0,0,4,2)<<0.23,0.1, 0.29,0, 0.262,-0.0645, 0.193,-0.115;
            body_vertices[0].block(0,0,4,2)<<0.23,0.1, 0.29,0, 0.262,-0.0445, 0.193,-0.1;
            // r_vertices[0]<<0.18,0.2, 0.27,0.2, 0.34,0, 0.15,0;
            // r_vertices[1]<<0.15,0, 0.34,0, 0.26,-0.2, 0.135,-0.035;
            // r_vertices[2]<<0.135,-0.035, 0.26,-0.2, 0.12,-0.27, 0.08,-0.075;
            // r_vertices[3]<<0.08,-0.075, 0.12,-0.27, 0,-0.27, 0,-0.075;
            // r_vertices[4]<<0,-0.075, 0,-0.27,-0.01,-0.18, -0.01,-0.075;          
            // body_vertices[0].block(0,0,4,2)<<0.235,0.1, 0.3,0, 0.28,-0.08, 0.19,-0.12;        
        }
        else if(hedra_type =="shrink"){
            GREEN("shrink");
            // double shrk=0.02;
            r_vertices[0]<<0.22,0.2, 0.25,0.2, 0.32,0, 0.19,0;
            r_vertices[1]<<0.19,0, 0.32,0, 0.24,-0.19, 0.1623,-0.0645;
            r_vertices[2]<<0.1623,-0.0645, 0.24,-0.19, 0.12,-0.25, 0.093,-0.115;
            r_vertices[3]<<0.093,-0.115, 0.12,-0.25, 0,-0.25, 0,-0.115;
            r_vertices[4]<<0,-0.115, 0,-0.25, -0.15,-0.16, -0.15,-0.115;
            // body_vertices[0].block(0,0,4,2)<<0.23,0.1, 0.29,0, 0.262,-0.0645, 0.193,-0.115;
            body_vertices[0].block(0,0,4,2)<<0.23,0.1, 0.29,0, 0.262,-0.0445, 0.193,-0.1;
            // r_vertices[0]<<0.22,0.2, 0.25,0.2, 0.32,0, 0.19,0;
            // r_vertices[1]<<0.19,0, 0.32,0, 0.24,-0.19, 0.1623,-0.0645;
            // r_vertices[2]<<0.1623,-0.0645, 0.24,-0.19, 0.12,-0.25, 0.093,-0.115;
            // r_vertices[3]<<0.093,-0.115, 0.12,-0.25, 0,-0.25, 0,-0.115;
            // r_vertices[4]<<0,-0.115, 0,-0.25, -0.01,-0.16, -0.01,-0.115;
            // body_vertices[0].block(0,0,4,2)<<0.235,0.1, 0.3,0, 0.28,-0.08, 0.19,-0.12;            
        }
        else{
            RED("------------>ROBOT_POLYHEDRA2 type error, should be 'normal or shrink'<----------------");
            return;
        }

        for(int i=0;i<polys_num;++i){
            r_vertices[i].col(0)=r_vertices[i].col(0).array()+0.1;
            l_vertices[i]=r_vertices[i];
            l_vertices[i].col(0)=-l_vertices[i].col(0).array();
        }

        for(int i=0;i<4;++i){
            body_vertices[0].row(i+4)=body_vertices[0].row(3-i);
        }
        body_vertices[0].bottomRows(4).col(0).array()=-body_vertices[0].bottomRows(4).col(0).array();
        // body_vertices[2]=body_vertices[1];
        // body_vertices[2].col(0)=-body_vertices[2].col(0).array();
        // cout<<"Body Vertices:\n";
        // cout<<body_vertices[0]<<endl<<body_vertices[1]<<endl<<body_vertices[2]<<endl;
        // cout<<body_vertices<<endl;

        //计算凸包H表达
        GREEN("Cal_H_Rep");
        Cal_H_Rep();
    }
    //计算凸包H表达(在身体坐标系下)，矩阵系数和常数项A[x z]>=b
    void Cal_H_Rep(){
        Vector2d point1,point2;
        Vector2d point_mean;
        for(int i=0;i<r_vertices.size();++i){
            //系数矩阵每一行=x1-x2 -(z1-z2)  常数项b=x1z2-x2z1
            //确保每一项都是 a1x+a2z>=b
            point_mean=r_vertices[i].colwise().mean();
            for(int j=0;j<r_vertices[i].rows();++j){
                point1=r_vertices[i].row(j);
                point2= j==r_vertices[i].rows()-1? r_vertices[i].row(0):r_vertices[i].row(j+1);
                r_coeff[i].row(j)<<point2(1)-point1(1),point1(0)-point2(0);
                r_b[i](j)=point1(0)*point2(1)-point2(0)*point1(1);
                if(r_coeff[i].row(j).dot(point_mean)<r_b[i](j)){
                    r_coeff[i].row(j)=-r_coeff[i].row(j);
                    r_b[i](j)=-r_b[i](j);
                }
                else if(r_coeff[i].row(j).dot(point_mean)==r_b[i](j)){
                    RED("r_coeff Mean Points on the line");
                }
            }
            point_mean=l_vertices[i].colwise().mean();
            for(int j=0;j<l_vertices[i].rows();++j){
                point1=l_vertices[i].row(j);
                point2= j==l_vertices[i].rows()-1? l_vertices[i].row(0):l_vertices[i].row(j+1);
                l_coeff[i].row(j)<<point2(1)-point1(1),point1(0)-point2(0);
                l_b[i](j)=point1(0)*point2(1)-point2(0)*point1(1);
                if(l_coeff[i].row(j).dot(point_mean)<l_b[i](j)){
                    l_coeff[i].row(j)=-l_coeff[i].row(j);
                    l_b[i](j)=-l_b[i](j);
                }
                else if(l_coeff[i].row(j).dot(point_mean)==l_b[i](j)){
                    RED("l_coeff Mean Points on the line");
                }
            }
        }
        for(int i=0;i<body_vertices.size();++i){
            point_mean=body_vertices[i].colwise().mean();
            for(int j=0;j<body_vertices[i].rows();++j){
                point1=body_vertices[i].row(j);
                point2= j==body_vertices[i].rows()-1? body_vertices[i].row(0):body_vertices[i].row(j+1);
                body_coeff[i].row(j)<<point2(1)-point1(1),point1(0)-point2(0);
                body_b[i](j)=point1(0)*point2(1)-point2(0)*point1(1);
                if(body_coeff[i].row(j).dot(point_mean)<body_b[i](j)){
                    body_coeff[i].row(j)=-body_coeff[i].row(j);
                    body_b[i](j)=-body_b[i](j);
                }
                else if(body_coeff[i].row(j).dot(point_mean)==body_b[i](j)){
                    RED("body_coeff[0] Mean Points on the line");
                }
            }
        }

        // cout<<"r_coeff:r_b"<<endl;
        // for(int i=0;i<r_coeff.size();++i){
        //     cout<<"i="<<i<<endl;
        //     for(int j=0;j<r_coeff[i].rows();++j){
        //         cout<<r_coeff[i].row(j)<<":"<<r_b[i](j)<<endl;
        //     }
        //     cout<<endl;
        // }

        // cout<<"l_coeff:"<<endl<<l_coeff[0].transpose()<<endl<<l_coeff[1].transpose()<<endl<<l_coeff[2].transpose()<<endl<<l_coeff[3].transpose()<<endl;
        // cout<<"l_b:"<<endl<<l_b[0].transpose()<<endl<<l_b[1].transpose()<<endl<<l_b[2].transpose()<<endl<<l_b[3].transpose()<<endl;        
        // cout<<"body_coeff[0]:"<<endl<<body_coeff[0]<<endl;
        // cout<<"body_b:"<<endl<<body_b<<endl;
    }
    //计算直线与单个凸包相交的上下限
    bool Cross_Limits(const Vector3d &line, const vector<MatrixX2d> &polys_coeff, const vector<VectorXd> &polys_b, const Matrix2d &limits,
                      vector<Vector2d> &cross_points){
        //line=[a b c] ax+bz+c=0 coeff中每一项为构成凸包H表达直线的系数，b为常数项，limits为直线两个端点坐标[x_1,z_1; x_2,z_2]
        //xz_index表示cross_limits指的哪一个维度，0表示x，1表示z，cross_limits表示指定维度的上下界,假设线段穿过凸包是一整段，没有分段现象
        int xz_index;
        bool cross=false;
        double a=line(0),b=line(1),c=line(2);
        double upper_limit,lower_limit;
        double highest_limit,lowest_limit;
        double new_const;
        Vector2d xz_coeff_vec,const_vec;
        VectorXd xz_coeff,left_const;
        Vector2d cross_limits;
        if (abs(a)>1e-3){
            xz_index=1;
            xz_coeff_vec<<-b/a,1;
            const_vec<<-c/a,0;
        }
        else if(abs(b)>1e-3){
            xz_index=0;
            xz_coeff_vec<<1,-a/b;
            const_vec<<0,-c/b;
        }
        else{
            RED("Line coeff all equal to 0; a="<<a<<" b="<<b);
            exit(0);
            // return false;
        }
        // cout<<"xz_index="<<xz_index<<endl;
        highest_limit=limits.col(xz_index).minCoeff();
        lowest_limit=limits.col(xz_index).maxCoeff();

        for(int i=0;i<polys_coeff.size();++i){
            bool break_flag=false;
            lower_limit=limits.col(xz_index).minCoeff();
            upper_limit=limits.col(xz_index).maxCoeff();
            
            xz_coeff.resize(polys_coeff[i].rows());
            left_const.resize(polys_coeff[i].rows());
            xz_coeff=polys_coeff[i]*xz_coeff_vec;
            left_const=polys_coeff[i]*const_vec;
            for(int j=0;j<polys_coeff[i].rows();++j){
                if (xz_coeff(j)!=0){
                    new_const=(polys_b[i](j)-left_const(j))/xz_coeff(j);
                }
                else{
                    new_const=left_const(j);
                }
                if (xz_coeff(j)>0){
                    if (new_const>=lower_limit){
                        lower_limit=new_const;
                    }
                }
                else if(xz_coeff(j)<0){
                    if(new_const<=upper_limit){
                        upper_limit=new_const;
                    }
                }
                else{
                    if(new_const<polys_b[i](j)){
                        break_flag=true;
                        break;
                    }
                }
            }

            // cout<<"i="<<i<<" upper_limit="<<upper_limit<<" lower_limit="<<lower_limit<<" break_flag="<<break_flag<<endl;
            //为了防止浮点数计算误差，不用严格大于
            if(upper_limit-lower_limit>1e-6&&!break_flag){
                cross=true;
                // cout<<"cross i="<<i<<"limit="<<lower_limit<<" "<<upper_limit<<endl;
                highest_limit=max(highest_limit,upper_limit);
                lowest_limit=min(lowest_limit,lower_limit);
            }
        }
        cross_limits<<lowest_limit,highest_limit;
        // cout<<"cross_limits="<<cross_limits.transpose()<<endl;
        //xz_index=0, line(xz_index)=a
        cross_points.resize(2);
        for(int i=0;i<2;++i){
            cross_points[i](xz_index)=cross_limits(i);
            cross_points[i](!xz_index)=-(line(xz_index)*cross_limits(i)+c)/line(!xz_index);
        }
        if(cross_points[0](0)>cross_points[1](0)){
            swap(cross_points[0],cross_points[1]);
        }
        return cross;   
    }
    
    //判断直线是否穿过凸包,并计算直线穿过凸包的区域
    bool Cross(const Vector3d &line, const vector<MatrixX2d> &polys_coeff, const vector<VectorXd> &polys_b, const Matrix2d &limits){

        // cout<<"polys_coeff:\n";
        // for(int i=0;i<polys_coeff.size();++i){
        //     cout<<polys_coeff[i]<<endl<<polys_b[i]<<endl;
        // }
        int xz_index;
        double a=line(0),b=line(1),c=line(2);
        double upper_limit,lower_limit;
        double new_const;
        Vector2d xz_coeff_vec,const_vec;
        VectorXd xz_coeff,left_const;
        if (abs(a)>1e-3){
            xz_index=1;
            xz_coeff_vec<<-b/a,1;
            const_vec<<-c/a,0;
        }
        else if(abs(b)>1e-3){
            xz_index=0;
            xz_coeff_vec<<1,-a/b;
            const_vec<<0,-c/b;
        }
        else{
            RED("Line coeff all equal to 0; a="<<a<<" b="<<b);
            
            return false;
        }
        for(int i=0;i<polys_coeff.size();++i){
            bool break_flag=false;
            lower_limit=limits.col(xz_index).minCoeff();
            upper_limit=limits.col(xz_index).maxCoeff();
            
            xz_coeff.resize(polys_coeff[i].rows());
            left_const.resize(polys_coeff[i].rows());
            xz_coeff=polys_coeff[i]*xz_coeff_vec;
            left_const=polys_coeff[i]*const_vec;
            for(int j=0;j<polys_coeff[i].rows();++j){
                if (xz_coeff(j)!=0){
                    new_const=(polys_b[i](j)-left_const(j))/xz_coeff(j);
                }
                else{
                    new_const=left_const(j);
                }
                if (xz_coeff(j)>0){
                    if (new_const>=lower_limit){
                        lower_limit=new_const;
                    }
                }
                else if(xz_coeff(j)<0){
                    if(new_const<=upper_limit){
                        upper_limit=new_const;
                    }
                }
                else{
                    if(new_const<polys_b[i](j)){
                        break_flag=true;
                        break;
                    }
                }
            }
            // cout<<"i="<<i<<" upper_limit="<<upper_limit<<" lower_limit="<<lower_limit<<" break_flag="<<break_flag<<endl;
            //为了防止浮点数计算误差，不适用严格大于
            if(upper_limit-lower_limit>1e-6&&!break_flag){
                return true;
            }
        }
        return false;
    }

    //判断一点是否在工作空间内，该点需要为身体坐标系下表示
    bool In_Workspace(const Vector2d &point, const vector<MatrixX2d> &polys_coeff, const vector<VectorXd> &polys_b){
        bool in_workspace=false;
        double dot_res;
        for(int i=0;i<polys_coeff.size();++i){
            bool in_poly=true;
            for(int j=0;j<polys_coeff[i].rows();++j){
                dot_res=polys_coeff[i].row(j)*point;
                if(dot_res<polys_b[i](j)){
                    //出现不满足条件，点在这个凸包之外
                    in_poly=false;
                    break;
                }
            }
            if(in_poly){
                in_workspace=true;
                break;
            }
        }
        return in_workspace;
    }
    //判断一点是否在工作空间的凸包内
    bool In_Polyhedron(const Vector2d &point, const vector<MatrixX2d> &polys_coeff, const vector<VectorXd> &polys_b,int poly_index){
        bool in_poly=true;
        double dot_res;
        if(poly_index>0&&poly_index<polys_coeff.size()){
            for(int j=0;j<polys_coeff[poly_index].rows();++j){
                dot_res=polys_coeff[poly_index].row(j)*point;
                if(dot_res<polys_b[poly_index](j)){
                    in_poly=false;
                    break;
                }
            }
        }
        else{
            RED("poly_index out of range, poly_index="<<poly_index<<" polys_coeff.size()="<<polys_coeff.size());
        }
        return in_poly;
    }
    //判断在哪一个凸包内部，返回凸包的序号从上到下到左，与vertex定义顺序一致,0,1,2,3 -1表示不在任何凸包内部,设置approxi=true,当点超出凸包，会计算距离哪个凸包最近，并返回该凸包
    int In_Which_Poly(const Vector2d &point, const vector<MatrixX2d> &polys_coeff, const vector<VectorXd> &polys_b, bool approxi=true ){
        double dot_res;
        double min_dist=1e10;
        int min_index=0;
        for(int i=0;i<polys_coeff.size();++i){
            bool in_poly=true;
            double min_poly_dis=1e10,cur_dis;
            for(int j=0;j<polys_coeff[i].rows();++j){
                dot_res=polys_coeff[i].row(j)*point;
                if(dot_res<polys_b[i](j)){
                    in_poly=false;
                    break;
                }
                cur_dis=abs(dot_res-polys_b[i](j));
                if(cur_dis<min_poly_dis){
                    min_poly_dis=cur_dis;
                }
            }
            if(min_poly_dis<min_dist){
                min_dist=min_poly_dis;
                min_index=i;
            }
            if(in_poly){
                return i;
            }
        }
        if(approxi) {
            return min_index;
            YELLOW("point out of workspace, min_index="<<min_index<<" min_dist="<<min_dist<<endl);
        }
        return -1;
    }

};
#endif
