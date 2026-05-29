#ifndef B_SPLINE_HPP
#define B_SPLINE_HPP
#include<Eigen/Dense>
#include<vector>
#include<iostream>
#include"ColorCout.h"
using namespace Eigen;
using namespace std;

template<typename T>
//template T must be Eigen::Vector (3d Xd etc)
class Bspline{
public:
    int n; //n+1 control points >>> p0,p1,,,pn
    int m; //m+1 knots >>> [u0,u1,,,um]
    int p; //degree of the spline
    vector<double> paras; //normalize data_points to [0,1]
    vector<T> CP; //control points
    vector<double> U; //knots
    VectorXd N; //basis functions
    //for least squares, control points and knots are calculated 
    //n+1 control points and p degrees b-spline to fit data_points
    Bspline(int n, int p, vector<T>& data_points):n(n),p(p){
        normalize_parameters(data_points);
        // GREEN("--------normalize------------");
        get_knots();
        // GREEN("--------get_knots------------");

        if(check()){
            approximation(data_points);
            // GREEN("--------approximate------------");

            GREEN_WHITE("--------->Fitting Bspline wiht "<<p<<" degrees and "<<CP.size()<<" control points<-----------");
        }
        else{
            RED_WHITE("------------->In Bspline.h: Failed to Fit data to B-spline"<<"n="<<n<<" p="<<p<<" m="<<m<<"------------");
        }
    }
    //for normal B-spline, control points and knots are given
    //p degrees and control points CP
    Bspline(int p, vector<T> control_points):p(p){
        SetCP(control_points);
    }
    
    //Minmum intialize, only set p degree
    Bspline(int p):p(p){}

    //set control points CP
    bool SetCP(vector<T> control_points){
        CP=control_points;
        n=CP.size()-1;
        m=n+p+1;
        U.resize(m+1);
        if(check()){
            //inaitialize knots
            // GREEN_WHITE("--------->Creating Bspline wiht "<<p<<" degrees and "<<CP.size()<<" control points<-----------");
            for(int i=0;i<p+1;i++){
                U[i]=0.0;
                U[m-i]=1.0;
            }
            for(int i=1;i<n-p+1;i++){
                U[i+p]=i/double(n-p+1);
            }
            return true;
        }
        else{
            RED_WHITE("In Bspline.h: Failed to initialize B-spline "<<"n="<<n<<" p="<<p<<" m="<<m);
            return false;
        }
    }
    bool check(){
        return (n>=p && n>1 && m==n+p+1);
        // return (n>1 && m==n+p+1);
    }
    void BaseFunction(double& u){
        N=VectorXd::Zero(m+1);
        if(u==U[0]){
            N[0]=1.0;
            return;
        }
        else if(u==U[m]){
            N[n]=1.0;
            return;
        }
        int k=0;
        for(int i=0;i<m;i++){
            if(u>=U[i]&&u<U[i+1]){
                break;
            }
            k+=1;
        }
        N[k]=1.0;
        for(int d=1;d<this->p+1;d++){
            double r_max=this->m-d-1;
            if(k-d>=0){
                if(U[k+1]-U[k-d+1]){
                    N[k-d]=(U[k+1]-u)/(U[k+1]-U[k-d+1])*N[k-d+1];
                }
                else{
                    N[k-d]=(U[k+1]-u)/1.0*N[k-d+1];
                }
            }

            double Denominator1,Denominator2;
            for(int i=k-d+1;i<k;++i){
                if(i>=0&&i<=r_max){
                    Denominator1=U[i+d]-U[i];
                    Denominator2=U[i+d+1]-U[i+1];
                }
                if(Denominator1==0.0){
                    Denominator1=1.0;
                }
                if(Denominator2==0.0){
                    Denominator2=1.0;
                }
                N[i]=(u-U[i])/(Denominator1)*N[i]+(U[i+d+1]-u)/(Denominator2)*N[i+1];
            }

            if(k<=r_max){
                if(U[k+d]-U[k]){
                    N[k]=(u-U[k])/(U[k+d]-U[k])*N[k];
                }
                else{
                    N[k]=(u-U[k])/1.0*N[k];
                }
            }
        }
    }

    T De_Boor(double& u){
        int k=0;
        for(int i=0;i<m;i++){
            if(u>=U[i]&&u<U[i+1]){
                break;
            }
            k+=1;
        }
        
        bool u_in_U=false;
        for(int i=0;i<U.size();i++){
            if(u==U[i]){
                u_in_U=true;
            }
        }
        int sk=0,h;
        if(u_in_U){
            for(int i=0;i<U.size();i++){
                if(U[k]==U[i]) sk++;
            }
            h=p-sk;
        }
        else{
            sk=0;
            h=p;
        }
        if(h==-1){
            if(k==p){
                return CP[0];
            }
            else if(k==m){
                return CP.back();
            }
        }
        vector<T> P;
        P.resize(p-sk+1);

        for(int i=k-p;i<k-sk+1;i++){
            P[i-k+p]=CP[i];
        }
        int dis=k-p;
        for(int r=1;r<h+1;r++){
            vector<T> temp;
            for(int i=k-p+r;i<k-sk+1;i++){
                double a_ir=(u-U[i])/(U[i+p-r+1]-U[i]);
                temp.push_back( (1.0-a_ir)*P[i-dis-1]+a_ir*P[i-dis] );
            }
            for(int i=k-p+r-dis;i<k-sk+1-dis;i++){
                P[i]=temp[i-k+p-r+dis];
            }
        }
        return P.back();
    }

    void bs(vector<double>&us,vector<T>&curve_points){
        curve_points.clear();
        for(auto u:us){
            curve_points.push_back(De_Boor(u));
        }
    }

    void normalize_parameters(vector<T>& data_points){
        vector<double>Li;
        double L=0;
        for(auto data_point:data_points){
            Li.push_back(data_point.norm());
            L+=Li.back();
        }
        paras.clear();
        for(int i=0;i<Li.size();i++){
            double Lki=0;
            for(int j=0;j<i+1;j++){
                Lki+=Li[j];
            }
            paras.push_back(Lki/L);
        }
    }

    void get_knots(){
        U.resize(n+p+2);
        m=n+p+1;
        int num=m-p;
        vector<double> ind_double=linspace(0,paras.size()-1,num);
        vector<double> paras_knots;
        for(auto i:ind_double){
            paras_knots.push_back(paras[floor(i)]);
        }
        for(int j=1;j<n-p+1;j++){
            double k_temp=0;
            for(int i=j;i<j+p;i++){
                k_temp+=paras_knots[i];
            }
            k_temp/=p;
            U[p+j]=k_temp;
        }
        for(int i=U.size()-1;i>U.size()-p-2;i--){
            U[i]=1;
        }

        // int num=m-p;
        // vector<double> ind_double=linspace(0,paras.size()-1,num+2);
        // vector<double> paras_knots;
        // for(auto i:ind_double){
        //     paras_knots.push_back(paras[floor(i)]);
        // }
        // for(int j=0;j<n-p+2;j++){
        //     double k_temp=0;
        //     for(int i=j;i<j+p+1;i++){
        //         k_temp+=paras_knots[i];
        //     }
        //     k_temp/=p+1;
        //     U[p+j]=k_temp;
        // }
        // for(int i=U.size()-1;i>U.size()-p-1;i--){
        //     U[i]=1;
        // }        
    }

    void approximation(vector<T>&data_points){
        int num=data_points.size()-1;
        CP.resize(n+1);
        CP[0]=data_points[0];
        CP.back()=data_points.back();
        MatrixXd N_mat;
        N_mat.resize(paras.size(),n+1);
        for( int i=0;i<paras.size();++i){
            this->BaseFunction(paras[i]);
            N_mat.row(i)=(N.block(0,0,n+1,1)).transpose();
        }
        MatrixXd Q;
        VectorXd Q_k;
        Q.resize(num-1,data_points[0].rows());
        Q_k.resize(data_points[0].rows());
        for(int k=0;k<num-1;k++){
            Q_k=data_points[k+1]-N_mat(k+1,0)*data_points[0]-N_mat(k+1,n)*data_points.back();
            Q.row(k)=Q_k;
        }
        MatrixXd B;
        
        B.resize(n-1,data_points[0].rows());
        for(int i=1;i<n;++i){
            VectorXd b_temp=VectorXd::Zero(data_points[0].rows());
            for(int k=0;k<num-1;k++){
                b_temp=b_temp+ (N_mat(k+1,i)*Q.row(k)).transpose();
            }
            B.row(i-1)=b_temp.transpose();
        }
        MatrixXd N_mat_block;
        N_mat_block=N_mat.block(1,1,num-1,n-1);
        // Map<Eigen::MatrixXd> N_mat_block(N_mat.data()+1*N_mat.cols()+1,num-1,n-1); //(data_points' number-2 X control points' number -2 )
        MatrixXd A=N_mat_block.transpose()*N_mat_block;
        LLT<MatrixXd> llt(A);
        Eigen::MatrixXd CP_block=llt.solve(B);
        for(int i=0;i<CP_block.rows();i++){
            CP[i+1]=CP_block.row(i);
        }
        
    }
    
    vector<double> linspace(double start,double end,int num){
        vector<double> res;
        double step=(end-start)/double(num-1);
        for(int i=0;i<num;i++){
            res.push_back(i*step+start);
        }
        return res;
    }
};

#endif