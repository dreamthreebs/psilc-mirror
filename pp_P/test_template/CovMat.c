#include<stdio.h>
#include<math.h>
#define p 1e-5
#define pi acos(-1.)

// Calculate the Associated Legendre Polynomial P_l^0
//double Plz(double l, double z, double *PLZ)
//{
//	//if(PLZ[(int)l]!=1000.) return PLZ[(int) l];
//	double plz;
//	if(l==0.) plz = 1.;
//	if(l==1.) plz = z;
//	if(l>1) plz = ((2*l-1)*z*Plz(l-1, z, PLZ)-(l-1)*Plz(l-2, z, PLZ))/l;
//	PLZ[(int)(l)] = plz;
//	printf("l=%f,plz=%f\n", l, plz);
//	return plz;
//}

double Plz(double l, double z, double *PLZ){
	int i;
	PLZ[0] = 1.;
	PLZ[1] = z;
	for(i=2.;i<=l;i++){
		PLZ[i] = ((2*i-1)*z*PLZ[i-1]-(i-1)*PLZ[i-2])/i;
	}
}

// Calculate the Associated Legendre Polynomial P_l^2
//double P2lz(double l, double z, double *PLZ)
//{
//	double plz;
//	if(l<2) plz = 0.;
//	if(l==2.) plz = 3*(1-pow(z, 2));
//	if(l==3.) plz = 5*z*P2lz(2, z, PLZ);
//	if(l>=4.) plz = ((2*l-1)*z*P2lz(l-1, z, PLZ)-(l+1)*P2lz(l-2, z, PLZ))/(l-2);
//	PLZ[(int)(l)] = plz;
//	return plz;
//}
double P2lz(double l, double z, double *PLZ){
	int i;
	PLZ[0] = 0.;
	PLZ[1] = 0.;
	PLZ[2] = 3*(1-pow(z, 2));
	PLZ[3] = 5*z*PLZ[2];
	for(i=4.;i<=l;i++){
		PLZ[i] = ((2*i-1)*z*PLZ[i-1]-(i+1)*PLZ[i-2])/(i-2);
	}
}

// Calculate the corss product of two vectors; input ri, rj; output r;
void CrossProd(double *ri, double *rj, double *r)
{
	*(r+0) = *(ri+1)*(*(rj+2))-*(ri+2)*(*(rj+1));
	*(r+1) = *(ri+2)*(*(rj+0))-*(ri+0)*(*(rj+2));
	*(r+2) = *(ri+0)*(*(rj+1))-*(ri+1)*(*(rj+0));
	Norm(r);
}

// Calculate the length of a vector.
double Dist(double *r)
{
	int i;
	double dist=0.;
	for(i=0;i<3;i++)
	{
		//printf("riii=%f\n", *(r+i));
		dist = dist+pow(*(r+i), 2);
		//dist = dist+*(r+i)*(*(r+i));
	}
	//printf("dist=%f\n", dist);
	return sqrt(dist);
}

// Calculate the dot product of two vectors
double DotProd(double *ri, double *rj)
{
	int i;
	double prod=0.;
	for(i=0;i<3;i++)
	{
		prod = prod+*(ri+i)*(*(rj+i));
	}
	if(prod>1.) prod=1.;
	if(prod<-1.) prod=-1.;
	return prod;
}

// Normalization vector
void Norm(double *r)
{
	int i;
	double dist;
	dist = Dist(r);
	if(dist!=0.){
		for(i=0;i<3;i++)
		{
			*(r+i) = *(r+i)/dist;
		}
	}
}


// Add a small displacement to a vector
void modify(double *r)
{
	int i;
	for(i=1;i<3;i++)
	{
		*(r+1)+=p;
	}
}

// Print Matrix
void printM(double M[][3], char *s){
	int i, j;
	printf("%s\n", s);
	for(i=0;i<3;i++){
		for(j=0;j<3;j++){
			printf("%f, ", M[i][j]);
			}
		printf("\n");
		}
}

// Print vector
void printv(double *vec, char *s){
	int i;
	printf("%s\n", s);
	for(i=0;i<3;i++){
		printf("%f, ", vec[i]);
	}
	printf("\n");
}

// Calculate the rotation angle alpha using the vector method; 
// The output is alphaij and alphaji
void Calc_alpha(double *ri, double *rj, double *rij, double *rji, double *rsi, double *rsj, double *alpha, const double *zvec)
{
	CrossProd(ri, rj, rij);
	CrossProd(rj, ri, rji);
	CrossProd(zvec, ri, rsi);
	CrossProd(zvec, rj, rsj);
	//printf("Distrij%f\n", Dist(rij));
	//getchar();
	//printv(ri, "ri");
	//printv(rj, "rj");
	//printv(zvec, "zvec");
	//printv(rij, "rij");
	//printv(rji, "rji");
	//printv(rsi, "rsi");
	//printv(rsj, "rsj");
	if(Dist(rij)<p)
	{
		//printf("HH\n");
		alpha[0] = 0.;
		alpha[1] = 0.;
	}
	else
	{
		//if(Dist(rsi)<p) modify(ri);
		//if(Dist(rsj)<p) modify(rj);
		if(DotProd(rij, zvec)>0.) alpha[0] = acos(DotProd(rij, rsi));
		else alpha[0] = -acos(DotProd(rij, rsi));
		if(DotProd(rji, zvec)>0.) alpha[1] = acos(DotProd(rji, rsj));
		else alpha[1] = -acos(DotProd(rji, rsj));
	}
}

// Construct the rotation matrix with rotation angle alpha
void RotMat(double alpha, double R[][3])
{
	R[0][0] = 1.;
	// because the R matrix initialized as {0.}
	//R[0][1] = 0.;
	//R[0][2] = 0.;
	//R[1][0] = 0.;
	//R[2][0] = 0.;
	R[1][1] = cos(2*alpha);
	R[2][2] = cos(2*alpha);
	R[1][2] = -sin(2*alpha);
	R[2][1] = sin(2*alpha);
}

// Transpose the matrix
void MatT(double M[][3])
{
	double tmp;
	int i,j;
	for(i=0;i<3;i++)
	{
		for(j=i+1;j<3;j++)
		{
			tmp = M[i][j];
			M[i][j] = M[j][i];
			M[j][i] = tmp;
			//tmp = *(M+3*i+j);
			//*(M+3*i+j) = *(M+3*j+i);
			//*(M+3*j+i) = tmp;
		}
	}
}

// Matrix multiplication
void Mat_Mul(double *A, double *B, double *C)
{
	int i,j,k;
	for(i=0;i<3;i++)
	{
		for(j=0;j<3;j++)
		{
			*(C+3*i+j) = 0.;
			//C[i][j] = 0.;
			for(k=0;k<3;k++)
			{
				*(C+3*i+j)+=*(A+3*i+k)**(B+3*k+j);
			}
		}
	}
}

// Calculate the covariance matrix before rotation
void Calc_M(double *l, double *Cls, double M[][3], double *PLZ, double *P2LZ, double z, int lmax, int nl)
{
	int i;
	double li;
	double F10, F12, F22;
	double TiTj=0., TiQj=0., TiUj=0., QiQj=0., UiUj=0.;
	Plz(lmax, z, PLZ);
	P2lz(lmax, z, P2LZ);
//	for(i=0;i<=lmax;i++){
//		printf("i=%d\n", i);
//		printf("PLZi=%f\n", P2LZ[i]);
//	}
	//getchar();
	for(i=0;i<nl;i++)
	{
		li = l[i];
		TiTj+=(2*li+1)/(4*pi)*PLZ[(int)(li)]**(Cls+nl*0+i);
		
		if(fabs((fabs(z)-1.))>1e-8)
		{
			//F10 = 2.*((li*z)/(1-pow(z,2))*Plz(li-1,z)-(li/(1-pow(z,2))+(li*(li-1))/2.)*Plz(li,z))/(sqrt((li-1)*li*(li+1)*(li+2)));
			//F12 = 2.*((li+2)*z/(1-pow(z,2))*P2lz(li-1,z)-((li-4)/(1-pow(z,2))+(li*(li-1))/2.)*P2lz(li,z))/((li-1)*li*(li+1)*(li+2));
			//F22 = 4.*((li+2)*P2lz(li-1,z)-((li-1)*z*P2lz(li,z)))/((li-1)*li*(li+1)*(li+2)*(1-pow(z,2)));
			F10 = 2.*((li*z)/(1-pow(z,2))*PLZ[(int)(li)-1]-(li/(1-pow(z,2))+(li*(li-1))/2.)*PLZ[(int)(li)])/(sqrt((li-1)*li*(li+1)*(li+2)));
			F12 = 2.*((li+2)*z/(1-pow(z,2))*P2LZ[(int)(li)-1]-((li-4)/(1-pow(z,2))+(li*(li-1))/2.)*P2LZ[(int)(li)])/((li-1)*li*(li+1)*(li+2));
			F22 = 4.*((li+2)*P2LZ[(int)(li)-1]-((li-1)*z*P2LZ[(int)(li)]))/((li-1)*li*(li+1)*(li+2)*(1-pow(z,2)));
		}
		else
		{
			F10 = 0.;
			if(z>0.)
			{
				F12 = 1./2;
				F22 = -1./2;
			}
			else
			{
				F12 = 1./2*pow(-1,(int) li);
				F22 = 1./2*pow(-1,(int) li);
			}
		}
		TiQj+=-(2*li+1)/(4*pi)*F10*(*(Cls+nl*3+i));
		TiUj+=-(2*li+1)/(4*pi)*F10*(*(Cls+nl*4+i));
		QiQj+=(2*li+1)/(4*pi)*(F12*(*(Cls+nl*1+i))-F22*(*(Cls+nl*2+i)));
		UiUj+=(2*li+1)/(4*pi)*(F12*(*(Cls+nl*2+i))-F22*(*(Cls+nl*1+i)));
	}
	// Construct the matrix M
	M[0][0] = TiTj;
	M[0][1] = TiQj;
	M[0][2] = TiUj;
	M[1][0] = TiQj;
	M[1][1] = QiQj;
	// because ClEB always be 0, so QiUj always be 0.
	M[1][2] = 0.;
	M[2][0] = TiUj;
	M[2][1] = 0.;
	M[2][2] = UiUj;
//	if(z==1.){
//		printf("TiUj%f\n", TiQj);
//	}
}

// Calculate the theoretical covariance matrix
void CovMat(double *vecs, double *l, double *Cls, double *covmat, int npix, int nl)
{
	#pragma omp parallel
	{
	int kr,kl,t;
	double z;
	double ri[3], rj[3], rij[3], rji[3], rsi[3], rsj[3];
	double M[3][3], M1[3][3], covmatij[3][3];
	double Rij[3][3]={0.}, Rji[3][3]={0.};
	double alpha[2];
	int lmax=(int) (l[nl-1]);
	double* PLZ=(double*)malloc((lmax+1)*sizeof(double));
	double* P2LZ=(double*)malloc((lmax+1)*sizeof(double));
	for(int i=0;i<=lmax;i++){
		PLZ[i] = P2LZ[i] = 0.;
	}
	static const double zvec[3]={0.,0.,1.};
	#pragma omp for schedule(dynamic, 1)
	//npix = 3;
	for(int i=0;i<npix;i++)
	{
		printf("i=%d\n", i);
		for(int j=0;j<npix;j++)
		{
//			printf("j=%d\n", j);
			for(t=0;t<3;t++){
				ri[t] = *(vecs+t*npix+i);
				rj[t] = *(vecs+t*npix+j);
//				if(i==j){
//				    printf("rit=%f\n", ri[t]);
//				    printf("rjt=%f\n", rj[t]);}
			}
			// normalize the vector
			Norm(ri);
			Norm(rj);
			z = DotProd(ri, rj);
//			if(i==j) printf("z=%.9f\n", z);
			Calc_M(l, Cls, M, PLZ, P2LZ, z, lmax, nl);

			Calc_alpha(ri, rj, rij, rji, rsi, rsj, alpha, zvec);

//			printf("alphaij=%f\n", alpha[0]);
//			printf("alphaji=%f\n", alpha[1]);
//			getchar();
//			if(i==0 && j==1){
//				printv(ri, "ri");
//				printv(rj, "rj");
//				printf("alphaij=%f\n", alpha[0]);
//				printf("alphaji=%f\n", alpha[1]);
//				getchar();
//			}

			RotMat(alpha[0], Rij);
			RotMat(alpha[1], Rji);

			//RotMat(0., Rij);
			//RotMat(0., Rji);
      //

			MatT(Rji);
			Mat_Mul(M, Rji, M1);
			Mat_Mul(Rij, M1, covmatij);

//			if(i==j){
//				printM(M, "M");
//				getchar();
////				printM(Rji, "Rji");
////				printM(Rij, "Rij");
////				MatT(covmatij);
////				printM(covmatij, "covmat");
//			}
			//if(i==5 && j==1){
			//	printM(M, "M51");
			//	printM(Rji, "R15T");
			//	printM(Rij, "R51");
			//	printM(covmatij, "covmat51");
			//}
			// fill into the covmat
			for(kr=0;kr<3;kr++)
			{
				for(kl=0;kl<3;kl++)
				{
					//*(covmat+(3*i+kr)*3*npix+(3*j+kl)) = *(covmatij+3*kr+kl);
					*(covmat+(3*i+kr)*3*npix+(3*j+kl)) = covmatij[kr][kl];
					/* *(covmat+(3*i+kr)*3*npix+(3*j+kl)) = M[kr][kl]; */
				}
			}
		}
	}
	}
	//printf("hehe\n");
}
