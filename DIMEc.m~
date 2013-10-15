function [Dj_plus, Dj_minus, tau_plus, tau_minus] = DIMEc(pi,mu,Sigma,j,dim, group)
% DIME calculation: 
% compute DIME
% on the cluster/suptype level (mixture of normal mixtures)
% 
% p: dimension , k: number of clusters 
% dim: nnxpp, subset of x-dimension. dim(i,:): ith subset of x. Use 0s to
% keep places
% fj: 1xnn vector
% pi = column k vector probs
% mu = p.k matrix of component means 
% Sigma = p.p.k array of variance matrice 
% group: groupings of components,row represents component grouping
% j: jth cluster

%Dj_plus = E( f_j |x~\ f_j)/E(f_j | x ~ f_j)
%Dj_minus = E( f_j |x~\ f_j)/E(f_-j | x ~\ f_j)
%tau_plus: upper bound of tau
%tau_minus: lower bound of tau

[p,k] = size(mu);
[nn,pp] = size(dim);

Dj_plus = double(zeros(1,nn));
Dj_minus = Dj_plus;
tau_plus = double(zeros(1,nn));
tau_minus = tau_plus;

gpj = group(j,:);
gpj(gpj==0)=[];

indexj = 1:k; indexj(gpj) = [];
cpj = 1-sum(pi(gpj));

D=zeros(k,k,nn); 

for tt = 1:nn
    deno = 0; nume = 0; deno1 = 0;
    dimm = dim(tt,:); dimm(dimm==0)=[];
    for i=1:k
        mi=mu(dimm,i); Si=Sigma(dimm,dimm,i); %mi: mean of ith component, Si: variance-covariance matrix for ith component.
        for jj=i:k,
            D(i,jj,tt)=mvnormpdf(mu(dimm,jj),mi,Si+Sigma(dimm,dimm,jj)); D(jj,i,tt)=D(i,jj,tt); % D(i,j) = f_ij
        end
    end

    for ii = gpj
        for ppp = indexj
            nume= nume + (pi(ii)/(1-cpj))*(pi(ppp)/cpj)*D(ii,ppp,tt);
        end
    end
    %nume = nume*cpj;
  
    for ii = gpj
        for ppp = gpj
            deno = deno + pi(ii)/(1-cpj)*(pi(ppp)/(1-cpj))*D(ii,ppp,tt);
        end
    end
    
    for ii = indexj
        for ppp = indexj
            deno1 = deno1 + (pi(ii)/cpj)*(pi(ppp)/cpj)*D(ii,ppp,tt);
        end
    end

    Dj_plus(tt) = double(nume/deno); % Delta_j in the paper
    Dj_minus(tt) = double(nume/deno1); %Delta_j in the paper
    tau_plus(tt) = double((1-cpj)/((1-cpj)+cpj*Dj_plus(tt)));
    tau_minus(tt) = double((1-cpj)*Dj_minus(tt)/(cpj+(1-cpj)*Dj_minus(tt)));
end
