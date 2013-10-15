DIME
====

MATLAB code for DIME calculation

1. compute DIME on the cluster/suptype level (mixture of normal mixtures)

2. p: dimension , k: number of clusters 

3. dim: nnxpp, subset of x-dimension. dim(i,:): ith subset of x. Use 0s to keep places

4. fj: 1xnn vector

5. pi = column k vector probs

6. mu = p.k matrix of component means 

7. Sigma = p.p.k array of variance matrice

8. group: groupings of components,row represents component grouping

9. j: jth cluster

10. Output: DIME values Delta_j (i.e. Dj_plus), Delta_-j (i.e. Di_minus), tau_plus and tau_minus (which are the upper and lower bounds for tau)

11. The paper associated with this code is located [here](http://ftp.stat.duke.edu/WorkingPapers/10-23.pdf) 
