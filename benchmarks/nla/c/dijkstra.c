#include <stdio.h>
#include <assert.h>
#include <stdlib.h>  //required for afloat to work


int mainQ(int n){
  assert (n >= 0);
  int p,q,r,h;
  p = 0;
  q = 1;
  r = n;
  h = 0;

  // loop 1
  while (1){
    //assert(p == 0 && r == n && h == 0 && n >= 0);
    //%%%traces: int r, int p, int n, int q, int h
    if(!(q<=n)) break;
    q=4*q;
  }

  // loop 2
  while (1){
    //assert(r < 2*p + q);
    //assert(p*p + r*q == n*q);
    //assert(r >= 0);
    //%%%traces: int r, int p, int n, int q, int h

    if(!(q!=1)) break;

    q=q/4;
    h=p+q;
    p=p/2;
    if (r>=h){
      p=p+q;
      r=r-h;
    }
  }
  return p;
}


int main(int argc, char **argv){
  mainQ(atoi(argv[1]));
  return 0;
}
