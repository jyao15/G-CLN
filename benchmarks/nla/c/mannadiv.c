#include <stdio.h>
#include <assert.h>

int mainQ(int A, int B){
     
     assert (A >= 0);
     assert (B >= 1);
     
     int q,r,t;
     q = 0;
     r = 0;
     t = A;

	 // loop 1
     while(1) {
	  //assert(q*B + r + t == A);
	  //assert(r < B && r >= 0);
	  //%%%traces: int q, int r, int t, int A, int B
	  
	  if(!(t != 0)) break;
	  
	  if (r + 1 == B) {
	       q = q + 1;
	       r = 0;
	       t = t - 1;
	  }
	  else {
	       r = r + 1;
	       t = t - 1;
	  }
     }

     //assert(q == A / B);
     return q;
}

int main(int argc, char **argv){
     mainQ(atoi(argv[1]), atoi(argv[2]));
     return 0;
}

