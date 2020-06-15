#include <stdio.h>
#include <assert.h>

int mainQ(int a, int b){
     assert(a>=1);
     assert(b>=1);
     int x,y,u,v;

     x=a;
     y=b;
     u=b;
     v=0;

	 // loop 1
     while(1) {
	  //assert(x*u + y*v == a*b);
	  //assert(GCD(x,y) == GCD(a,b));
	  //%%%traces: int a, int b, int x, int y, int u, int v
	  if (!(x!=y)) break;
	  
	  // loop 2
	  while (1){
		   //assert(x*u + y*v == a*b);
	       //assert(GCD(x,y) == GCD(a,b));
	       //%%%traces: int a, int b, int x, int y, int u, int v
	       if(!(x>y)) break;
	       x=x-y;
	       v=v+u;
	  }
    
	  // loop 3
	  while (1){
		   //assert(x*u + y*v == a*b);
	       //assert(GCD(x,y) == GCD(a,b));
	       //%%%traces: int a, int b, int x, int y, int u, int v
	       if(!(x<y)) break;
	       y=y-x;
	       u=u+v;
	  }

     }

     //x==gcd(a,b)
     int r = u+v; 
     return r; //lcm     
}


int main(int argc, char **argv){
     mainQ(atoi(argv[1]), atoi(argv[2]));
     return 0;
}

