#!/bin/bash

inv=$(cat $1)
top=$(cat $2.smt.1)
pre=$(cat $2.smt.2)
loop=$(cat $2.smt.3)
post=$(cat $2.smt.4)

echo "checking $2:" 
echo `cat $1`

# for pre
echo "smt check for pre-condition:"
echo -e "$top\n$inv\n$pre\n(check-sat)\n(get-model)" > tmppre
z3 -T:2 -smt2 tmppre

# for loop
echo "smt check for recursive condition"
echo -e "$top\n$inv\n$loop\n(check-sat)\n(get-model)" > tmploop
z3 -T:2 -smt2 tmploop


# for post
echo "smt check for post-condition"
echo -e "$top\n$inv\n$post\n(check-sat)\n(get-model)" > tmppost
z3 -T:2 -smt2 tmppost

#rm tmppost tmppre tmploop
