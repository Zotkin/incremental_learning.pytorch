FIRST_INCREMENT=50;
MEMORIES=(2000 5000 10000 40000);
INCREMENTS=(1 5 10);
for m in ${MEMORIES[@]};
do
for incement in ${INCREMENTS[@]};
  do
  sbatch -o %N_%j.out -e %N_%j.err run_slurm_v2.sh "$FIRST_INCREMENT" "$incement" "$m" ;
  done
done