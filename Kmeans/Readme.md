### Environment

puffin

### How to use

1. To compile, run

   ```bash
   make
   ```

2. Initialize the problem size

   default:

   ```bash
   ./init
   ```

   or set the number of data  points (-n) and centroids (-k)

   ```bash
   ./init -n 100000 -k 1000
   ```

3. Run serial, openmp and cuda code

   ```bash
   ./main
   ```

4. Run MPI code

   ```bash
   mpirun ./main_mpi
   ```

5. Check whether the results of the parallel version clustering are consistent with the serial version 

   ```bash
   ./comp
   ```


6. Remove compiled files

   ```bash
   ./make clean
   ```

   or remove compiled files and csv files

   ```BASH
   ./make cleanall
   ```

   
