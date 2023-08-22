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

   or set the number of data  points (-n) 

   ```bash
   ./init -n 100000
   ```

3. Run serial, openmp and cuda code

   ```bash
   ./main
   ```

   Use -w to set window size and use -e to set shift epsilon

   ```C++
   ./main -w 5 -e 0.1
   ```

   Run online updated serial code

   ```C++
   ./main_ou
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

   
