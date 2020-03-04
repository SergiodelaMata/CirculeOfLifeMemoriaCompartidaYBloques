#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <ctime>
#include <time.h>
#include "../common/book.h"

//Elabora un número aleatorio

__global__ void make_rand(int seed, char* m, int size) {
    float myrandf;
    int num;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, idx, 0, &state); //Se prepara la ejecución del random de CUDA
    myrandf = curand_uniform(&state);
    myrandf *= (size - 0 + 0.999999);
    num = myrandf;
    if (m[num] == 'O')
    {
        m[num] = 'X';
    }
}
//Se da el valor inicial de las distintas casillas de la matriz
__global__ void prepare_matrix(char* p, int number_columns, int number_rows, int width_block)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Columna de un hilo entre todos los bloques
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // Fila de un hilo entre todos los bloques
    int index = idx + idy * number_columns;
    int valAux = 0;
    if (idx < number_columns && idy < number_rows)
    {
        p[index] = 'O';
    }
}

//Se genera una matriz de manera que los elementos bajan una fila
__global__ void matrix_operation(char* m, char* p, int width, int size, int number_columns, int number_rows, int width_block) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Columna de un hilo entre todos los bloques
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // Fila de un hilo entre todos los bloques
    int x = threadIdx.x; // Columna de un hilo en un bloque
    int y = threadIdx.y; // Fila de un hilo en un bloque
    int index = idx + idy * number_columns;
    int widthSharedMatrix = 18;
    int indexBlock = (x + 1) + (y + 1) * widthSharedMatrix; // Id en la matriz de compartida
    int counter = 0;
    __shared__ int sharedMatrix[18 * 18];
    if (idx < number_columns && idy < number_rows)
    {
        sharedMatrix[indexBlock] = m[index];
        //printf("COCO %d %d\n", indexBlock, index);
        //printf("COCO %c %c\n", sharedMatrix[indexBlock], m[index]);
        //printf("%c\n", sharedMatrix[indexBlock]);
        if ((x == 0) && (idx != 0)) // Elemento del lateral izquierdo del bloque tiene un elemento a su izquierda
        {
            sharedMatrix[indexBlock - 1] = m[index - 1];
        }
        else
        {
            sharedMatrix[indexBlock - 1] = 'O';
        }
        if ((x == widthSharedMatrix - 1) && (idx != number_columns - 1))// Elemento del lateral izquierdo del bloque tiene un elemento a su derecha
        {
            sharedMatrix[indexBlock + 1] = m[index + 1];
        }
        else
        {
            sharedMatrix[indexBlock + 1] = 'O';
        }
        if ((y == 0) && (idy != 0)) // Elemento del lateral superior del bloque tiene un elemento por encima
        {
            sharedMatrix[indexBlock - widthSharedMatrix] = m[index - number_columns];
        }
        else
        {
            sharedMatrix[indexBlock - widthSharedMatrix] = 'O';
        }
        if ((y == widthSharedMatrix - 1) && (idy != number_rows)) // Elemento del lateral inferior del bloque tiene un elemento por debajo
        {
            sharedMatrix[indexBlock + widthSharedMatrix] = m[index + number_columns];
        }
        else
        {
            sharedMatrix[indexBlock + widthSharedMatrix] = 'O';
        }

        if (indexBlock == 19) // Elemento de la esquina superior izquierda del bloque
        {
            if (index - number_columns - 1 >= 0)
            {
                sharedMatrix[0] = m[index - number_columns - 1];
            }
            else
            {
                sharedMatrix[0] = 'O';
            }
        }
        else if (indexBlock == 34) // Elemento de la esquina superior derecha del bloque
        {
            if (index - number_columns + 1 >= 0)
            {
                sharedMatrix[17] = m[index - number_columns + 1];
            }
            else
            {
                sharedMatrix[17] = 'O';
            }
        }
        else if (indexBlock == 289) // Elemento de la esquina inferior izquierda del bloque
        {
            if (index + number_columns - 1 < number_columns * number_rows)
            {
                sharedMatrix[306] = m[index + number_columns - 1];
            }
            else
            {
                sharedMatrix[306] = 'O';
            }
        }
        else if (indexBlock == 304) // Elemento de la esquina inferior izquierda del bloque
        {
            if (index + number_columns + 1 < number_columns * number_rows)
            {
                sharedMatrix[323] = m[index + number_columns + 1];
            }
            else
            {
                sharedMatrix[323] = 'O';
            }
        }
    }
    __syncthreads();
    /*if (idx < number_columns && idy < number_rows)
    {
        if (indexBlock == 19)
        {
            printf("\n");
            for (int i = 0; i < widthSharedMatrix*widthSharedMatrix; i++)//Representación matriz inicial
            {
                if (i % widthSharedMatrix == widthSharedMatrix- 1)
                {
                    printf("%c\n", sharedMatrix[i]);
                }
                else
                {
                    printf("%c ", sharedMatrix[i]);
                }
            }
        }
    }
    __syncthreads();*/
    if (idx < number_columns && idy < number_rows)
    {
        if (sharedMatrix[indexBlock - widthSharedMatrix - 1] == 'X') // Estudia si existe esquina superior izquierda y si tiene una célula viva
        {
            counter++;
        }
        if (sharedMatrix[indexBlock - 1] == 'X') //Estudia si existe el casilla en el lateral izquierdo y si tiene una célula viva
        {
            counter++;
        }
        if (sharedMatrix[indexBlock - widthSharedMatrix] == 'X') //Estudia si existe el casilla en el lateral superior y si tiene una célula viva
        {
            counter++;
        }
        if (sharedMatrix[indexBlock - widthSharedMatrix + 1] == 'X') // Estudia si existe esquina superior derecha y si tiene una célula viva
        {
            counter++;
        }
        if (sharedMatrix[indexBlock + 1] == 'X') //Estudia si existe el casilla en el lateral derecho y si tiene una célula viva
        {
            counter++;
        }
        if (sharedMatrix[indexBlock + widthSharedMatrix - 1] == 'X') // Estudia si existe esquina inferior izquierda y si tiene una célula viva
        {
            counter++;
        }
        if (sharedMatrix[indexBlock + widthSharedMatrix] == 'X') //Estudia si existe el casilla en el lateral inferior y si tiene una célula viva
        {
            counter++;
        }
        if (sharedMatrix[indexBlock + widthSharedMatrix + 1] == 'X') // Estudia si existe esquina inferior derecha y si tiene una célula viva
        {
            counter++;
        }
        if ((counter == 3) && (sharedMatrix[indexBlock] == 'O')) // Una célula muerte se convierte en viva si tiene 3 células vivas alrededor de ella
        {
            p[index] = 'X';
        }
        else if (((counter < 2) || (counter > 3)) && (sharedMatrix[indexBlock] == 'X')) // Una célula viva se convierte en muerte si alrededor de ella hay un número de células distinto de 2 o 3
        {
            p[index] = 'O';
        }
        else //La célula mantiene su estado
        {
            p[index] = sharedMatrix[indexBlock];
        }
    }
}

void operation(int size, int width, dim3 nBlocks, dim3 nThreads, int number_columns, int number_rows, char execution_mode, int width_block);
void generate_matrix(char* m, int size, dim3 nBlocks, dim3 nThreads, int number_columns, int number_rows, int width_block);
int generate_random(int min, int max);
void step_life(char* m, char* p, int width, int size, dim3 nBlocks, dim3 nThreads, int number_columns, int number_rows, int width_block);
void show_info_gpu_card();
int main(int argc, char* argv[])
{
    show_info_gpu_card(); //Muestra las características de la tarjeta gráfica
    printf("Comienza el juego de la vida:\n");
    int number_blocks = 1;
    //int number_threads = 1;
    int number_rows = 32;
    int number_columns = 32;
    int width_block = 16;
    char execution_mode = 'a';
    // Condiciones para los casos en los que se está pasando por terminal una serie de parámetros
    if (argc == 2)
    {
        execution_mode = argv[1][0];
    }
    else if (argc == 3)
    {
        execution_mode = argv[1][0];
        number_rows = atoi(argv[2]);
    }
    else if (argc >= 4)
    {
        execution_mode = argv[1][0];
        number_rows = atoi(argv[2]);
        number_columns = atoi(argv[3]);
    }
    int size = number_rows * number_columns;
    int width = number_columns;
    dim3 threads(width_block, width_block);
    if (size <= 32 * 32)
        // Si el tamaño de la matriz es inferior o igual a 1024 
    {
        //number_threads = width_block * width_block;
        if ((number_rows % width_block == 0) && (number_columns % width_block == 0))// Número de elementos múltiplos de 16 tanto en fila como en columna
        {
            number_blocks = (number_rows / width_block) * (number_columns / width_block);
        }
        else if (number_rows % width_block == 0)// Número de elementos múltiplos de 16 en fila 
        {
            number_blocks = (number_rows / width_block) * ((number_columns / width_block) + 1);
        }
        else if (number_columns % width_block == 0)// Número de elementos múltiplos de 16 en columna
        {
            number_blocks = ((number_rows / width_block) + 1) * (number_columns / width_block);
        }
        else
        {
            number_blocks = ((number_rows / width_block) + 1) * ((number_columns / width_block) + 1);
        }
        dim3 nBlocks(1, number_blocks);
        operation(size, width, nBlocks, threads, number_columns, number_rows, execution_mode, width_block);
    }
    else
    {
        printf("No son válidas las dimensiones introducidas para la matriz.\n");
    }


    getchar();
    getchar();
    return 0;
}

void operation(int size, int width, dim3 nBlocks, dim3 nThreads, int number_columns, int number_rows, char execution_mode, int width_block)
//Realiza todas las operaciones del juego de la vida
{
    int counter = 1;
    char* a = (char*)malloc(size * sizeof(char));
    char* b = (char*)malloc(size * sizeof(char));
    generate_matrix(a, size, nBlocks, nThreads, number_columns, number_rows, width_block);
    printf("Situacion Inicial:\n");
    for (int i = 0; i < size; i++)//Representación matriz inicial
    {
        if (i % width == width - 1)
        {
            printf("%c\n", a[i]);
        }
        else
        {
            printf("%c ", a[i]);
        }
    }
    while (true)
    {
        if (counter % 2 == 1)
        {
            step_life(a, b, width, size, nBlocks, nThreads, number_columns, number_rows, width_block);
            printf("Matriz paso %d:\n", counter);
            for (int i = 0; i < size; i++)//Representación matriz inicial
            {
                if (i % width == width - 1)
                {
                    printf("%c\n", b[i]);
                }
                else
                {
                    printf("%c ", b[i]);
                }
            }
        }
        else
        {
            step_life(b, a, width, size, nBlocks, nThreads, number_columns, number_rows, width_block);
            printf("Matriz paso %d:\n", counter);
            for (int i = 0; i < size; i++)//Representación matriz inicial
            {
                if (i % width == width - 1)
                {
                    printf("%c\n", a[i]);
                }
                else
                {
                    printf("%c ", a[i]);
                }
            }
        }
        counter++;
        if (execution_mode == 'm') //Si el modo seleccionado no es automático para hasta que el usuario pulse una tecla
        {
            getchar();
        }
    }

    free(a);
    free(b);
}


void generate_matrix(char* m, int size, dim3 nBlocks, dim3 nThreads, int number_columns, int number_rows, int width_block)
// Genera la matriz con su estado inicial
{
    srand(time(NULL));
    int seed = rand() % 50000;
    char* m_d;
    int numElem = generate_random(1, size * 0.25);// Genera un número aleatorio de máxima número de células vivas en la etapa inicial siendo el máximo un 15% del máximo número de casillas
    cudaMalloc((void**)&m_d, size * sizeof(char));
    cudaMemcpy(m_d, m, size * sizeof(char), cudaMemcpyHostToDevice);
    prepare_matrix << <nBlocks, nThreads >> > (m_d, number_columns, number_rows, width_block); //Prepara la matriz con todas las casillas con células muertas
    make_rand << <1, numElem >> > (seed, m_d, size); // Va colocando de forma aleatoria células vivas en las casillas de la matriz
    cudaDeviceSynchronize();
    cudaMemcpy(m, m_d, size * sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(m_d);
}

void step_life(char* m, char* p, int width, int size, dim3 nBlocks, dim3 nThreads, int number_columns, int number_rows, int width_block)
// Genera la matriz resultado a partir de una matriz inicial con las restricciones marcadas para cada casilla
{
    char* m_d;
    char* p_d;
    cudaMalloc((void**)&m_d, size * sizeof(char));
    cudaMalloc((void**)&p_d, size * sizeof(char));
    cudaMemcpy(m_d, m, size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(p_d, p, size * sizeof(char), cudaMemcpyHostToDevice);
    matrix_operation << <nBlocks, nThreads >> > (m_d, p_d, width, size, number_columns, number_rows, width_block);// Estudia el cambio o no de valor de las distintas casillas de la matriz
    cudaDeviceSynchronize();
    cudaMemcpy(m, m_d, size * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(p, p_d, size * sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(m_d);
    cudaFree(p_d);
}

int generate_random(int min, int max) // Genera un número aleatorio entre un mínimo y un máximo
{
    srand(time(NULL));
    int randNumber = rand() % (max - min) + min;
    return randNumber;
}

void show_info_gpu_card() // Muestra las características de la tarjeta gráfica usada
{
    cudaDeviceProp prop;

    int count;
    //Obtención número de dispositivos compatibles con CUDA
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    printf("Numero de dispositivos compatibles con CUDA: %d.\n", count);

    //Obtención de características relativas a cada dispositivo
    for (int i = 0; i < count; i++)
    {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        printf("Informacion general del dispositivo %d compatible con CUDA:\n", i + 1);
        printf("Nombre GPU: %s.\n", prop.name);
        printf("Capacidad de computo: %d,%d.\n", prop.major, prop.minor);
        printf("Velocidad de reloj: %d kHz.\n", prop.clockRate);
        printf("Copia solapada dispositivo: ");
        if (prop.deviceOverlap)
        {
            printf("Activo.\n");
        }
        else
        {
            printf("Inactivo.\n");
        }
        printf("Timeout de ejecucion del Kernel: ");
        if (prop.kernelExecTimeoutEnabled)
        {
            printf("Activo.\n");
        }
        else
        {
            printf("Inactivo.\n");
        }

        printf("\nInformacion de memoria para el dispositivo %d:\n", i + 1);
        printf("Memoria global total: %zu GB.\n", prop.totalGlobalMem / (1024 * 1024 * 1024));
        printf("Memoria constante total: %zu Bytes.\n", prop.totalConstMem);
        printf("Memoria compartida por bloque: %zu Bytes.\n", prop.sharedMemPerBlock);
        printf("Ancho del bus de memoria global: &d.\n", prop.memoryBusWidth);
        printf("Numero registros compartidos por bloque: %d.\n", prop.regsPerBlock);
        printf("Numero hilos maximos por bloque: %d.\n", prop.maxThreadsPerBlock);
        printf("Memoria compartida por multiprocesador: %zu Bytes.\n", prop.sharedMemPerMultiprocessor);
        printf("Numero registros compartidos por multiprocesador: %d.\n", prop.regsPerMultiprocessor);
        printf("Numero hilos maximos por multiprocesador: %d.\n", prop.maxThreadsPerMultiProcessor);
        printf("Numero de hilos en warp: %d.\n", prop.warpSize);
        printf("Alineamiento maximo de memoria: %zu.\n", prop.memPitch);
        printf("Textura de alineamiento: %zd.\n", prop.textureAlignment);
        printf("Total de multiprocesadores: %d.\n", prop.multiProcessorCount);
        printf("Maximas dimensiones de un hilo: (%d, %d, %d).\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Maximas dimensiones de grid: (%d, %d, %d).\n\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    }
    getchar();
}