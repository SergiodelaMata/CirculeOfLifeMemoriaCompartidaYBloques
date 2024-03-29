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

//Elabora un n�mero aleatorio

__global__ void make_rand(int seed, char* m, int size) {
    float myrandf;
    int num;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, idx, 0, &state); //Se prepara la ejecuci�n del random de CUDA
    myrandf = curand_uniform(&state);
    myrandf *= (size - 0 + 0.999999);
    num = myrandf;
    if (m[num] == 'O') // Convierte una c�lula muerta a viva
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
    int index = idx + idy * number_columns; //�ndice de la matriz seg�n el id de la columna y el id de la fila de un hilo
    int widthSharedMatrix = 18; // Ancho de la matriz en memoria compartida
    int indexBlock = (x + 1) + (y + 1) * widthSharedMatrix; // Id en la matriz de compartida teniendo en cuenta que la matriz compartida va a tener dos elementos m�s por fila y por columna que la matriz original
    int counter = 0; // Contador del n�mero de posiciones con c�lulas vivas
    __shared__ int sharedMatrix[18 * 18];
    if (idx < number_columns && idy < number_rows) 
        //Solo se realizar� si el id del hilo se encuentra dentro de los limites de la matriz (en relaci�n al n�mero de elementos por columna y al n�mero de de elementos por fila)  
    {
        sharedMatrix[indexBlock] = m[index]; //Se introduce el elemento del matriz completa marcado por el �ndice del hilo en la posici�n de la matriz en memoria compartida a partir del �ndice del hilo en el bloque en el que se encuentra
        if (x == 0) // Considera los elementos del bloque que se encuentran en la primera columna del bloque
        {
            if (idx != 0) // Considera aquellos elementos que no se encuentran en la primera columna de la matriz
            {
                sharedMatrix[indexBlock - 1] = m[index - 1]; //Se coloca el elemento que se encuentra a su lado izquierdo
            }
            else // Considera aquellos elementos que s� se encuentran en la primera columna de la matriz
            {
                sharedMatrix[indexBlock - 1] = 'O'; //Se coloca una c�lula muerta al lado izquierdo del elemento
            }
        }
        if (x == widthSharedMatrix - 1)// Considera los elementos del bloque que se encuentran en la �ltima columna del bloque
        {
            if (idx != number_columns - 1)// Considera aquellos elementos que no se encuentran en la �ltima columna de la matriz
            {
                sharedMatrix[indexBlock + 1] = m[index + 1]; //Se coloca el elemento que se encuentra a su lado derecha
            }
            else// Considera aquellos elementos que s� se encuentran en la �ltima columna de la matriz
            {
                sharedMatrix[indexBlock + 1] = 'O'; //Se coloca una c�lula muerta al lado derecho del elemento
            }
        }
        if (y == 0) // Considera los elementos del bloque que se encuentran en la primera fila del bloque
        {
            if (idy != 0) // Considera aquellos elementos que no se encuentran en la primera fila de la matriz
            {
                sharedMatrix[indexBlock - widthSharedMatrix] = m[index - number_columns]; //Se coloca un elemento que se encuentra encima del elemento estudiado actual
            }
            else // Considera aquellos elementos que s� se encuentran en la primera fila de la matriz
            {
                sharedMatrix[indexBlock - widthSharedMatrix] = 'O'; //Se coloca una c�lula muerta encima del elemento estudiado actual
            }
        }
        if (y == widthSharedMatrix - 1) // Considera los elementos del bloque que se encuentran en la �ltima fila del bloque
        {
            if (idy != number_rows) // Considera aquellos elementos que no se encuentran en la �ltima fila de la matriz
            {
                sharedMatrix[indexBlock + widthSharedMatrix] = m[index + number_columns]; //Se coloca un elemento que se encuentra debajo del elemento estudiado actual
            }
            else  // Considera aquellos elementos que s� se encuentran en la �ltima fila de la matriz
            {
                sharedMatrix[indexBlock + widthSharedMatrix] = 'O'; //Se coloca una c�lula muerta debajo del elemento estudiado actual
            }
        }

        if (indexBlock == widthSharedMatrix + 1) // Consideraci�n para el hilo que se encuentra en la primera fila y primera columna del bloque 
        {
            if (index - number_columns - 1 >= 0) //Considera si el �ndice indicado se encuentra superiormente dentro de la matriz
            {
                sharedMatrix[0] = m[index - number_columns - 1]; //Se coloca un elemento en la esquina superior izquierda con el respecto al �ndice del hilo actual estudiado
            }
            else //Considera si el �ndice indicado no se encuentra superiormente dentro de la matriz
            {
                sharedMatrix[0] = 'O'; //Se coloca una c�lula muerta en la esquina superior izquierda con el respecto al �ndice del hilo actual estudiado
            }
        }
        else if (indexBlock == widthSharedMatrix * 2 - 2) // Consideraci�n para el hilo que se encuentra en la primera fila y �ltima columna del bloque 
        {
            if (index - number_columns + 1 >= 0) //Considera si el �ndice indicado se encuentra superiormente dentro de la matriz
            {
                sharedMatrix[widthSharedMatrix - 1] = m[index - number_columns + 1]; //Se coloca un elemento en la esquina superior derecha con el respecto al �ndice del hilo actual estudiado
            }
            else //Considera si el �ndice indicado no se encuentra superiormente dentro de la matriz
            {
                sharedMatrix[widthSharedMatrix - 1] = 'O'; //Se coloca una c�lula muerta en la esquina superior derecha con el respecto al �ndice del hilo actual estudiado
            }
        }
        else if (indexBlock == widthSharedMatrix * (widthSharedMatrix - 2) + 1) // Consideraci�n para el hilo que se encuentra en la �ltima fila y primera columna del bloque 
        {
            if (index + number_columns - 1 < number_columns * number_rows) //Considera si el �ndice indicado se encuentra inferiormente dentro de la matriz
            {
                sharedMatrix[widthSharedMatrix * (widthSharedMatrix - 1)] = m[index + number_columns - 1]; //Se coloca un elemento en la esquina inferior izquierda con el respecto al �ndice del hilo actual estudiado
            }
            else //Considera si el �ndice indicado no se encuentra inferiormente dentro de la matriz
            {
                sharedMatrix[widthSharedMatrix * (widthSharedMatrix - 1)] = 'O'; //Se coloca una c�lula muerta en la esquina inferior izquierda con el respecto al �ndice del hilo actual estudiado
            }
        }
        else if (indexBlock == widthSharedMatrix * (widthSharedMatrix - 1) - 2) // Consideraci�n para el hilo que se encuentra en la �ltima fila y �ltima columna del bloque 
        {
            if (index + number_columns + 1 < number_columns * number_rows) //Considera si el �ndice indicado se encuentra inferiormente dentro de la matriz
            {
                sharedMatrix[widthSharedMatrix * widthSharedMatrix - 1] = m[index + number_columns + 1]; //Se coloca un elemento en la esquina inferior derecha con el respecto al �ndice del hilo actual estudiado
            }
            else //Considera si el �ndice indicado no se encuentra inferiormente dentro de la matriz
            {
                sharedMatrix[widthSharedMatrix * widthSharedMatrix - 1] = 'O'; //Se coloca una c�lula muerta en la esquina inferior derecha con el respecto al �ndice del hilo actual estudiado
            }
        }
    }
    __syncthreads(); // Se realiza la sincronizaci�n de los hilos del mismo bloque

    if (idx < number_columns && idy < number_rows)
    {
        if (sharedMatrix[indexBlock - widthSharedMatrix - 1] == 'X') // Estudia si existe una c�lula viva en la esquina superior izquierda
        {
            counter++;
        }
        if (sharedMatrix[indexBlock - 1] == 'X') // Estudia si existe una c�lula viva en el lateral izquierdo
        {
            counter++;
        }
        if (sharedMatrix[indexBlock - widthSharedMatrix] == 'X') // Estudia si existe una c�lula viva en el lateral superior
        {
            counter++;
        }
        if (sharedMatrix[indexBlock - widthSharedMatrix + 1] == 'X') // Estudia si existe una c�lula viva en la esquina superior derecha
        {
            counter++;
        }
        if (sharedMatrix[indexBlock + 1] == 'X') // Estudia si existe una c�lula viva en el lateral derecho
        {
            counter++;
        }
        if (sharedMatrix[indexBlock + widthSharedMatrix - 1] == 'X') // Estudia si existe una c�lula viva en la esquina inferior izquierda
        {
            counter++;
        }
        if (sharedMatrix[indexBlock + widthSharedMatrix] == 'X') // Estudia si existe una c�lula viva en el lateral inferior
        {
            counter++;
        }
        if (sharedMatrix[indexBlock + widthSharedMatrix + 1] == 'X') // Estudia si existe una c�lula viva en la esquina inferior derecha
        {
            counter++;
        }
        if ((counter == 3) && (sharedMatrix[indexBlock] == 'O')) // Considera si una c�lula muerte se convierte en viva si tiene 3 c�lulas vivas alrededor de ella
        {
            p[index] = 'X';
        }
        else if (((counter < 2) || (counter > 3)) && (sharedMatrix[indexBlock] == 'X')) // Considera si una c�lula viva se convierte en muerte si alrededor de ella hay un n�mero de c�lulas distinto de 2 o 3
        {
            p[index] = 'O';
        }
        else //Considera mantener el estado de la c�lula
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
int get_max_number_threads_block();
int main(int argc, char* argv[])
{
    show_info_gpu_card(); //Muestra las caracter�sticas de la tarjeta gr�fica
    int maxThreads = get_max_number_threads_block(); // Devuelve el n�mero m�ximo de hilos que se pueden ejecutar por bloque
    printf("Comienza el juego de la vida:\n");
    int number_blocks_x = 1; //N�mero de bloques por columnas
    int number_blocks_y = 1; //N�mero de bloques por filas
    int number_rows = 32; // N�mero de elementos por fila
    int number_columns = 32; // N�mero de elementos por columna
    int width_block = 16; // Ancho de bloque
    char execution_mode = 'a'; // Modo de ejecuci�n del programa
    // Condiciones para los casos en los que se est� pasando por terminal una serie de par�metros
    if (argc == 2) //Consideraci�n si solo se pasan dos par�metros por consola
    {
        execution_mode = argv[1][0];
    }
    else if (argc == 3) //Consideraci�n si solo se pasan tres par�metros por consola
    {
        execution_mode = argv[1][0];
        number_rows = atoi(argv[2]);
    }
    else if (argc >= 4) //Consideraci�n si solo se pasan cuatro o m�s par�metros por consola
    {
        execution_mode = argv[1][0];
        number_rows = atoi(argv[2]);
        number_columns = atoi(argv[3]);
    }
    int size = number_rows * number_columns; //Tama�o de la matriz
    int width = number_columns; //Ancho del bloque
    dim3 threads(width_block, width_block); //N�mero de hilos generados por bloques teniendo en cuenta dos dimensiones
    if (size <= maxThreads)
        // Si el tama�o de la matriz es inferior o igual al m�ximo n�mero de hilos por bloque que admite la GPU 
    {
        if ((number_rows % width_block == 0) && (number_columns % width_block == 0))// N�mero de elementos m�ltiplos de 16 tanto en fila como en columna
        {
            number_blocks_x = (number_columns / width_block);
            number_blocks_y = (number_rows / width_block);
        }
        else if (number_rows % width_block == 0)// N�mero de elementos m�ltiplos de 16 en fila 
        {
            number_blocks_x = (number_columns / width_block) + 1;
            number_blocks_y = (number_rows / width_block);
        }
        else if (number_columns % width_block == 0)// N�mero de elementos m�ltiplos de 16 en columna
        {
            number_blocks_x = (number_columns / width_block);
            number_blocks_y = (number_rows / width_block) + 1;
        }
        else
        {
            number_blocks_x = (number_columns / width_block) + 1;
            number_blocks_y = (number_rows / width_block) + 1;
        }
        dim3 nBlocks(number_blocks_x, number_blocks_y); //N�meros de bloques elaborados en un grid de dos dimensiones
        operation(size, width, nBlocks, threads, number_columns, number_rows, execution_mode, width_block);
    }
    else // Si el tama�o de la matriz es mayor al m�ximo n�mero de hilos por bloque que admite la GPU 
    {
        printf("No son validas las dimensiones introducidas para la matriz.\n");
    }
    getchar();
    getchar();
    return 0;
}

void operation(int size, int width, dim3 nBlocks, dim3 nThreads, int number_columns, int number_rows, char execution_mode, int width_block)
// Imprime las distintas matrices del juego de acuerdo al avance del mismo
{
    int counter = 1; //Contador de n�mero de paso de matriz del juego
    char* a = (char*)malloc(size * sizeof(char));
    char* b = (char*)malloc(size * sizeof(char));
    generate_matrix(a, size, nBlocks, nThreads, number_columns, number_rows, width_block);
    printf("Situacion Inicial:\n");
    for (int i = 0; i < size; i++)//Representaci�n matriz inicial
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
    while (execution_mode == 'm' || execution_mode == 'a')
    {
        //Va alternando las matrices para realizar las operaciones del juego considerando una como matriz inicial y otra como final
        if (counter % 2 == 1)
        {
            step_life(a, b, width, size, nBlocks, nThreads, number_columns, number_rows, width_block);//Se realiza un paso del juego de la vida
            printf("Matriz paso %d:\n", counter);
            for (int i = 0; i < size; i++)//Representaci�n matriz en un paso en concreto
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
            step_life(b, a, width, size, nBlocks, nThreads, number_columns, number_rows, width_block);//Se realiza un paso del juego de la vida
            printf("Matriz paso %d:\n", counter);
            for (int i = 0; i < size; i++)//Representaci�n matriz en un paso en concreto
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
        if (execution_mode == 'm') //Si el modo seleccionado no es autom�tico para hasta que el usuario pulse una tecla
        {
            getchar();
        }
    }
    if (execution_mode != 'm' && execution_mode != 'a')
    {
        printf("El modo de ejecucion del programa es incorrecto.\n");
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
    int numElem = generate_random(1, size * 0.15);// Genera un n�mero aleatorio de m�xima n�mero de c�lulas vivas en la etapa inicial siendo el m�ximo un 15% del m�ximo n�mero de casillas
    cudaMalloc((void**)&m_d, size * sizeof(char));
    cudaMemcpy(m_d, m, size * sizeof(char), cudaMemcpyHostToDevice);
    prepare_matrix << <nBlocks, nThreads >> > (m_d, number_columns, number_rows, width_block); //Prepara la matriz con todas las casillas con c�lulas muertas
    make_rand << <1, numElem >> > (seed, m_d, size); // Va colocando de forma aleatoria c�lulas vivas en las casillas de la matriz
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

int generate_random(int min, int max) // Genera un n�mero aleatorio entre un m�nimo y un m�ximo
{
    srand(time(NULL));
    int randNumber = rand() % (max - min) + min;
    return randNumber;
}

int get_max_number_threads_block()// Devuelve el n�mero m�ximo de hilos que se pueden ejecutar por bloque
{
    cudaDeviceProp prop;

    int count;
    //Obtenci�n n�mero de dispositivos compatibles con CUDA
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    return prop.maxThreadsPerBlock;

}

void show_info_gpu_card() // Muestra las caracter�sticas de la tarjeta gr�fica usada
{
    cudaDeviceProp prop;

    int count;
    //Obtenci�n n�mero de dispositivos compatibles con CUDA
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    printf("Numero de dispositivos compatibles con CUDA: %d.\n", count);

    //Obtenci�n de caracter�sticas relativas a cada dispositivo
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