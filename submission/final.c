#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "mpi.h"
#define INPUT_FILE_NAME "Input.txt"
#define OUTPUT_FILE_NAME "Output.txt"

typedef struct{
	int id;
	int size;
	int* values;
} Picture;

typedef struct{
	int id;
	int size;
	int* values;
} Object;

float matching (int* sizeOfObject, int* sizeOfPicture, int* picture, int* object){
	float diff=0;
	int p,o;

	for (int i=0 ; i< *sizeOfObject; i++){
		for (int j=0 ; j< *sizeOfObject; j++){
			p = picture[i*(*sizeOfPicture) + j];
			o = object[i*(*sizeOfObject) + j];
			diff += fabs((float)(p-o)/p);
		}
	}

	return diff;
}

int main(int argc, char* argv[]){
	int  my_rank; /* rank of process */
	int  p;       /* number of processes */
	FILE* input_file, *output_file;
	float match_value; // shared
	int num_of_pictures=0, num_of_objects=0;
	Picture* pictures;
	Object* objects;
	Object obj;
	MPI_Status status;
	Picture my_picture;
	int flag=0;

	/* start up MPI */
	MPI_Init(&argc, &argv);

	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	if (my_rank == 0) {
		if ((input_file = fopen(INPUT_FILE_NAME, "r")) == 0) {
			printf("cannot open file %s for reading\n", INPUT_FILE_NAME);
		}

		// scan match value
		fscanf(input_file, "%f ", &match_value);

		// scan number of pictures and allocate space for N pictures
		fscanf(input_file, "%d ", &num_of_pictures);
		pictures = (Picture*)malloc(num_of_pictures * sizeof(Picture));

		for (int j=0 ; j< num_of_pictures; j++) {
			// scan picture ID
			fscanf(input_file, "%d ", &pictures[j].id);
			int id = pictures[j].id;

			// scan picture size and allocate values space
			fscanf(input_file, "%d ", &pictures[id].size);
			pictures[id].values = (int*)malloc(pictures[id].size * pictures[id].size * sizeof(int));
			int size = pictures[id].size;

			for (int i=0 ; i< size ; i++){
				for (int k=0 ; k< size ; k++) {
					fscanf(input_file, "%d ", &pictures[id].values[i*size+k]);
				}
			}
		}

		//scan number of objects
		fscanf(input_file, "%d ", &num_of_objects);
		objects = (Object*)malloc(num_of_objects * sizeof(Object));

		for (int j=0 ; j< num_of_objects; j++) {
			// scan object ID
			fscanf(input_file, "%d ", &objects[j].id);

			// scan picture size
			fscanf(input_file, "%d ", &objects[j].size);
			objects[j].values = (int*)malloc(objects[j].size * objects[j].size * sizeof(int));

			for (int i=0 ; i< objects[j].size ; i++){
				for (int k = 0 ; k<objects[j].size ; k++) {
					fscanf(input_file, "%d ", &objects[objects[j].id].values[i*objects[objects[j].id].size+k]);
				}
			}
		}

		fclose(input_file);

		// send all objects to all processes
		for (int j = 1 ; j< p ; j++){
			MPI_Send(&num_of_objects, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
			for (int i=0 ; i< num_of_objects; i++){
				MPI_Send(&objects[i].id, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
				MPI_Send(&objects[i].size, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
				MPI_Send(objects[i].values, objects[i].size*objects[i].size, MPI_INT, j, 0, MPI_COMM_WORLD);
			}
		}

		// send each picture to different process
		for (int i=0 ; i< num_of_pictures; i++){
			if (i > p ) i  = i % p ;
			MPI_Send(&pictures[i].id, 1, MPI_INT, i+1, 0, MPI_COMM_WORLD);
			MPI_Send(&pictures[i].size, 1, MPI_INT, i+1, 0, MPI_COMM_WORLD);
			MPI_Send(pictures[i].values, pictures[i].size*pictures[i].size, MPI_INT, i+1, 0, MPI_COMM_WORLD);
		}
	}
	else {
		// open file for writing
		if ((output_file = fopen(OUTPUT_FILE_NAME,"w")) == NULL){
		       printf("Error! opening file");
		       // Program exits if the file pointer returns NULL.
		       exit(1);
		   }

		// each process receives all objects
		MPI_Recv(&num_of_objects, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		objects = (Object*)malloc(num_of_objects * sizeof(Object));

		for (int i=0 ; i<num_of_objects; i++){
			MPI_Recv(&objects[i].id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(&objects[i].size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			objects[i].values = (int*)malloc(objects[i].size * objects[i].size * sizeof(int));
			MPI_Recv(objects[i].values, objects[i].size * objects[i].size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		}

		// each process receives its picture
		MPI_Recv(&my_picture.id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&my_picture.size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		my_picture.values = (int*)malloc(sizeof(int) * my_picture.size * my_picture.size);
		MPI_Recv(my_picture.values, my_picture.size * my_picture.size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

#pragma omp parallel shared(num_of_objects, objects, flag, output_file)
		{
			// loop on all objects
			for (int o = 0 ; o < num_of_objects; o++){
				obj = objects[o];

#pragma omp for schedule(dynamic,1) nowait
				// loop on picture to calculate diff on every position inside borders
				for (int j =0 ;j< my_picture.size - obj.size +1; j++)
					if (flag != 1)
						for (int k=0 ; k< my_picture.size - obj.size +1;k ++)
							if (matching(&obj.size, &my_picture.size, &(my_picture.values[j*my_picture.size+k]), &(obj.values[0])) == 0) {
								fprintf(output_file,"Picture %d: found Object %d in Position (%d,%d)\n", my_picture.id, obj.id, j, k);
								flag = 1;
							}
			}
#pragma omp single
			if (flag == 0){
				fprintf(output_file,"Picture %d: No objects were found\n", my_picture.id);

			}
		}
	}



	/* shut down MPI */
	MPI_Finalize();

	return 0;
}

