/* rc4.h */ 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <math.h>
#include <Windows.h>

//This is binary so all characters are valid
#define START_CHARACTER 0x00
#define END_CHARACTER 0xFF
#define KEY (END_CHARACTER-START_CHARACTER+1)

#define BLOCK_NUM 32
#define MAX_THREAD_NUM 256

// space is actually enough with 10, the reason for taking 20 is mainly to avoid bank conflicts
#define MEMORY_PER_THREAD 276
#define MAX_KEY_LENGTH 5 //max key length
#define STATE_LEN	256
#define MAX_KNOWN_STREAM_LEN 5

__constant__ unsigned long long maxNum=0x10000000000;
__constant__ unsigned int maxKeyLen=MAX_KEY_LENGTH;
__constant__ unsigned int keyNum=KEY;
__constant__ unsigned int start=START_CHARACTER;
__constant__ unsigned int memory_per_thread=MEMORY_PER_THREAD;
__constant__ unsigned char knownStreamLen_device;
//__constant__ unsigned char initialArray_device[STATE_LEN];
__constant__ unsigned char knowStream_device[MAX_KNOWN_STREAM_LEN];


extern __shared__ unsigned char shared_mem[];

__device__ __host__ unsigned char rc4_single(unsigned char*x, unsigned char * y, unsigned char *s_box);
__device__ __host__ static void swap_byte(unsigned char *a, unsigned char *b);
__device__ bool device_isKeyRight(const unsigned char *known_stream, int known_len,unsigned char *validateKey,int key_len);
__device__ __host__ unsigned char rc4_single(unsigned char*x, unsigned char * y, unsigned char *s_box);
void prepare_key(unsigned char *key_data_ptr, int key_data_len,unsigned char *s_box);
void rc4(unsigned char *buffer_ptr, int buffer_len, unsigned char *s_box);
/************************************************************************/
/* the data type is unsigned char,so the %256 is no necessary           */
/************************************************************************/

/**
 * \brief swap two bytes
 */
__device__ __host__ static void swap_byte(unsigned char *a, unsigned char *b) 
{ 
	unsigned char swapByte;  

	swapByte = *a;  
	*a = *b;      
	*b = swapByte; 
}

__device__ bool device_isKeyRight(const unsigned char *validateKey, const int key_len, volatile bool* found) 
{ 
	//KSA
  unsigned char* state = (shared_mem + (memory_per_thread * threadIdx.x) + maxKeyLen);
	//unsigned char state[STATE_LEN];
	unsigned char index1=0, index2=0;
	short counter=0;

  //if(*found) asm("exit;");   
  //memcpy(state, "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x20\x21\x22\x23\x24\x25\x26\x27\x28\x29\x2a\x2b\x2c\x2d\x2e\x2f\x30\x31\x32\x33\x34\x35\x36\x37\x38\x39\x3a\x3b\x3c\x3d\x3e\x3f\x40\x41\x42\x43\x44\x45\x46\x47\x48\x49\x4a\x4b\x4c\x4d\x4e\x4f\x50\x51\x52\x53\x54\x55\x56\x57\x58\x59\x5a\x5b\x5c\x5d\x5e\x5f\x60\x61\x62\x63\x64\x65\x66\x67\x68\x69\x6a\x6b\x6c\x6d\x6e\x6f\x70\x71\x72\x73\x74\x75\x76\x77\x78\x79\x7a\x7b\x7c\x7d\x7e\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff",STATE_LEN);
  // I've tried a couple of ways to speed up loading this (packing, copying from global literals etc.)
  // and this is the most efficent short of hand coding assembly so that each state staying in a register
	for(counter = 0; counter < STATE_LEN; counter++)
		state[counter] = counter;
  //memcpy(state, initialArray_device, STATE_LEN);

	//if(*found) asm("exit;");

	for(counter = 0; counter < STATE_LEN; counter++)      
	{             
		index2 = (validateKey[index1] + state[counter] + index2);            
		swap_byte(&state[counter], &state[index2]);
		index1 = (index1 + 1) % key_len;  
	} 

	//if(*found) asm("exit;");

	//PRGA
	index1=0, index2=0, counter=0; 
	for (; counter < knownStreamLen_device; counter++)
	{
		if(knowStream_device[counter] != rc4_single(&index1,&index2,state))
			return false;
	}

	//if(*found) asm("exit;");

	return true;
} 
/**
 * \brief rc4 encryption and decryption function
 * 
 * \param buffer_ptr,the data string to encryption 
 * \param buffer_len,the data length
 * \param key,rc4's s-box and the two key pointers,this was used to encryption the data
 *
 * \return void
**/
__device__ __host__ unsigned char rc4_single(unsigned char* x, unsigned char* y, unsigned char* s_box) 
{  
	unsigned char* state, xorIndex;

	state = &s_box[0];

	*x = (*x + 1);            
	*y = (state[*x] + *y);
	swap_byte(&state[*x], &state[*y]);

	xorIndex = (state[*x] + state[*y]);

	return  state[xorIndex];        
}

/**
 * \brief rc4 s-box init
 * 
 * \param key_data_ptr,the encryption key
 * \param key_data_len,the encryption key length,less than 256
 * \param key,rc4's s-box and the key two pointers
 *
 * \return void
**/
void prepare_key(unsigned char *key_data_ptr, int key_data_len,unsigned char *s_box) 
{ 
	unsigned char index1=0, index2=0, * state; 
	short counter;    

	state = &s_box[0];        
	for(counter = 0; counter < STATE_LEN; counter++)          
		state[counter] = counter;   
	for(counter = 0; counter < STATE_LEN; counter++)      
	{             
		index2 = (key_data_ptr[index1] + state[counter] + index2);            
		swap_byte(&state[counter], &state[index2]);          

		index1 = (index1 + 1) % key_data_len;  
	}      
} 

void rc4(unsigned char *buffer_ptr, int buffer_len, unsigned char *s_box) 
{  
	unsigned char x=0, y=0, * state;
	short counter; 

	state = &s_box[0];        
	for(counter = 0; counter < buffer_len; counter ++)
	{  
		buffer_ptr[counter] ^= rc4_single(&x,&y,state);        
	}            
} 