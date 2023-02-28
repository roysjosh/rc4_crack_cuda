#include "rc4.h"

/************************************************************************/
/* 
The original idea is to obtain one key at a time, decrypt the corresponding ciphertext, and see if the resulting plaintext satisfies a certain condition.
But the process requires too many intermediate variables, and on second thought, the plaintext and ciphertext are heterogeneous or related, so the known plaintext
If the text and the ciphertext are dissimilar, we can get the value of some position of the key stream. This saves a lot of space~~
*/
/************************************************************************/

__device__ void genKey(unsigned char* keyStartAddr, unsigned long long myKeyNum, int* key_len)
{
/*	char p = maxKeyLen - 1;
	while (myKeyNum && p >=0) {
		keyStartAddr[p--] = (myKeyNum - 1) % keyNum + start;
		myKeyNum = (myKeyNum - 1) / keyNum;
	}
	*key_len = (maxKeyLen - p - 1);
	return keyStartAddr + p + 1;
*/
  size_t i = 0;
  while (myKeyNum && i < maxKeyLen)
  {
    keyStartAddr[i++] = (myKeyNum - 1) % keyNum + start;
    myKeyNum = (myKeyNum - 1) / keyNum;
  }
  *key_len = (i);
}

__global__ void crackRc4Kernel(unsigned char* key, volatile bool* found)
{
	int keyLen = 0;
	const unsigned long long totalThreadNum = gridDim.x * blockDim.x;
	const unsigned long long keyNum_per_thread = maxNum / totalThreadNum;
	unsigned long long myKeyNum = (threadIdx.x + blockIdx.x * blockDim.x);
	bool justIt;
	for (unsigned long long i=0; i <= keyNum_per_thread; myKeyNum += totalThreadNum, ++i)
	{
		//vKey is a pointer to share_memory
		unsigned char* vKey = (shared_mem + memory_per_thread * threadIdx.x);
    genKey(vKey, myKeyNum, &keyLen);
    // Pad with nulls
    while (keyLen < maxKeyLen)
    {
      vKey[keyLen++] = '\x00';
    }

		justIt=device_isKeyRight(vKey,keyLen,found);

		//Exit if one of the other blocks found it
		if(*found) asm("exit;");

		// the current key is not the requested one
		if (justIt)
    {
      // Find the matching key, write it to Host, save the data, modify found, and exit the program
      *found = true;
      memcpy(key, vKey, keyLen);
      key[keyLen]=0;
      __threadfence();
      asm("exit;");
      break;
    }
	}
}

void cleanup(unsigned char *key_dev, bool* found_dev)
{
  cudaFree(key_dev);
  cudaFree(found_dev);
  return;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t crackRc4WithCuda(unsigned char* knownKeyStream_host, int knownStreamLen_host, unsigned char*key, bool*found)
{
	cudaError_t cudaStatus;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    return cudaStatus;
	}

	unsigned char *key_dev ;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	cudaStatus = cudaMalloc((void**)&key_dev, (MAX_KEY_LENGTH + 1) * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(key_dev);
    return cudaStatus;
  }

  bool* found_dev;

	cudaStatus = cudaMalloc((void**)&found_dev, sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cleanup(key_dev, found_dev);
    return cudaStatus;
  }

	//Check if the key variable is found
	cudaStatus = cudaMemcpy(found_dev, found, sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cleanup(key_dev, found_dev);
    return cudaStatus;
  }

	//Copy constant memory
	cudaStatus = cudaMemcpyToSymbol(knowStream_device, knownKeyStream_host, sizeof(unsigned char) *knownStreamLen_host);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol stream failed!");
		cleanup(key_dev, found_dev);
    return cudaStatus;
  }

	cudaStatus = cudaMemcpyToSymbol((const void *) &knownStreamLen_device, (const void *) &knownStreamLen_host, sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol streamlen failed!");
		cleanup(key_dev, found_dev);
    return cudaStatus;
  }

	// Launch a kernel on the GPU with one thread for each element.
	int threadNum = floor( (double) (prop.sharedMemPerBlock / MEMORY_PER_THREAD) ), share_memory = prop.sharedMemPerBlock;
	if(threadNum > MAX_THREAD_NUM )
  {
		threadNum = MAX_THREAD_NUM;
		share_memory = threadNum * MEMORY_PER_THREAD;
	}

	crackRc4Kernel<<<BLOCK_NUM, threadNum, share_memory>>>(key_dev, found_dev);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cleanup(key_dev, found_dev);
    return cudaStatus;
  }

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		cleanup(key_dev, found_dev);
    return cudaStatus;
  }

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(key, key_dev, (MAX_KEY_LENGTH+1) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cleanup(key_dev, found_dev);
    return cudaStatus;
  }

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(found, found_dev,  sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cleanup(key_dev, found_dev);
    return cudaStatus;
  }

	return cudaStatus;
}

int main(int argc, char *argv[])
{

	unsigned char* s_box = (unsigned char*)malloc(sizeof(unsigned char)*256);

	//Key
	unsigned char encryptKey[host_max_key] = "KeyK";
  unsigned char buffer[] = "In cryptography, RC4 (Rivest Cipher 4, also known as ARC4 or ARCFOUR, meaning Alleged RC4, see below) is a stream cipher.";
  size_t buffer_len = strlen( (char*)buffer);
  size_t key_len = strlen( (char*)encryptKey);
  // Pad the key
  while (key_len < host_max_key)
  {
    encryptKey[key_len++] = '\x00';
  }
	
  prepare_key(encryptKey, key_len, s_box);
	rc4(buffer, buffer_len, s_box);	
  
	unsigned char knownPlainText[] = "In cr";
	int known_p_len = strlen( (char*)knownPlainText);
	unsigned char* knownKeyStream = (unsigned char*) malloc(sizeof(unsigned char) * known_p_len);
	for (int i = 0; i < known_p_len; i++)
	{
		knownKeyStream[i] = knownPlainText[i] ^ buffer[i];
	}

	unsigned char* key = (unsigned char*) malloc( sizeof(unsigned char) * (MAX_KEY_LENGTH + 1));

	cudaEvent_t start,stop;
	cudaError_t cudaStatus = cudaEventCreate( &start);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventCreate(start) failed!");
		return 1;
	}
	cudaStatus=cudaEventCreate( &stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventCreate(stop) failed!");
		return 1;
	}

	cudaStatus=cudaEventRecord(start, 0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventRecord(start) failed!");
		return 1;
	}

	bool found=false;
	cudaStatus = crackRc4WithCuda(knownKeyStream, known_p_len , key, &found);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	cudaStatus=cudaEventRecord(stop,0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventRecord(stop) failed!");
		return 1;
	}

	cudaStatus=cudaEventSynchronize(stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventSynchronize failed!");
		return 1;
	}
	float useTime;
	cudaStatus = cudaEventElapsedTime(&useTime,start,stop);
	useTime /= 1000;
	printf("The time we used was:%fs\n",useTime);
	if (found)
	{
		printf("The right key has been found.The right key is:%s\n",key);
		prepare_key(key, key_len, s_box);
		rc4(buffer, buffer_len, s_box);
		printf ("\nThe clear text is:\n%s\n", buffer);
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	free(key);
	free(knownKeyStream);
	free(s_box);
	cudaThreadExit();
	return 0;
}



