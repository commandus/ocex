#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <math.h>
#include <CL/opencl.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

#define DATA_SIZE (1024)

// Simple compute kernel which computes the square of an input array 
//
const char *KernelSource = "\n" \
"__kernel void square(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] * input[i];                                \n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////


std::string GetPlatformName(cl_platform_id &id) 
{
    size_t sz;
    clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &sz);
    std::string r;
    r.resize(sz);
    clGetPlatformInfo(id, CL_PLATFORM_NAME, sz, (void *) r.data(), &sz);
    return r;
}

std::string GetDeviceName(cl_device_id &id)
{
    size_t sz;
    clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &sz);
    std::string r;
    r.resize(sz);
    clGetDeviceInfo(id, CL_DEVICE_NAME, sz, (void*)r.data(), &sz);

    cl_uint units;
    cl_device_type devtype;
    size_t lmem;
    cl_uint dims;
    size_t wisz[3];
    size_t wgsz;
    size_t gmsz;
    clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, 0);
    clGetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(devtype), &devtype, 0);
    clGetDeviceInfo(id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, 0);
    clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(dims), &dims, 0);
    clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(wisz), &wisz, 0);
    clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wgsz), &wgsz, 0);

    std::stringstream ss;
    ss << r
        << " compute units: " << units
        << " type: " << devtype << " (GPU = " << CL_DEVICE_TYPE_GPU
        << ") local mem size: " << lmem
        << " max item dimensions: " << dims
        << " max item sizes: " << wisz
        << " max work group size: " << wgsz;
    return ss.str();
}

int getDeviceList(
    std::vector<cl_platform_id> &platformIds,
    std::vector<cl_device_id> &deviceIds
)
{
    // http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clGetPlatformIDs.html
    cl_uint platformIdCount = 0;
    clGetPlatformIDs(0, nullptr, &platformIdCount);

    if (platformIdCount == 0) {
        std::cerr << "No OpenCL platform found" << std::endl;
        return 0;
    }
    else {
        std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
    }

    platformIds.resize(platformIdCount);
    clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

    int r = 0;
    for (cl_uint i = 0; i < platformIdCount; ++i) {
        std::cout << GetPlatformName(platformIds[i]) << std::endl;
        // http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clGetDeviceIDs.html
        cl_uint deviceIdCount = 0;
        clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_GPU, 0, nullptr,
            &deviceIdCount);

        if (deviceIdCount == 0) {
            std::cerr << "No OpenCL devices found" << std::endl;
            continue;
        }
        else {
            std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
        }
        r += deviceIdCount;
        deviceIds.resize(deviceIdCount);
        clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_GPU, deviceIdCount,
            deviceIds.data(), nullptr);

        for (cl_uint d = 0; d < deviceIdCount; ++d) {
            std::cout << GetDeviceName(deviceIds[d]) << std::endl;
        }
    }
    return r;
}

int main(int argc, char** argv)
{
    int err;                            // error code returned from api calls
      
    float data[DATA_SIZE];              // original data set given to device
    float results[DATA_SIZE];           // results returned from device
    unsigned int correct;               // number of correct results returned

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    
    // Fill our data set with random float values
    //
    int i = 0;
    unsigned int count = DATA_SIZE;
    for(i = 0; i < count; i++)
        data[i] = rand() / (float)RAND_MAX;

    std::vector<cl_platform_id> platformIds;
    std::vector<cl_device_id> deviceIds;
 
    int dcount = getDeviceList(platformIds, deviceIds);

    // Connect to a compute device
    //
    const cl_device_id &device_id = deviceIds[0];
    
    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "square", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
    
    // Write our data set into the input array in device memory 
    //
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    //
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    // Validate our results
    //
    correct = 0;
    for(i = 0; i < count; i++)
    {
        if(results[i] == data[i] * data[i])
            correct++;
    }
    
    // Print a brief summary detailing the results
    //
    printf("Computed '%d/%d' correct values!\n", correct, count);
    
    // Shutdown and cleanup
    //
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}

